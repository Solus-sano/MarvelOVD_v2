from typing import Dict, List, Tuple, Union
import torch
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY
from detectron2.structures import Boxes, Instances
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, _log_classification_stats
from .fast_rcnn_ssod import SSODFastRCNNOutputLayers
import torch.nn.functional as F
from torchvision.ops import box_iou

__all__ = ["SSODFastRCNNOutputLayers_2"]

class SSODFastRCNNOutputLayers_2(SSODFastRCNNOutputLayers):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        logits, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        logits = logits.split(num_inst_per_image, dim=0)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            logits,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )
        
def fast_rcnn_inference(
        boxes: List[torch.Tensor],
        scores: List[torch.Tensor],
        logits: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
        score_thresh: float,
        nms_thresh: float,
        topk_per_image: int,
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    # result_per_image = [
    #     fast_rcnn_inference_single_image(
    #         boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
    #     )
    #     for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    # ]
    # return [x[0] for x in result_per_image], [x[1] for x in result_per_image]
    Instances_lst, inds_lst = [], []
    for scores_per_image, logits_per_image, boxes_per_image, image_shape in zip(scores, logits, boxes, image_shapes):
        _, inds = fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        
        new_cls_scores, new_boxes_per_image = faster_update_logic_after_nms_3(
            logits_per_image, boxes_per_image, inds
        )
        results, kept_indices = fast_rcnn_inference_single_image(
            new_boxes_per_image, new_cls_scores, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        Instances_lst.append(results)
        inds_lst.append(kept_indices)
    return Instances_lst, inds_lst


def fast_rcnn_inference_single_image(
        boxes,
        scores,
        image_shape: Tuple[int, int],
        score_thresh: float,
        nms_thresh: float,
        topk_per_image: int,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)

    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    boxes = boxes.repeat(1, scores.shape[1], 1)
    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


def faster_update_logic_after_nms_2(
    cls_logits: torch.Tensor, 
    dt_boxes: torch.Tensor, 
    inds: torch.Tensor,
    beta: float = 0.5,
    sigma: float = 0.05
) -> torch.Tensor:
    """
    N=1000(nms前数量), C=80, n<=100(nms后数量)
    cls_logits: (N, C+1)
    dt_boxes: (N, 4)
    inds: (n,)   (已经全部//(class_num), 数值范围是 0~N-1)
    """
    set_inds = list(set(inds)) # len = n1
    dt_boxes_after_nms, cls_logits_after_nms = dt_boxes[set_inds,:], cls_logits[set_inds,:]
    cls_score_after_nms = F.softmax(cls_logits_after_nms,dim=-1)[:,:-1].max(dim=-1)[0]
    # 仿照multiclass_nms的方法处理, 将框的每个类别都看作是一个框, 调整每个类别logit的值, 但是选取相邻框时只考虑logit最大的那个
    num_classes = cls_logits.size(1) - 1 # 移除背景类
    logits, background_logits = cls_logits_after_nms[:,:-1], cls_logits_after_nms[:,-1:] # 移除背景类
    cls_logits_after_nms_label = torch.argmax(logits, dim=1)
    labels = torch.arange(num_classes, dtype=torch.long, device=logits.device)
    labels = labels.view(1, -1).expand_as(logits)
    
    novel = torch.tensor([4,5,11,12,15,16,21,23,27,29,32,34,45,47,54,58,63]).to(cls_logits_after_nms_label.device)
    is_novel = torch.isin(cls_logits_after_nms_label, novel)
    novel_idx, base_idx = is_novel.nonzero(as_tuple=False).squeeze(1), (is_novel == 0).nonzero(as_tuple=False).squeeze(1)
    
    scores_base, scores_novel = cls_score_after_nms[base_idx], cls_score_after_nms[novel_idx] 
    logits_base, logits_novel = logits[base_idx,:], logits[novel_idx,:]
    if len(logits_base.shape) == 1:
        logits_base = logits_base.unsqueeze(0)
    if len(logits_novel.shape) == 1:
        logits_novel = logits_novel.unsqueeze(0)
    background_logits_base, background_logits_novel = background_logits[base_idx,:], background_logits[novel_idx,:]
    if len(background_logits_base.shape) == 1:
        background_logits_base = background_logits_base.unsqueeze(0)
    if len(background_logits_novel.shape) == 1:
        background_logits_novel = background_logits_novel.unsqueeze(0)
    bboxes_base, bboxes_novel = dt_boxes_after_nms[base_idx], dt_boxes_after_nms[novel_idx]
    labels_base, labels_novel = labels[base_idx], labels[novel_idx]
    base_cnt, novel_cnt = len(base_idx.cpu().tolist()), len(novel_idx.cpu().tolist())
    
    bboxes_base_flatten, bboxes_novel_flatten = bboxes_base.unsqueeze(1).expand(base_cnt, num_classes, 4), bboxes_novel.unsqueeze(1).expand(novel_cnt, num_classes, 4)
    # (n1*C, 4), (n1*C, ), (n1*C, )
    bboxes_base_flatten, logits_base_flatten, labels_base_flatten = bboxes_base_flatten.reshape(-1, 4), logits_base.reshape(-1), labels_base.reshape(-1)
    bboxes_novel_flatten, logits_novel_flatten, labels_novel_flatten = bboxes_novel_flatten.reshape(-1, 4), logits_novel.reshape(-1), labels_novel.reshape(-1)
    
    # cls_scores = update_logit(
    #     torch.concat([bboxes_base_flatten, bboxes_novel_flatten], dim=0),
    #     torch.concat([logits_base_flatten, logits_novel_flatten], dim=0),
    #     torch.concat([labels_base_flatten, labels_novel_flatten], dim=0),
    #     torch.concat([bboxes_base, bboxes_novel], dim=0),
    #     torch.concat([logits_base, logits_novel], dim=0),
    #     torch.concat([scores_base, scores_novel], dim=0),
    #     torch.concat([background_logits_base, background_logits_novel], dim=0)
    # )
    # return cls_scores, torch.concat([bboxes_base, bboxes_novel], dim=0)
    new_cls_scores_base = update_logit(
        bboxes_base_flatten,
        logits_base_flatten,
        labels_base_flatten,
        bboxes_base,
        logits_base,
        scores_base,
        background_logits_base,
        beta,
        sigma
    )
    new_cls_scores_novel = update_logit(
        bboxes_novel_flatten,
        logits_novel_flatten,
        labels_novel_flatten,
        torch.concat([bboxes_base, bboxes_novel], dim=0),
        torch.concat([logits_base, logits_novel], dim=0),
        torch.concat([scores_base, scores_novel], dim=0),
        background_logits_novel,
        beta,
        sigma
    )
    return torch.concat([new_cls_scores_base,new_cls_scores_novel], dim=0), torch.concat([bboxes_base, bboxes_novel], dim=0)

def faster_update_logic_after_nms_3(
    cls_logits: torch.Tensor, 
    dt_boxes: torch.Tensor, 
    inds,
    beta: float = 0.5,
    sigma: float = 0.0
) -> torch.Tensor:
    """
    N=1000(nms前数量), C=80, n<=100(nms后数量)
    cls_logits: (N, C+1)
    dt_boxes: (N, 4)
    inds: (n,)   (已经全部//(class_num), 数值范围是 0~N-1)
    """
    set_inds = list(set(inds)) # len = n1
    dt_boxes_after_nms, cls_logits_after_nms = dt_boxes[set_inds,:], cls_logits[set_inds,:]
    cls_score_after_nms = F.softmax(cls_logits_after_nms,dim=-1)[:,:-1].max(dim=-1)[0]
    # 仿照multiclass_nms的方法处理, 将框的每个类别都看作是一个框, 调整每个类别logit的值, 但是选取相邻框时只考虑logit最大的那个
    num_classes = cls_logits.size(1) - 1 # 移除背景类
    logits, background_logits = cls_logits_after_nms[:,:-1], cls_logits_after_nms[:,-1:] # 移除背景类
    cls_logits_after_nms_label = torch.argmax(logits, dim=1)
    labels = torch.arange(num_classes, dtype=torch.long, device=logits.device)
    labels = labels.view(1, -1).expand_as(logits)
    
    novel = torch.tensor([4,5,11,12,15,16,21,23,27,29,32,34,45,47,54,58,63]).to(cls_logits_after_nms_label.device)
    is_novel = torch.isin(cls_logits_after_nms_label, novel)
    novel_idx, base_idx = is_novel.nonzero(as_tuple=False).squeeze(1), (is_novel == 0).nonzero(as_tuple=False).squeeze(1)
    
    novel_base_idx_to_init_idx = {b:a for a,b in enumerate(base_idx.tolist() + novel_idx.tolist())}
    reverse_idx = torch.tensor([novel_base_idx_to_init_idx[i] for i in range(len(novel_base_idx_to_init_idx))]).to(logits.device)
    
    scores_base, scores_novel = cls_score_after_nms[base_idx], cls_score_after_nms[novel_idx] 
    logits_base, logits_novel = logits[base_idx,:], logits[novel_idx,:]
    if len(logits_base.shape) == 1:
        logits_base = logits_base.unsqueeze(0)
    if len(logits_novel.shape) == 1:
        logits_novel = logits_novel.unsqueeze(0)
    background_logits_base, background_logits_novel = background_logits[base_idx,:], background_logits[novel_idx,:]
    if len(background_logits_base.shape) == 1:
        background_logits_base = background_logits_base.unsqueeze(0)
    if len(background_logits_novel.shape) == 1:
        background_logits_novel = background_logits_novel.unsqueeze(0)
    bboxes_base, bboxes_novel = dt_boxes_after_nms[base_idx], dt_boxes_after_nms[novel_idx]
    labels_base, labels_novel = labels[base_idx], labels[novel_idx]
    base_cnt, novel_cnt = len(base_idx.cpu().tolist()), len(novel_idx.cpu().tolist())
    
    bboxes_base_flatten, bboxes_novel_flatten = bboxes_base.unsqueeze(1).expand(base_cnt, num_classes, 4), bboxes_novel.unsqueeze(1).expand(novel_cnt, num_classes, 4)
    # (n1*C, 4), (n1*C, ), (n1*C, )
    bboxes_base_flatten, logits_base_flatten, labels_base_flatten = bboxes_base_flatten.reshape(-1, 4), logits_base.reshape(-1), labels_base.reshape(-1)
    bboxes_novel_flatten, logits_novel_flatten, labels_novel_flatten = bboxes_novel_flatten.reshape(-1, 4), logits_novel.reshape(-1), labels_novel.reshape(-1)
    
    # cls_scores = update_logit(
    #     torch.concat([bboxes_base_flatten, bboxes_novel_flatten], dim=0),
    #     torch.concat([logits_base_flatten, logits_novel_flatten], dim=0),
    #     torch.concat([labels_base_flatten, labels_novel_flatten], dim=0),
    #     torch.concat([bboxes_base, bboxes_novel], dim=0),
    #     torch.concat([logits_base, logits_novel], dim=0),
    #     torch.concat([scores_base, scores_novel], dim=0),
    #     torch.concat([background_logits_base, background_logits_novel], dim=0)
    # )
    # return cls_scores, torch.concat([bboxes_base, bboxes_novel], dim=0)
    new_cls_scores_base = update_logit(
        bboxes_base_flatten,
        logits_base_flatten,
        labels_base_flatten,
        bboxes_base,
        logits_base,
        scores_base,
        background_logits_base,
        beta,
        sigma
    )
    new_cls_scores_novel = update_logit(
        bboxes_novel_flatten,
        logits_novel_flatten,
        labels_novel_flatten,
        torch.concat([bboxes_base, bboxes_novel], dim=0),
        torch.concat([logits_base, logits_novel], dim=0),
        torch.concat([scores_base, scores_novel], dim=0),
        background_logits_novel,
        beta,
        sigma
    )
    return torch.concat([new_cls_scores_base,new_cls_scores_novel], dim=0)[reverse_idx,:], dt_boxes_after_nms

def update_logit(
    bboxes: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    p_bboxes: torch.Tensor,
    p_logits: torch.Tensor,
    p_score: torch.Tensor,
    background_logits: torch.Tensor,
    beta: float,
    sigma: float
):
    mask = p_score > 0.5
    filt_idx = mask.nonzero(as_tuple=False).squeeze(1)
    p_dt_boxes, p_cls_logits = p_bboxes[filt_idx], p_logits[filt_idx]
    # 计算所有框之间的IOU
    # print(bboxes.shape,"\n" ,p_dt_boxes.shape)
    dt_dt_iou = box_iou(bboxes, p_dt_boxes) # shape: (n1*C, num_filt)
    # 过滤iou值以找到合适的框
    valid_iou = (dt_dt_iou > 0.1) & (dt_dt_iou < 0.7)
    higher_score = logits.unsqueeze(1) < p_cls_logits.max(dim=-1)[0].unsqueeze(0) # shape: (n1*C, num_filt)
    distrub_class =  (labels.unsqueeze(1) != p_cls_logits.argmax(dim=-1).unsqueeze(0)) | ((labels.unsqueeze(1) == p_cls_logits.argmax(dim=-1).unsqueeze(0)) & higher_score) 
    # distrub_class = ((labels.unsqueeze(1) == p_cls_logits.argmax(dim=-1).unsqueeze(0)) & higher_score) 
    ppl_cond = get_condition(logits.reshape(-1,65),background_logits,sigma)
    ppl_cond = ppl_cond[:,None].expand_as(valid_iou)
    valid_pairs = valid_iou & distrub_class & ppl_cond
    # 使用广播选择满足条件的iou和分数，并计算得分抑制
    suppressed_scores = torch.where(valid_pairs, dt_dt_iou * p_cls_logits.max(dim=-1)[0].unsqueeze(0), torch.tensor(0.).to(logits.device))
    suppressed_alpha = torch.where(valid_pairs, dt_dt_iou, torch.tensor(0.).to(logits.device))
    # 计算新分数
    #---------------------------------------sum_alpha--------------------------------------
    p = suppressed_scores.sum(dim=1)
    sum_alpha = suppressed_alpha.sum(dim=1)
    merge_logic = logits + beta * (sum_alpha * logits - p)
    #--------------------------------------------------------------------------------------
    cls_scores = F.softmax(torch.concat((merge_logic.reshape(-1,65), background_logits), dim=-1), dim=-1)
    return cls_scores

def get_condition(
    logits: torch.Tensor,
    background_logits: torch.Tensor,
    sigma = 0.2
):
    """
    input:
        logits: (N,80)
        background_logits: (N,1)
    return:
        cond: (N*80,)
    """
    class_cnt = logits.size(1)
    scores_per_class = F.softmax(torch.concat([logits,background_logits],dim=1), dim=-1)# (N, 80)
    scores_per_class, scores_per_background = scores_per_class[:,:-1], scores_per_class[:,-1:]
    scores_max_class = scores_per_class.max(dim=-1)[0]# (N,)
    mask = scores_per_class < (scores_max_class * sigma)[:,None]# (N, 80)
    cond = mask.sum(dim=1)[:,None] < (class_cnt-1)# (N, 1)
    cond = cond.repeat(1,class_cnt).reshape(-1)
    # print(scores_per_background)
    
    return cond