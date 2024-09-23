import torch
import torch.nn.functional as F
import numpy as np
from torchvision.ops import box_iou


def faster_update_logits(
    cls_logits: torch.Tensor, 
    dt_boxes: torch.Tensor, 
    beta: float = 0.5,
    sigma: float = 0.0
) -> torch.Tensor:
    """
    参数:
        cls_logits (torch.Tensor): 形状为 (N, C) 的张量，包含每个候选框的类别逻辑值。
        dt_boxes (torch.Tensor): 形状为 (N, 4) 的张量，表示候选框的坐标。
        beta (float): 控制逻辑调整强度的参数，默认为 0.5。
        sigma (float): 控制背景逻辑影响的参数，默认为 0.0。
    返回:
        torch.Tensor: 更新后的logits, (N, C)。

    示例:
        # 假设有一组类别逻辑值和边界框数据
        cls_logits = torch.randn(1000, 80)  # 包括80个类别
        dt_boxes = torch.randn(1000, 4)
        updated_scores = faster_update_logic_after_nms_3(cls_logits, dt_boxes)
    """
    dt_boxes_after_nms, cls_logits_after_nms = dt_boxes, cls_logits
    cls_score_after_nms = F.softmax(cls_logits_after_nms,dim=-1)[:,:-1].max(dim=-1)[0]
    # 仿照multiclass_nms的方法处理, 将框的每个类别都看作是一个框, 调整每个类别logit的值, 但是选取相邻框时只考虑logit最大的那个
    num_classes = cls_logits.size(1) 
    logits = cls_logits_after_nms 
    cls_logits_after_nms_label = torch.argmax(logits, dim=1)
    labels = torch.arange(num_classes, dtype=torch.long, device=logits.device)
    labels = labels.view(1, -1).expand_as(logits)
    
    novel = torch.tensor([4,5,11,12,15,16,21,23,27,29,32,34,45,47,54,58,63]).to(cls_logits_after_nms_label.device)
    is_novel = torch.isin(cls_logits_after_nms_label, novel)
    is_novel = torch.ones_like(cls_logits_after_nms_label)
    novel_idx, base_idx = is_novel.nonzero(as_tuple=False).squeeze(1), (is_novel == 0).nonzero(as_tuple=False).squeeze(1)
    
    novel_base_idx_to_init_idx = {b:a for a,b in enumerate(base_idx.tolist() + novel_idx.tolist())}
    reverse_idx = torch.tensor([novel_base_idx_to_init_idx[i] for i in range(len(novel_base_idx_to_init_idx))]).to(logits.device)
    
    scores_base, scores_novel = cls_score_after_nms[base_idx], cls_score_after_nms[novel_idx] 
    logits_base, logits_novel = logits[base_idx,:], logits[novel_idx,:]
    if len(logits_base.shape) == 1:
        logits_base = logits_base.unsqueeze(0)
    if len(logits_novel.shape) == 1:
        logits_novel = logits_novel.unsqueeze(0)
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
    new_cls_logits_base = update_logit(
        bboxes_base_flatten,
        logits_base_flatten,
        labels_base_flatten,
        bboxes_base,
        logits_base,
        scores_base,
        beta,
        sigma
    )
    new_cls_logits_novel = update_logit(
        bboxes_novel_flatten,
        logits_novel_flatten,
        labels_novel_flatten,
        torch.concat([bboxes_base, bboxes_novel], dim=0),
        torch.concat([logits_base, logits_novel], dim=0),
        torch.concat([scores_base, scores_novel], dim=0),
        beta,
        sigma
    )
    return torch.concat([new_cls_logits_base,new_cls_logits_novel], dim=0)[reverse_idx,:]

def update_logit(
    bboxes: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    p_bboxes: torch.Tensor,
    p_logits: torch.Tensor,
    p_score: torch.Tensor,
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
    # higher_score = logits.unsqueeze(1) < p_cls_logits.max(dim=-1)[0].unsqueeze(0) # shape: (n1*C, num_filt)
    scores_per_cls = logits.reshape(-1,17).softmax(dim=-1).reshape(-1)
    p_cls_scores = p_cls_logits.softmax(dim=-1)
    higher_score = scores_per_cls.unsqueeze(1) < p_cls_scores.max(dim=-1)[0].unsqueeze(0)
    distrub_class =  (labels.unsqueeze(1) != p_cls_logits.argmax(dim=-1).unsqueeze(0)) | ((labels.unsqueeze(1) == p_cls_logits.argmax(dim=-1).unsqueeze(0)) & higher_score) 
    # distrub_class = ((labels.unsqueeze(1) == p_cls_logits.argmax(dim=-1).unsqueeze(0)) & higher_score) 
    ppl_cond = get_condition(logits.reshape(-1,17),sigma)
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
    return merge_logic.reshape(-1,17)

def get_condition(
    logits: torch.Tensor,
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
    scores_per_class = F.softmax(logits, dim=-1)# (N, 80)
    scores_max_class = scores_per_class.max(dim=-1)[0]# (N,)
    mask = scores_per_class < (scores_max_class * sigma)[:,None]# (N, 80)
    cond = mask.sum(dim=1)[:,None] < (class_cnt-1)# (N, 1)
    cond = cond.repeat(1,class_cnt).reshape(-1)
    # print(scores_per_background)
    
    return cond