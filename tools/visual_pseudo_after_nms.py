import json
import torch
from logits_decode import faster_update_logits
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer
import os
import cv2
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)
import matplotlib.pyplot as plt
from tqdm import tqdm
from detectron2.config import get_cfg
import torchvision
import numpy as np

from pycocotools.coco import COCO
from utils import COCO_NOVEL_CatName as novelCatNames
from utils import get_coco_ids_by_order, detections2json
import meta_data
# from ..MarvelOVD.data.datasets import coco

output_dir = "./visual"
os.makedirs(output_dir, exist_ok=True)
idx_to_CatID =torch.tensor([28,21,47,6,76,41,18,63,32,36,81,22,61,87,5,17,49])

def visual_per_image(img_file, instance_1, instance_2):
    """plot two instances on the same image"""
    plt.close()
    plt.rcParams['savefig.dpi'] = 150 #图片像素
    plt.rcParams['figure.dpi'] = 150 #分辨率
    fig, axis = plt.subplots(1,2,figsize=(10, 5))
    img = cv2.imread(img_file)
    
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get('coco_openvoc_train_with_clip'), scale = 1.2)#MetadataCatalog.get('coco_openvoc_train_with_clip')
    # print(instance_1.pred_classes)
    out_img = v.draw_instance_predictions(instance_1).get_image()
    axis[0].imshow(out_img[:,:,::-1])
    
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get('coco_openvoc_train_with_clip'), scale = 1.2)
    out_img = v.draw_instance_predictions(instance_2).get_image()
    axis[1].imshow(out_img[:,:,::-1])
    save_path = os.path.join(output_dir, os.path.basename(img_file))
    plt.savefig(save_path[:-4]+'.png')

def make_instance(img_size, pred_boxes, pred_classes, scores):
    
    instance = Instances(img_size)
    instance.pred_boxes = pred_boxes
    instance.pred_classes = pred_classes
    # instance.pred_classes = [novelCatNames[x] for x in result_after_nms[2]]
    instance.scores = scores
    return instance

def postprocessing(
    cls_logits_List,
    box_List, 
    clip_score_List,
    pred_id_list,
    use_thre=True,
    thres=0.5,
    use_pre_cls_nms=True,
    pre_cls_nms_thres=0.6,
    device='cpu'
):
    # thresholding
    currLogitsArr = np.array(cls_logits_List)
    curBoxArr = np.array(box_List)
    curCLIPScoreArr = np.array(clip_score_List)
    curPredIdArr = np.array(pred_id_list)

    if use_thre:
        thresMask = curCLIPScoreArr > thres
        currLogitsArr = currLogitsArr[thresMask, :]
        curBoxArr = curBoxArr[thresMask, :]
        curPredIdArr = curPredIdArr[thresMask]
        curCLIPScoreArr = curCLIPScoreArr[thresMask]

    # pre-class NMS
    if use_pre_cls_nms:
        currUsedBoxTR = torch.as_tensor(curBoxArr).to(device)
        currCatIDsTR = torch.as_tensor(curPredIdArr).to(device)
        curCLIPScoreArr = torch.as_tensor(curCLIPScoreArr).to(device)
        selBoxIds = torchvision.ops.batched_nms(currUsedBoxTR, curCLIPScoreArr, currCatIDsTR, pre_cls_nms_thres)

        currLogitsArr = currLogitsArr[selBoxIds, :]
        curBoxList = currUsedBoxTR[selBoxIds, :].cpu().numpy().tolist()
        curPredIdList = currCatIDsTR[selBoxIds].cpu().numpy().tolist()
        curCLIPScoreList = curCLIPScoreArr[selBoxIds].cpu().numpy().tolist()
    else:
        curBoxList = curBoxArr.tolist()
        curPredIdList = curPredIdArr.tolist()

    return torch.tensor(currLogitsArr) ,torch.tensor(curBoxList), torch.tensor(curCLIPScoreList), torch.tensor(curPredIdList)

def main():
    metadata = MetadataCatalog.get('coco_openvoc_val_all')
    json_file = "./CLIP_scores_for_PLs/CLIP_scores_99.json"
    CLIPScoreData = json.load(open(json_file, 'r'))

    imgId_list = CLIPScoreData['img_ids_list']
    bbox_all_list = CLIPScoreData['bbox_all_list']
    rpn_score_all_list = CLIPScoreData['rpn_score_all_list']
    clip_score_all_list = CLIPScoreData['clip_score_all_list']
    clip_catIDs_all_list = CLIPScoreData['clip_catIDs_all_list']
    clip_logits_all_list = CLIPScoreData['clip_logits_all_list']
    
    orig_COCOJson_file = '../../datasets/coco/annotations/instances_train2017.json'
    coco = COCO(orig_COCOJson_file)
    usedCatIds_inOrder = np.array(get_coco_ids_by_order(coco, novelCatNames))
    
    for img_id, bboxes_per_img, rpn_scores_per_img, clip_scores_per_img, clip_catIDs_per_img, clip_logits_per_img in tqdm(zip(imgId_list, bbox_all_list, rpn_score_all_list, clip_score_all_list, clip_catIDs_all_list, clip_logits_all_list)):
        img_file = os.path.join("/data1/liangzhijia/datasets/coco/train2017/",str(img_id).zfill(12)+".jpg")
        img = cv2.imread(img_file)
        img_size = torch.tensor(list(img.shape)[:2])
        
        cls_logits = torch.tensor(clip_logits_per_img)
        dt_boxes = torch.tensor(bboxes_per_img)
        clip_catIDs = torch.tensor(clip_catIDs_per_img)[:,0]
        clip_labels = torch.tensor([metadata.get('thing_dataset_id_to_contiguous_id')[int(k)] for k in clip_catIDs])
        clip_scores = torch.tensor(clip_scores_per_img).permute(1,0)[0]
        print(cls_logits)
        # use nms:
        cls_logits, dt_boxes, clip_scores, clip_labels = postprocessing(
            cls_logits,
            dt_boxes,
            clip_scores,
            clip_labels
        )
        if len(cls_logits.shape)==1: continue
        new_cls_logits = faster_update_logits(cls_logits, dt_boxes, beta=0.5, sigma=0.0)

        new_clip_scores = torch.softmax(new_cls_logits, dim=-1)
        new_clip_scores, new_cls_id = new_clip_scores[:,:17].topk(1)
        new_cls_id = torch.tensor(usedCatIds_inOrder[new_cls_id.cpu().numpy()])

        new_clip_scores, new_cls_id = new_clip_scores[:,0], new_cls_id[:,0]
        
        new_contiguous_id = torch.tensor([metadata.get('thing_dataset_id_to_contiguous_id')[int(k)] for k in new_cls_id])
        
        instance_2, instance_1 = make_instance(img_size, dt_boxes, new_contiguous_id, new_clip_scores), make_instance(img_size, dt_boxes, clip_labels, clip_scores)
        visual_per_image(img_file, instance_1, instance_2)

if __name__ == "__main__":
    # metadata = MetadataCatalog.get('coco_openvoc_val_all')
    # print(metadata)    
    main()