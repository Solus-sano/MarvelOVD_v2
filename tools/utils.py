from tqdm import tqdm, trange

import numpy as np
from PIL import Image, ImageDraw

import torch, torchvision
import torch.nn.functional as F

import clip

from detectron2.structures import ImageList, Boxes
from torchvision.ops import box_iou, box_area
from typing import Tuple
from torch import Tensor
import os
import matplotlib.pyplot as plt
import copy

import torchvision.transforms as T
from pycocotools.coco import COCO
from meta_data import BASE_CATEGORIES_ID

### class names
COCO_BASE_CatName = [
    "toilet",
    "bicycle",
    "apple",
    "train",
    "laptop",
    "carrot",
    "motorcycle",
    "oven",
    "chair",
    "mouse",
    "boat",
    "kite",
    "sheep",
    "horse",
    "sandwich",
    "clock",
    "tv",
    "backpack",
    "toaster",
    "bowl",
    "microwave",
    "bench",
    "book",
    "orange",
    "bird",
    "pizza",
    "fork",
    "frisbee",
    "bear",
    "vase",
    "toothbrush",
    "spoon",
    "giraffe",
    "handbag",
    "broccoli",
    "refrigerator",
    "remote",
    "surfboard",
    "car",
    "bed",
    "banana",
    "donut",
    "skis",
    "person",
    "truck",
    "bottle",
    "suitcase",
    "zebra"
]
COCO_NOVEL_CatName = [
    "umbrella",
    "cow",
    "cup",
    "bus",
    "keyboard",
    "skateboard",
    "dog",
    "couch",
    "tie",
    "snowboard",
    "sink",
    "elephant",
    "cake",
    "scissors",
    "airplane",
    "cat",
    "knife"
]


#### for CLIP text embeddings
def article(name):
    return 'an' if name[0] in 'aeiou' else 'a'

def processed_name(name, rm_dot=False):
  # _ for lvis
  # / for obj365
  res = name.replace('_', ' ').replace('/', ' or ').lower()
  if rm_dot:
    res = res.rstrip('.')
  return res

single_template = [
    'a photo of {article} {}.'
]
multiple_templates = [
    'There is {article} {} in the scene.',
    'There is the {} in the scene.',
    'a photo of {article} {} in the scene.',
    'a photo of the {} in the scene.',
    'a photo of one {} in the scene.',

    'itap of {article} {}.',
    'itap of my {}.',  # itap: I took a picture of
    'itap of the {}.',
    'a photo of {article} {}.',
    'a photo of my {}.',
    'a photo of the {}.',
    'a photo of one {}.',
    'a photo of many {}.',

    'a good photo of {article} {}.',
    'a good photo of the {}.',
    'a bad photo of {article} {}.',
    'a bad photo of the {}.',
    'a photo of a nice {}.',
    'a photo of the nice {}.',
    'a photo of a cool {}.',
    'a photo of the cool {}.',
    'a photo of a weird {}.',
    'a photo of the weird {}.',

    'a photo of a small {}.',
    'a photo of the small {}.',
    'a photo of a large {}.',
    'a photo of the large {}.',

    'a photo of a clean {}.',
    'a photo of the clean {}.',
    'a photo of a dirty {}.',
    'a photo of the dirty {}.',

    'a bright photo of {article} {}.',
    'a bright photo of the {}.',
    'a dark photo of {article} {}.',
    'a dark photo of the {}.',

    'a photo of a hard to see {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of {article} {}.',
    'a low resolution photo of the {}.',
    'a cropped photo of {article} {}.',
    'a cropped photo of the {}.',
    'a close-up photo of {article} {}.',
    'a close-up photo of the {}.',
    'a jpeg corrupted photo of {article} {}.',
    'a jpeg corrupted photo of the {}.',
    'a blurry photo of {article} {}.',
    'a blurry photo of the {}.',
    'a pixelated photo of {article} {}.',
    'a pixelated photo of the {}.',

    'a black and white photo of the {}.',
    'a black and white photo of {article} {}.',

    'a plastic {}.',
    'the plastic {}.',

    'a toy {}.',
    'the toy {}.',
    'a plushie {}.',
    'the plushie {}.',
    'a cartoon {}.',
    'the cartoon {}.',

    'an embroidered {}.',
    'the embroidered {}.',

    'a painting of the {}.',
    'a painting of a {}.',
]

def build_text_embedding(model, categories, templates, add_this_is=False, show_process=True):
    run_on_gpu = torch.cuda.is_available()

    with torch.no_grad():
        all_text_embeddings = []
        if show_process:
            print('Building text embeddings...')
        for catName in (tqdm(categories) if show_process else categories):
            texts = [template.format(catName, article=article(catName)) for template in templates]
            if add_this_is:
                texts = ['This is ' + text if text.startswith('a') or text.startswith('the') else text
                         for text in texts]
            texts = clip.tokenize(texts)  # tokenize
            if run_on_gpu:
                texts = texts.cuda()

            text_embeddings = model.encode_text(texts)  # embed with text encoder
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            all_text_embeddings.append(text_embedding)

        all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
        if run_on_gpu:
            all_text_embeddings = all_text_embeddings.cuda()

        return all_text_embeddings.transpose(dim0=0, dim1=1)


# get multi-scale CLIP image imbedding
def CLIP_score_multi_scale(clip_model, img_patch_scalelist, text_features, softmax_t=0.01, return_logits=False):
    # img_patch_scalelist: [ [n*patches for scale 1], [n*patches for scale 2], ...]
    # patchNum = img_patch_scalelist[0].shape[0]

    patchNum = len(img_patch_scalelist[0])
    patch_per_split = 1000

    splitIdxList = [i*patch_per_split for i in range(1 + patchNum//patch_per_split)]
    if splitIdxList[-1] != patchNum:
        splitIdxList.append(patchNum)

    allSimilarity = []
    allLogits = []

    for sp_idx in range(len(splitIdxList) - 1):
        startIdx = splitIdxList[sp_idx]
        endIdx = splitIdxList[sp_idx + 1]

        image_features = None
        for s_id, imgPatchesList in enumerate(img_patch_scalelist):
            imgPatches = torch.cat(imgPatchesList[startIdx:endIdx], dim=0)

            with torch.no_grad():
                curImgFeat = clip_model.encode_image(imgPatches)
                curImgFeat /= curImgFeat.norm(dim=-1, keepdim=True)

            if image_features is None:
                image_features = curImgFeat.clone()
            else:
                image_features += curImgFeat

        sacleNum = len(img_patch_scalelist)
        image_features /= sacleNum
        image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = ((1 / softmax_t) * image_features @ text_features.T).softmax(dim=-1)
        allSimilarity.append(similarity)

        logits = (1 / softmax_t) * image_features @ text_features.T # shape: (n, 17)
        allLogits.append(logits)
        
    allSimilarity = torch.cat(allSimilarity, dim=0)
    allLogits = torch.cat(allLogits, dim=0)
    if return_logits:
        return allSimilarity, allLogits
    return allSimilarity

def attn_mask_CLIP_score_multi_scale(clip_model, img_patch_scalelist, clip_attn_mask_scalelist, text_features, softmax_t=0.01, return_logits=False):
    # img_patch_scalelist: [ [n*patches for scale 1], [n*patches for scale 2], ...]
    # patchNum = img_patch_scalelist[0].shape[0]

    patchNum = len(img_patch_scalelist[0])
    patch_per_split = 1000

    splitIdxList = [i*patch_per_split for i in range(1 + patchNum//patch_per_split)]
    if splitIdxList[-1] != patchNum:
        splitIdxList.append(patchNum)

    allSimilarity = []
    allLogits = []

    for sp_idx in range(len(splitIdxList) - 1):
        startIdx = splitIdxList[sp_idx]
        endIdx = splitIdxList[sp_idx + 1]

        image_features = None
        mask_image_features = None
        for s_id, (imgPatchesList, attn_mask_list) in enumerate(zip(img_patch_scalelist, clip_attn_mask_scalelist)):
            imgPatches = torch.cat(imgPatchesList[startIdx:endIdx], dim=0)
            attn_mask = torch.cat(attn_mask_list[startIdx:endIdx] ,dim=0)

            with torch.no_grad():
                curImgFeat = clip_model.encode_image(imgPatches, attn_mask)
                curImgFeat /= curImgFeat.norm(dim=-1, keepdim=True)

            if image_features is None:
                image_features = curImgFeat.clone()
            else:
                image_features += curImgFeat

        sacleNum = len(img_patch_scalelist)
        image_features /= sacleNum
        image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = ((1 / softmax_t) * image_features @ text_features.T).softmax(dim=-1)
        allSimilarity.append(similarity)

        logits = (1 / softmax_t) * image_features @ text_features.T # shape: (1, 17)
        allLogits.append(logits)
        
    allSimilarity = torch.cat(allSimilarity, dim=0)
    allLogits = torch.cat(allLogits, dim=0)
    if return_logits:
        return allSimilarity, allLogits
    return allSimilarity

def merge_attn_mask_CLIP_score_multi_scale(clip_model, img_patch_scalelist, clip_attn_mask_scalelist, text_features, softmax_t=0.01, return_logits=False, merge_lambda=0.5):
    # img_patch_scalelist: [ [n*patches for scale 1], [n*patches for scale 2], ...]
    # patchNum = img_patch_scalelist[0].shape[0]

    patchNum = len(img_patch_scalelist[0])
    patch_per_split = 1000

    splitIdxList = [i*patch_per_split for i in range(1 + patchNum//patch_per_split)]
    if splitIdxList[-1] != patchNum:
        splitIdxList.append(patchNum)

    allSimilarity = []
    allLogits = []

    for sp_idx in range(len(splitIdxList) - 1):
        startIdx = splitIdxList[sp_idx]
        endIdx = splitIdxList[sp_idx + 1]

        image_features = None
        mask_image_features = None
        for s_id, (imgPatchesList, attn_mask_list) in enumerate(zip(img_patch_scalelist, clip_attn_mask_scalelist)):
            imgPatches = torch.cat(imgPatchesList[startIdx:endIdx], dim=0)
            attn_mask = torch.cat(attn_mask_list[startIdx:endIdx] ,dim=0)

            with torch.no_grad():
                curImgFeat = clip_model.encode_image(imgPatches)
                curImgFeat /= curImgFeat.norm(dim=-1, keepdim=True)
                
                mask_image_features = clip_model.encode_image(imgPatches, attn_mask)
                mask_image_features /= mask_image_features.norm(dim=-1, keepdim=True)

            if image_features is None:
                image_features = curImgFeat.clone()
            else:
                image_features += curImgFeat
                
            if mask_image_features is None:
                mask_image_features = mask_image_features.clone()
            else:
                mask_image_features += mask_image_features

        sacleNum = len(img_patch_scalelist)
        image_features /= sacleNum
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        mask_image_features /= sacleNum
        mask_image_features /= mask_image_features.norm(dim=-1, keepdim=True)

        origin_logit = (1 / softmax_t) * image_features @ text_features.T
        mask_logit = (1 / softmax_t) * mask_image_features @ text_features.T
        merge_logit = origin_logit * (1 - merge_lambda) + mask_logit * merge_lambda
        
        similarity = (merge_logit).softmax(dim=-1)
        allSimilarity.append(similarity)

        # logits = (1 / softmax_t) * image_features @ text_features.T # shape: (1, 17)
        allLogits.append(merge_logit)
        
    allSimilarity = torch.cat(allSimilarity, dim=0)
    allLogits = torch.cat(allLogits, dim=0)
    if return_logits:
        return allSimilarity, allLogits
    return allSimilarity

# partially run roi_head of the detectron2 model
def refineBoxByRoIHead(roi_head, features, proposals):
    with torch.no_grad():
        features = [features[f] for f in roi_head.box_in_features]
        box_features = roi_head.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = roi_head.box_head(box_features)
        predictions = roi_head.box_predictor(box_features)
        del box_features

        boxes = roi_head.box_predictor.predict_boxes(predictions, proposals)

    # choose only one proposal delats
    meanBoxesList = list()

    num_preds_per_image = [len(p) for p in proposals]
    for iidx, curBoxes in enumerate(boxes):
        predNum = num_preds_per_image[iidx]
        curBoxes = curBoxes.reshape(predNum, -1, 4)  # N x (k x 4)
        curBoxes = curBoxes.mean(dim=1)  # N x 4
        meanBoxesList.append(curBoxes)

    return meanBoxesList

def get_region_proposal(input_img, CA_maskRCNN, DataAug=None, roihead_num=10, topK_box=100):
    # load image tensor & pre-process
    height, width = input_img.shape[:2] # BGR

    if DataAug is not None:
        imgTR = DataAug.get_transform(input_img).apply_image(input_img)
    else:
        imgTR = input_img
    imgTR = torch.as_tensor((imgTR).astype("float32").transpose(2, 0, 1)).to(CA_maskRCNN.device)
    imgTR = ((imgTR - CA_maskRCNN.pixel_mean) / CA_maskRCNN.pixel_std)
    resizeRatio = height / imgTR.shape[1]

    images = ImageList.from_tensors([imgTR], CA_maskRCNN.backbone.size_divisibility)

    # get region proposals by repeating the roi head
    with torch.no_grad():
        features = CA_maskRCNN.backbone(images.tensor)
        proposals, _ = CA_maskRCNN.proposal_generator(images, features, None)

        curProposals = proposals
        for roiCount in range(roihead_num):
            boxes = refineBoxByRoIHead(CA_maskRCNN.roi_heads, features, curProposals)
            curProposals[0].proposal_boxes = Boxes(boxes[0])

        proposal_boxes = curProposals[0].proposal_boxes.tensor.cpu().numpy() * resizeRatio  # need to rescale
        proposal_boxes = proposal_boxes.tolist()
        proposal_boxes = proposal_boxes[:topK_box]

        pp_scores = torch.sigmoid(proposals[0].objectness_logits).cpu().numpy()
        pp_scores = pp_scores.tolist()
        pp_scores = pp_scores[:topK_box]

    return proposal_boxes, pp_scores

def get_CLIP_pred_for_proposals(input_img, proposal_boxes, pp_scores,
                                CLIP_model, preprocess, clip_text_embed, usedCatIds_inOrder,
                                box_scalelist=[1, 1.5], topK_clip_scores=1, device='cuda', return_logits=False, image_id = 0, coco_api = None):
    '''
    input_img: from cv2.imread, in BGR
    proposal_boxes: [[xyxy], [xyxy], ...]
    pp_scores: objectness scores for each region proposal
    '''
    proposal_boxes, pp_scores, _ = filt_gt_base_box(image_id=image_id, proposal_boxes=proposal_boxes, pp_scores=pp_scores, coco=coco_api, iou_thresh=0.5)
    proposal_boxes, pp_scores = proposal_boxes.tolist(), pp_scores.tolist()
    height, width = input_img.shape[:2]  # BGR
    pilImg = Image.fromarray(input_img[:, :, ::-1])  # RGB

    usedCatNum = len(usedCatIds_inOrder)

    curBoxList = list()
    curRPNScoreList = list()
    curCLIPScoreList = list()
    curPredCatIdList = list()
    curCLIPLogitList = list()

    clipInput_list_scalelist = [[] for i in range(len(box_scalelist))]
    for b_idx, box in enumerate(proposal_boxes):
        box = scale_box(box, 1, max_H=height, max_W=width)  # ensure every box is in the image
        if box[2] - box[0] >= 5 and box[3] - box[1] >= 5:
            curBoxList.append(box)
            curRPNScoreList.append(pp_scores[b_idx])

            # add scales
            for scale_id, boxScale in enumerate(box_scalelist):
                scaledBox = scale_box(box, boxScale, max_H=height, max_W=width)
                cropImg = pilImg.crop(scaledBox)

                clipInput = preprocess(cropImg).unsqueeze(0).to(device)
                clipInput_list_scalelist[scale_id].append(clipInput)

    if len(curBoxList) > 0:
        allSimilarity, allLogits = CLIP_score_multi_scale(CLIP_model, clipInput_list_scalelist, clip_text_embed, return_logits=return_logits)
        curCLIPLogitList.append(allLogits.tolist())

        ############### merge CLIP and RPN scores
        for b_idx, box in enumerate(curBoxList):
            clipScores, indices = allSimilarity[b_idx][:usedCatNum].topk(topK_clip_scores)

            curCLIPScoreList.append(clipScores.cpu().numpy().tolist())
            curPredCatIdList.append(usedCatIds_inOrder[indices.cpu().numpy()].tolist())

    if return_logits:
        return curBoxList, curRPNScoreList, curCLIPScoreList, curPredCatIdList, curCLIPLogitList
    return curBoxList, curRPNScoreList, curCLIPScoreList, curPredCatIdList

def get_better_CLIP_pred_for_proposals(image_id, input_img, proposal_boxes, pp_scores,
                                CLIP_model, preprocess, clip_text_embed, usedCatIds_inOrder,
                                box_scalelist=[1, 1.5], topK_clip_scores=1, device='cuda', return_logits=False,
                                visual_root = None, coco_api = None, gt_base_boxes = None
                            ):
    '''
    input_img: from cv2.imread, in BGR
    proposal_boxes: [[xyxy], [xyxy], ...]
    pp_scores: objectness scores for each region proposal
    '''
    if len(proposal_boxes) == 0:
        if return_logits:
            return [], [], [], [], []
        return [], [], [], []
    height, width = input_img.shape[:2]  # BGR
    pilImg = Image.fromarray(input_img[:, :, ::-1])  # RGB
    proposal_boxes, pp_scores = sort_pp_box(proposal_boxes, pp_scores)
    # get proposal boxes that score > 0.8
    better_proposal_boxes = []
    for pp_b, pp_s in zip(proposal_boxes, pp_scores):
        if pp_s < 0.95:
            break
        better_proposal_boxes.append(copy.deepcopy(pp_b))

    if gt_base_boxes is not None:
        better_proposal_boxes = gt_base_boxes.tolist() + better_proposal_boxes

    usedCatNum = len(usedCatIds_inOrder)

    curBoxList = list()
    curRPNScoreList = list()
    curCLIPScoreList = list()
    curPredCatIdList = list()
    curCLIPLogitList = list()

    trans = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
    ])

    proposal_boxes, pp_scores, _ = filt_gt_base_box(image_id=image_id, proposal_boxes=proposal_boxes, pp_scores=pp_scores, coco=coco_api, iou_thresh=0.5)
    proposal_boxes, pp_scores = proposal_boxes.tolist(), pp_scores.tolist()
    
    clipInput_list_scalelist = [[] for i in range(len(box_scalelist))]
    clip_attn_mask_list_scalelist = [[] for i in range(len(box_scalelist))]
    for b_idx, box in enumerate(proposal_boxes):
        box = scale_box(box, 1, max_H=height, max_W=width)  # ensure every box is in the image
        if box[2] - box[0] >= 5 and box[3] - box[1] >= 5:
            curBoxList.append(box)
            curRPNScoreList.append(pp_scores[b_idx])

            mask_PilImg, attention_mask, mask_flag = get_mask_img(input_img,box,better_proposal_boxes,mask_thr=0.5)
            # add scales
            for scale_id, boxScale in enumerate(box_scalelist):
                scaledBox = scale_box(box, boxScale, max_H=height, max_W=width)
                cropImg = pilImg.crop(scaledBox)
                mask_cropImg = None
                # flag = 0
                # if len(better_proposal_boxes) > 0:
                    # mask_cropImg, flag = mask_overlaps(input_img, scaledBox, better_proposal_boxes, mask_thr=0.5, visual_root = visual_root)
                    # mask_cropImg = Image.fromarray(mask_cropImg)
                # mask_cropImg = mask_PilImg.crop(scaledBox)
                attention_crop_mask = attention_mask.crop(scaledBox)

                clipInput = preprocess(cropImg).unsqueeze(0).to(device)
                # mask_clipInput = preprocess(mask_cropImg).unsqueeze(0).to(device)
                attention_crop_mask = trans(attention_crop_mask).unsqueeze(0)[:,-1:,:,:]
                attention_crop_mask = attention_crop_mask / torch.max(attention_crop_mask)
                # print(attention_crop_mask)
                # cnt = torch.sum(attention_crop_mask)
                # plt.close()
                # fig, axis = plt.subplots(1,2,figsize=(10, 5))
                # # title
                # title_1 = f'mask_img'
                # title_2 = f'mask_{cnt}'
                # axis[0].imshow(mask_clipInput[0].cpu().numpy().transpose(1, 2, 0))
                # axis[0].set_title(title_1)
                # axis[1].imshow(attention_crop_mask[0,0,:,:].cpu().numpy())
                # axis[1].set_title(title_2)
                # plt.savefig(os.path.join(visual_root, f'mask_{b_idx}_{scale_id}.png'))
                # plt.close()
                
                # if mask_flag == 1:
                #     with torch.no_grad():
                #         curImgFeat = CLIP_model.encode_image(clipInput)
                #         curImgFeat /= curImgFeat.norm(dim=-1, keepdim=True)
                        
                #         mask_curImgFeat = CLIP_model.encode_image(clipInput, attention_crop_mask)
                #         mask_curImgFeat /= mask_curImgFeat.norm(dim=-1, keepdim=True)
                    
                #         similarity = ((1 / 0.01) * curImgFeat @ clip_text_embed.T).softmax(dim=-1)
                #         mask_similarity = ((1 / 0.01) * mask_curImgFeat @ clip_text_embed.T).softmax(dim=-1)
                        
                #         score, label = torch.max(similarity, dim=-1)
                #         mask_score, mask_label = torch.max(mask_similarity, dim=-1)
                        
                #         catID = int(usedCatIds_inOrder[label])
                #         mask_catID = int(usedCatIds_inOrder[mask_label])

                #         cat_name = coco_api.loadCats(catID)[0]['name']
                #         mask_cat_name = coco_api.loadCats(mask_catID)[0]['name']
                        
                #         # mask 后变化不大的不进行可视化
                #         if score < 0.8 and mask_score < 0.8:
                #             continue
                        
                #         plt.close()
                #         fig, axis = plt.subplots(1,2,figsize=(10, 5))
                #         # title
                #         title_1 = f'{cat_name}_{float(score):.2f}'
                #         title_2 = f'{mask_cat_name}_{float(mask_score):.2f}'
                #         axis[0].imshow(cropImg)
                #         axis[0].set_title(title_1)
                #         axis[1].imshow(mask_cropImg)
                #         axis[1].set_title(title_2)
                #         plt.savefig(os.path.join(visual_root, f'proposal_{b_idx}_{scale_id}.png'))
                #         plt.close()
                
                # continue
                clipInput_list_scalelist[scale_id].append(clipInput)
                clip_attn_mask_list_scalelist[scale_id].append(attention_crop_mask)
                # mask_clipInput_list_scalelist[scale_id].append(mask_clipInput)
    # return
    if len(curBoxList) > 0:
        # allSimilarity, allLogits = CLIP_score_multi_scale(CLIP_model, clipInput_list_scalelist, clip_text_embed, return_logits=return_logits)
        allSimilarity, allLogits = attn_mask_CLIP_score_multi_scale(CLIP_model, clipInput_list_scalelist, clip_attn_mask_list_scalelist, clip_text_embed, return_logits=return_logits)
        # allSimilarity, allLogits = merge_attn_mask_CLIP_score_multi_scale(CLIP_model, clipInput_list_scalelist, clip_attn_mask_list_scalelist, clip_text_embed, return_logits=return_logits)

        curCLIPLogitList.append(allLogits.cpu().numpy().tolist())
        ############### merge CLIP and RPN scores
        for b_idx, box in enumerate(curBoxList):
            clipScores, indices = allSimilarity[b_idx][:usedCatNum].topk(topK_clip_scores)

            curCLIPScoreList.append(clipScores.cpu().numpy().tolist())
            curPredCatIdList.append(usedCatIds_inOrder[indices.cpu().numpy()].tolist())

    if return_logits:
        return curBoxList, curRPNScoreList, curCLIPScoreList, curPredCatIdList, curCLIPLogitList
    return curBoxList, curRPNScoreList, curCLIPScoreList, curPredCatIdList

def detection_postprocessing(box_List, score_List, pred_id_list,
                                use_thre=True, thres=0.05,
                                use_pre_cls_nms=True, pre_cls_nms_thres=0.6, device='cuda'):
    # thresholding
    curBoxArr = np.array(box_List)
    curScoreArr = np.array(score_List)
    curPredIdArr = np.array(pred_id_list)

    if use_thre:
        thresMask = curScoreArr > thres
        curBoxArr = curBoxArr[thresMask, :]
        curScoreArr = curScoreArr[thresMask]
        curPredIdArr = curPredIdArr[thresMask]

    # pre-class NMS
    if use_pre_cls_nms:
        currUsedBoxTR = torch.as_tensor(curBoxArr).to(device)
        currScoreTR = torch.as_tensor(curScoreArr).to(device)
        currCatIDsTR = torch.as_tensor(curPredIdArr).to(device)
        selBoxIds = torchvision.ops.batched_nms(currUsedBoxTR, currScoreTR, currCatIDsTR, pre_cls_nms_thres)

        curBoxList = currUsedBoxTR[selBoxIds, :].cpu().numpy().tolist()
        curScoreList = currScoreTR[selBoxIds].cpu().numpy().tolist()
        curPredIdList = currCatIDsTR[selBoxIds].cpu().numpy().tolist()
    else:
        curBoxList = curBoxArr.tolist()
        curScoreList = curScoreArr.tolist()
        curPredIdList = curPredIdArr.tolist()

    return curBoxList, curScoreList, curPredIdList

#### data processing
def get_coco_ids_by_order(coco, cat_name_list):
    catIDs_list = list()
    for catName in cat_name_list:
        catID = coco.getCatIds(catNms=[catName])
        if len(catID) != 1:
            print('%s is not valid cat name' % catName)
        else:
            catIDs_list.append(catID[0])

    return catIDs_list

def xyxy2xywh(bbox):
    """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
    evaluation.

    Args:
        bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
            ``xyxy`` order.

    Returns:
        list[float]: The converted bounding boxes, in ``xywh`` order.
    """
    if isinstance(bbox, list):
        _bbox = bbox
    else:
        _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0],
        _bbox[3] - _bbox[1],
    ]

def detections2json(img_ids, box_all_list, score_all_list, pred_catId_all_list):
    """Convert proposal results to COCO json style."""
    json_results = []

    for idx, img_id in enumerate(img_ids):
        img_id = img_ids[idx]
        bboxesList = box_all_list[idx]
        scoreList = score_all_list[idx]
        catIdList = pred_catId_all_list[idx]

        for b_idx, box in enumerate(bboxesList):
            box = xyxy2xywh(box)
            score = scoreList[b_idx]
            catId = catIdList[b_idx]

            data = {'image_id': img_id,
                    'category_id': catId,
                    'bbox': box,
                    'score': score}
            json_results.append(data)

    return json_results

def scale_box(box, scale, max_H=np.inf, max_W=np.inf):
    # box: x0, y0, x1, y1
    # scale: float
    x0, y0, x1, y1 = box

    cx = ((x0 + x1) / 2)
    cy = ((y0 + y1) / 2)
    bw = (x1 - x0) * scale
    bh = (y1 - y0) * scale

    new_x0 = max(cx - bw / 2, 0)
    new_y0 = max(cy - bh / 2, 0)
    new_x1 = min(cx + bw / 2, max_W)
    new_y1 = min(cy + bh / 2, max_H)

    return [new_x0, new_y0, new_x1, new_y1]


def sort_pp_box(proposal_boxes, pp_scores):
    # sort pp boxes according to pp scores
    proposal_boxes, pp_scores = np.array(proposal_boxes), np.array(pp_scores)
    sorted_inds = np.argsort(-pp_scores)
    sorted_boxes = proposal_boxes[sorted_inds,:]
    sorted_scores = pp_scores[sorted_inds]
    return sorted_boxes, sorted_scores

def box_inter_union(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    """计算两组矩形框的交集和并集"""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # 找到交集的左上角和右下角坐标
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[..., 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[..., 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[..., 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[..., 3])

    # 计算交集的面积
    inter_area = box_area(torch.stack([inter_x1, inter_y1, inter_x2, inter_y2], dim=-1))
    inter_area = inter_area.clamp(min=0)  # 确保面积不为负

    # 计算并集的面积
    union_area = area1[:, None] + area2 - inter_area

    return inter_area, union_area

def get_2_type_boxes_iou(
        boxes1: Tensor, 
        boxes2: Tensor
    ) -> Tuple[Tensor, Tensor]:
    
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # 找到交集的左上角和右下角坐标
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[..., 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[..., 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[..., 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[..., 3])

    # 计算交集的面积
    # print(torch.concat([inter_x1, inter_y1, inter_x2, inter_y2], dim=).shape)
    inter_area = box_area(torch.stack([inter_x1, inter_y1, inter_x2, inter_y2], dim=-1)[0])
    inter_area = inter_area.clamp(min=0)  # 确保面积不为负

    # 计算并集的面积
    # print(area1.shape)
    # print(area2.shape)
    # print(inter_area.shape)
    union_area = area1[:, None] + area2 - inter_area
    
    normal_iou = inter_area / union_area
    bias_iou = inter_area / area1[:, None]
    return normal_iou, bias_iou

def mask_overlaps(img,
                box_a, 
                boxes_b, 
                mask_thr=0.5,
                visual_root = None
            ):
    """
    input:
        img: 输入图像, numpy array of shape (H, W, 3)
        box_a: 主候选框 A, 格式为 (x0, y0, x1, y1)
        boxes_b: 一系列候选框 Bi, 格式为 [(xi0, yi0, xi1, yi1), ...]
        boxes_b_scores: 一系列候选框 Bi 的得分(已按从大到小排序)
        mask_thr: 面积遮罩阈值
    return:
        crop_img: 裁剪后的图像
    """
    scaleBox = copy.deepcopy(box_a)
    box_a, boxes_b = torch.tensor(np.array(box_a)), torch.tensor(np.array(boxes_b))
    
    plt.close()
    # fig, axis = plt.subplots(1,2,figsize=(10, 5))
    
    # 裁剪出候选框 A
    # pilImage = Image.fromarray(img[:,:,::-1])
    mask_img = copy.deepcopy(img)
    # crop_img = img[int(box_a[1]):int(box_a[3]), int(box_a[0]):int(box_a[2])].copy()
    # axis[0].imshow(crop_img[:,:,::-1])
    img_A_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    total_mask_area = 0

    normal_iou, bias_iou = get_2_type_boxes_iou(box_a[None, :], boxes_b)
    normal_iou, bias_iou = normal_iou[0], bias_iou[0]
    # 计算并遮盖重叠区域
    flag = 0
    for i, box in enumerate(boxes_b):
        # 计算交集
        x0 = int(max(box_a[0], box[0]))
        y0 = int(max(box_a[1], box[1]))
        x1 = int(min(box_a[2], box[2]))
        y1 = int(min(box_a[3], box[3]))

        iou_condition = (bias_iou[i]>0.1) & (normal_iou[i]<0.7)
        # 检查是否有交集
        if x0 < x1 and y0 < y1 and iou_condition:
            # 转换坐标到裁剪图像上
            # x0 -= int(box_a[0])
            # x1 -= int(box_a[0])
            # y0 -= int(box_a[1])
            # y1 -= int(box_a[1])
            
            # 计算遮盖面积
            mask_area = (y1 - y0) * (x1 - x0)
            total_mask_area += mask_area
            if (total_mask_area/img_A_area) > mask_thr:
                continue
            
            # 应用遮罩
            # print(x0, " ", x1, " ", y0, " ", y1, " ")
            # crop_img[y0:y1, x0:x1] = 0  # 假设使用黑色遮罩
            mask_img[y0:y1, x0:x1] = 0
            flag = 1
    mask_pilImage = Image.fromarray(mask_img[:,:,::-1])
    crop_img = mask_pilImage.crop(scaleBox)
    # if flag == 0:
    #     return crop_img, 
    # axis[1].imshow(crop_img[:,:,::-1])
    # file_name = generate_unique_filename(visual_root,"mask", ".jpg")
    # plt.savefig(file_name)
    return crop_img, flag

def get_mask_img(img,
                box_a, 
                boxes_b, 
                mask_thr=0.5,
            ):
    """
    input:
        img: 输入图像, numpy array of shape (H, W, 3)
        box_a: 主候选框 A, 格式为 (x0, y0, x1, y1)
        boxes_b: 一系列候选框 Bi, 格式为 [(xi0, yi0, xi1, yi1), ...]
        boxes_b_scores: 一系列候选框 Bi 的得分(已按从大到小排序)
        mask_thr: 面积遮罩阈值
    return:
        mask_PILimg: mask后的原图(PIL Image)
        attention_mask: 与mask_PILimg形状相同的01矩阵, 0代表该像素需要被mask(PIL Image)
        flag: 1则表示对图像进行了mask, 0则表示没有
    """
    mask_img = torch.tensor(copy.deepcopy(img),dtype=torch.float32) / 255.0
    attention_mask = torch.ones_like(mask_img)
    if len(boxes_b) == 0:
        return Image.fromarray(img[:,:,::-1]), Image.fromarray(attention_mask.numpy().astype(np.uint8)), 0
    box_a, boxes_b = torch.tensor(np.array(box_a)), torch.tensor(np.array(boxes_b))
    
    img_A_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    total_mask_area = 0

    normal_iou, bias_iou = get_2_type_boxes_iou(box_a[None, :], boxes_b)
    normal_iou, bias_iou = normal_iou[0], bias_iou[0]
    # 计算并遮盖重叠区域
    flag = 0
    for i, box in enumerate(boxes_b):
        # 计算交集
        box = scale_box(box, 1, max_H=img.shape[0], max_W=img.shape[1])
        x0 = int(max(box_a[0], box[0]))
        y0 = int(max(box_a[1], box[1]))
        x1 = int(min(box_a[2], box[2]))
        y1 = int(min(box_a[3], box[3]))

        iou_condition = (bias_iou[i]>0.1) & (normal_iou[i]<0.5)
        # 检查是否有交集
        if x0 < x1 and y0 < y1 and iou_condition:
            
            # 计算遮盖面积
            mask_area = (y1 - y0) * (x1 - x0)
            total_mask_area += mask_area
            if (total_mask_area/img_A_area) > mask_thr:
                continue
            # mask
            # print(x0, " ", x1, " ", y0, " ", y1, " ")
            # print(int(box[1]), " ", int(box[3]), " ", int(box[0]), " ", int(box[2]), " ")
            mask_img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :] = add_diffusion_noise(mask_img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :], noise_step=700, beta=0.01)
            attention_mask[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :] = 0
            flag = 1
    mask_img = (mask_img.numpy() * 255).clip(0, 255).astype(np.uint8)
    mask_pilImage = Image.fromarray(mask_img[:,:,::-1])

    return mask_pilImage, Image.fromarray(attention_mask.numpy().astype(np.uint8)), flag

def add_diffusion_noise(image_tensor, noise_step=999, beta=0.01):
        num_steps = 1000  # Number of diffusion steps

        # decide beta in each step
        betas = torch.linspace(-6,6,num_steps)
        betas = torch.sigmoid(betas) * (beta - 1e-5) + 1e-5

        # decide alphas in each step
        alphas = 1 - betas
        alphas_prod = torch.cumprod(alphas, dim=0)
        alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
        alphas_bar_sqrt = torch.sqrt(alphas_prod)
        one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)


        def q_x(x_0, t, mean, std):
            noise = torch.randn_like(x_0) * std + mean
            alphas_t = alphas_bar_sqrt[t]
            alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
            return (alphas_t * x_0 + alphas_1_m_t * noise)
        mean = torch.mean(image_tensor)
        std = torch.std(image_tensor)

        noise_delta = int(noise_step) # from 0-999
        noisy_image = image_tensor.clone()
        image_tensor_cd = q_x(noisy_image,noise_step, mean, std) 

        return image_tensor_cd

def generate_unique_filename(dir_path, base_filename, file_extension):
    i = 0
    while True:
        # 如果i为0，就不添加序号，从1开始添加
        if i == 0:
            new_filename = f"{base_filename}{file_extension}"
        else:
            new_filename = f"{base_filename}_{i}{file_extension}"
        new_file_path = os.path.join(dir_path, new_filename)
        # 检查是否存在同名文件
        if not os.path.exists(new_file_path):
            return new_file_path
        i += 1

def random_mask_image(
    img_size=(224, 224),
    mask_patch_size=(32, 32),
    mask_ratio=0.5
):
    """Randomly generate mask image with the given mask_ratio"""
    # Calculate total number of pixels and number of masked pixels
    total_pixels = img_size[0] * img_size[1]
    mask_pixels = int(total_pixels * mask_ratio)

    # Create an empty image
    mask = np.zeros(img_size, dtype=np.uint8)

    # Calculate number of patches based on mask patch size
    num_patches_x = img_size[0] // mask_patch_size[0]
    num_patches_y = img_size[1] // mask_patch_size[1]
    total_patches = num_patches_x * num_patches_y
    mask_patches = int(total_patches * mask_ratio)

    # Generate random positions for the patches
    positions = np.random.choice(total_patches, mask_patches, replace=False)

    # Place masked patches on the image
    for pos in positions:
        row = (pos // num_patches_y) * mask_patch_size[0]
        col = (pos % num_patches_y) * mask_patch_size[1]
        mask[row:row+mask_patch_size[0], col:col+mask_patch_size[1]] = 1

    return mask


def filt_gt_base_box(image_id, proposal_boxes, pp_scores, coco: COCO, iou_thresh=0.5):
    """
    proposal_boxes 为rpn生成的候选框, 该函数用于过滤掉proposal_boxes中与原用的base类标注重叠过高的框,
    过滤后剩余的候选框将用于clip生成伪标签
    input:
        proposal_boxes: (N, 4)
        pp_scores: (N, )
    output:
        new_proposal_boxes: (M, 4)
        new_pp_scores: (M, )
        gt_base_boxes: (p, 4)
    """
    # 获取该图片的所有annotation IDs
    annotation_ids = coco.getAnnIds(imgIds=image_id)
    
    # 获取该图片对应的所有annotations
    annotations = coco.loadAnns(annotation_ids)
    
    # 仅保留基类标注
    gt_base_anns = [item for item in annotations if item['category_id'] in BASE_CATEGORIES_ID]
    if len(gt_base_anns) == 0:
        return torch.tensor(proposal_boxes), torch.tensor(pp_scores), None
    
    # 提取基类标注的bounding boxes
    gt_base_boxes = torch.tensor([ann['bbox'] for ann in gt_base_anns])
    #xyhw to xyxy
    gt_base_boxes[:, 2] = gt_base_boxes[:, 0] + gt_base_boxes[:, 2]  # x_max = x_min + width
    gt_base_boxes[:, 3] = gt_base_boxes[:, 1] + gt_base_boxes[:, 3]  # y_max = y_min + height
    
    pp_gt_iou = box_iou(torch.tensor(proposal_boxes), gt_base_boxes)
    
    keep = pp_gt_iou.max(dim=1).values < iou_thresh
    
    # 过滤proposal_boxes和对应的pp_scores
    new_proposal_boxes = torch.tensor(proposal_boxes)[keep]
    new_pp_scores = torch.tensor(pp_scores)[keep]
    
    return new_proposal_boxes, new_pp_scores, gt_base_boxes



if __name__ == '__main__':
    import cv2
    img_file = "/data1/liangzhijia/datasets/coco/val2017/000000557672.jpg"
    img = cv2.imread(img_file)
    img_tensor = torch.tensor(img.transpose(2, 0, 1))
    mask = random_mask_image(img_tensor.shape[1:])
    new_img = img_tensor * mask[None,:,:]
    # save
    save_path = "./visual_test.png"
    cv2.imwrite(save_path, new_img.numpy().transpose(1, 2, 0))
    