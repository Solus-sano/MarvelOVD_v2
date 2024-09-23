import os
import json
from detectron2.evaluation import (
    COCOEvaluator
)
from detectron2.evaluation import print_csv_format

from detectron2.utils.logger import create_small_table
from typing import List, Union
from detectron2.evaluation import DatasetEvaluator, DatasetEvaluators
from collections import OrderedDict, abc
from meta_data import BASE_CATEGORIES, EVAL_CATEGORIES
import numpy as np
from tabulate import tabulate
import itertools

from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup
import logging
import meta_data
from detectron2.data import (
    MetadataCatalog,
)
import copy
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

logger = logging.getLogger("detectron2")

def add_detector_config(cfg):
    _C = cfg
    _C.SOLVER.RESET_ITER = False
    _C.SOLVER.TRAIN_ITER = -1

    _C.MODEL.ROI_BOX_HEAD.EMB_DIM = 512
    _C.MODEL.ROI_BOX_HEAD.LOSS_WEIGHT_BACKGROUND = 1.0

    _C.INPUT.TRAIN_SIZE = 640
    _C.INPUT.TEST_SIZE = 640
    _C.INPUT.SCALE_RANGE = (0.1, 2.)
    # 'default' for fixed short/ long edge, 'square' for max size=INPUT.SIZE
    _C.INPUT.TEST_INPUT_TYPE = 'default' 


class COCO_json_evaluator(COCOEvaluator):
    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics

        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        print(len(class_names))
        print(precisions.shape)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[0, :, idx, 0, -1]  # calculate ap50
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))
        # self._logger.info(
        #     "Evaluation results for AP50 \n" + create_small_table({
        #         "results_base": np.mean([i[1] for i in results_per_category if i[0] in BASE_CATEGORIES]),
        #         "results_novel": np.mean([i[1] for i in results_per_category if i[0] in EVAL_CATEGORIES])
        #     })
        # )
        results_base = np.mean([i[1] for i in results_per_category if i[0] in BASE_CATEGORIES])
        results_novel = np.mean([i[1] for i in results_per_category if i[0] in EVAL_CATEGORIES])
        results_all = np.mean([i[1] for i in results_per_category])
        self._logger.info(
            "Evaluation results for AP50 \n" + create_small_table({
                "results_base": results_base,
                "results_novel": results_novel
            })
        )
        if iou_type=="bbox":
            self._logger.info("eval_result AP50@Novel,Base,All:"+" {:.3f}".format(results_novel)+", {:.3f}".format(results_base)+", {:.3f}".format(results_all))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP50"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP50: \n".format(iou_type) + table)

        results.update({"AP50-" + name: ap for name, ap in results_per_category})
        return results
    
    def process_json(self, coco_dt):
        tmp_map = {}
        thing_dataset_id_to_contiguous_id = MetadataCatalog.get("coco_openvoc_train").thing_dataset_id_to_contiguous_id
        for item in coco_dt["annotations"]:
            result = {
                "image_id": item["image_id"],
                "category_id": thing_dataset_id_to_contiguous_id[item["category_id"]],
                "bbox": item["bbox"],
                "score": item["score"],
            }
            if tmp_map.get(item["image_id"], None) is None:
                tmp_map[item["image_id"]] = [result]
            else:
                tmp_map[item["image_id"]].append(result)
        
        for key, item in tmp_map.items():
            prediction = {"image_id": key, "instances": item}
            self._predictions.append(prediction)

def inference_on_json(
    coco_dt_json_file,
    evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None],
    callbacks=None,
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.
        callbacks (dict of callables): a dictionary of callback functions which can be
            called at each stage of inference.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    logger = logging.getLogger(__name__)
    coco_dt_json = json.load(open(coco_dt_json_file, "r"))
    logger.info(f"Evaluating predictions saved in {coco_dt_json_file} ...")
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    evaluator.process_json(coco_dt_json)

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_detector_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if '/auto' in cfg.OUTPUT_DIR:
        file_name = os.path.basename(args.config_file)[:-5]
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/auto', '/{}'.format(file_name))
        logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def evaluate():
    args = default_argument_parser()
    args.add_argument("--config_file", type=str, default="/data1/liangzhijia/MarvelOVD_v2/configs/coco_ssod.yaml", help="path to config file")
    args = args.parse_args()
    args.config_file = "/data1/liangzhijia/MarvelOVD_v2/configs/coco_ssod.yaml"
    print(args)
    cfg = setup(args)
    
    PLs_json_file = "/data1/liangzhijia/datasets/coco/annotations/open_voc/score_train_novel_candidate_0.5.json"
    
    evaluator = COCO_json_evaluator("coco_train", cfg, False)
    result = inference_on_json(PLs_json_file, evaluator)

    print_csv_format(result)
    
def create_gt_json():
    data = load_json("/data1/liangzhijia/datasets/coco/annotations/instances_train2017.json")
    # dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
    new_ann = []
    new_category = []
    meta = MetadataCatalog.get("coco_train")
    thins_id_to_continugous_id = meta.thing_dataset_id_to_contiguous_id
    meta_thing_ids = list(thins_id_to_continugous_id.keys())
    print(len(meta_thing_ids))
    print("init data len: ", len(data['annotations']))
    print("init category len: ", len(data['categories']))
    
    for ann in tqdm(data['annotations']):
        catID = ann['category_id']
        if catID in meta_thing_ids:
            new_ann.append(ann)
    
    for cate in tqdm(data['categories']):
        catID = cate['id']
        if catID in meta_thing_ids:
            new_category.append(cate)
    
    data['annotations'] = new_ann
    data['categories'] = new_category
    print("after filter data len: ", len(data['annotations']))
    print("after filter category len: ", len(data['categories']))
    new_file = "/data1/liangzhijia/datasets/coco/annotations/instances_cat65_train2017.json"
    
    with open(new_file, 'w') as f:
        json.dump(data, f)
    
if __name__ == '__main__':
    # create_gt_json()
    evaluate()