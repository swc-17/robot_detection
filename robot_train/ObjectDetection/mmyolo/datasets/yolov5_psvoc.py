# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets import VOCDataset

from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from ..registry import DATASETS


@DATASETS.register_module()
class YOLOv5PSVOCDataset(BatchShapePolicyDataset, VOCDataset):
    """Dataset for YOLOv5 VOC Dataset.

    We only add `BatchShapePolicy` function compared with VOCDataset. See
    `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    # '道闸','地面垃圾','地锁','减速带','路障','限位器','雪糕筒'
    METAINFO = {
        'classes':
        ('barrier_gate', 'garbage','parking_lock', 'speed_bump', 'road_block', 'stopper', 'traffic_cone'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157), (0, 182, 199),
                    (0, 226, 252), (182, 182, 255), (0, 0, 230), (220, 20, 60),
                    (163, 255, 0), (0, 82, 0), (3, 95, 161), (0, 80, 100),
                    (183, 130, 88)]
    }