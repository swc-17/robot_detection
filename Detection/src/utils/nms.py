import numpy as np

def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.
    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format
    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
 
 
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
 
    wh = np.clip(rb - lt, 0, np.inf)    
    inter = wh[:, :, 0] * wh[:, :, 1]
 
    iou = inter / (area1[:, None] + area2 - inter)
    return iou
 
 
def nms(boxes, scores, iou_threshold):
    """
    :param scores: [N]
    :param iou_threshold: 0.7
    :return:
    """
    keep = []  
    idxs = scores.argsort()
 
    while idxs.size > 0: 
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]  
        keep.append(max_score_index)
        if idxs.shape[0] == 1: 
            break
        idxs = idxs[:-1]  
        other_boxes = boxes[idxs] 
        ious = box_iou(max_score_box, other_boxes) 
        idxs = idxs[ious[0] <= iou_threshold]
 
    keep = np.array(keep)    
    return keep
