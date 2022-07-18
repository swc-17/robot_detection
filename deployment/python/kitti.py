import os 
import sys

import cv2
import numpy as np

from utils.detector import Detector
from utils.decode import decode
from utils.visualize import add_2d_detection


from torch import Tensor
import torch
import torchvision
 
def box_area(boxes: Tensor) -> Tensor:
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
 
 
def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)  # 每个框的面积 (N,)
    area2 = box_area(boxes2)  # (M,)
 
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2] # N中一个和M个比较； 所以由N，M 个
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
 
    wh = (rb - lt).clamp(min=0)  # [N,M,2]  #小于0的为0  clamp 钳；夹钳；
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]  
 
    iou = inter / (area1[:, None] + area2 - inter)
    return iou  # NxM， boxes1中每个框和boxes2中每个框的IoU值；
 
 
def nms(boxes: Tensor, scores: Tensor, iou_threshold: float):
    """
    :param boxes: [N, 4]， 此处传进来的框，是经过筛选（NMS之前选取过得分TopK）之后， 在传入之前处理好的；
    :param scores: [N]
    :param iou_threshold: 0.7
    :return:
    """
    keep = []  # 最终保留的结果， 在boxes中对应的索引；
    idxs = scores.argsort()  # 值从小到大的 索引
 
    while idxs.numel() > 0:  # 循环直到null； numel()： 数组元素个数
        # 得分最大框对应的索引, 以及对应的坐标
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]  # [1, 4]
        keep.append(max_score_index)
        if idxs.size(0) == 1:  # 就剩余一个框了；
            break
        idxs = idxs[:-1]  # 将得分最大框 从索引中删除； 剩余索引对应的框 和 得分最大框 计算IoU；
        other_boxes = boxes[idxs]  # [?, 4]
        ious = box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
        idxs = idxs[ious[0] <= iou_threshold]
 
    keep = idxs.new(keep)  # Tensor
    return keep

HEIGHT_RESIZE = 320
WIDTH_RESIZE  = 320

def main():
    root = '/home/sunwenchao/robot/models'
    filepath = 'engine'
    img_path = "/home/sunwenchao/robot/test.jpg"

    filepath = os.path.join(root, filepath)
    detector = Detector(filepath,(HEIGHT_RESIZE ,WIDTH_RESIZE))

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (WIDTH_RESIZE, HEIGHT_RESIZE))

    import time
    s = time.time()
    for i in range(200):
        results = detector.inference(img)

        boxes = torch.from_numpy(results[0].reshape((-1, 4)))
        scores = torch.from_numpy(results[1].reshape((-1, 8)))
        scores, labels = torch.max(scores, dim=1)

        mask = scores > 0.3 
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        keep = nms(boxes, scores, 0.5)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
    
    e = time.time()
    print((e-s)/200)

    img = add_2d_detection(img, boxes, scores, labels)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("d.jpg", img)


if __name__ == '__main__':
    main()


# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
#       with open("/home/swc/CenterNet/src/model-fp32.trt", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
# 		      engine = runtime.deserialize_cuda_engine(f.read())
#       h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
#       h_output = [cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(i)), dtype=np.float32) for i in range(1,7)]
#       # Allocate device memory for inputs and outputs.
#       d_input = cuda.mem_alloc(h_input.nbytes)
#       d_output = [cuda.mem_alloc(h_output[i].nbytes) for i in range(6)]
#       # Create a stream in which to copy inputs/outputs and run inference.
#       stream = cuda.Stream()
#       bindings = [int(d_output[i]) for i in range(6)]
#       bindings.insert(0,int(d_input))

#       with engine.create_execution_context() as context:
#           # Transfer input data to the GPU.
#           cuda.memcpy_htod_async(d_input, h_input, stream)
#           # Run inference.
#           context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
#           # Transfer predictions back from the GPU.
#           [cuda.memcpy_dtoh_async(h_output[i], d_output[i], stream) for i in range(6)]
#           # Synchronize the stream
#           stream.synchronize()
