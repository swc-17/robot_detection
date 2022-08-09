# robot_detection
This repo is for train and deploy yolov3 based on mmdet and self-collected VOC-style dataset.
## Train Phase
### 1. Environment
* mmcv-full==1.4.7
* mmdet==2.22.0
* torch==1.10.2+cu113

### 2. Train
    sh scripts/train.sh

### 3. Test
    sh scripts/test.sh

## Deployment
### 1. Environment
Hardware: Nvidia AGX Xavier(ARM)  
Software: 
* JetPack 4.4
* CUDA 10.2
* TensorRT 7.1.3
* CUDNN 8.0.0
* onnx-tensorrt 7.2.1

### 1. pth to onnx
* add `return batch_mlvl_bboxes, batch_mlvl_scores` in mmdet/models/dense_heads/yolo_head.py line 601 to avoid export nms
* sh scripts/torch2onnx.sh
* onnx model can also be downloaded from [model](https://pan.baidu.com/s/1wTMtsXXfzf3ASpsvQNxO_A)(code:jzfn)

### 2. onnx to tensorrt
* build onnx-tensorrt and replace libonnxparser.so in trt with the one built in onnx-tensorrt
* run `cd Detection && sh Detection/models/trt.sh`

### 3. Inference
    python3 src/main.py
