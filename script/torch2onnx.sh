
# faster-rcnn voc
# python tools/deployment/pytorch2onnx.py \
#     configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py \
#     models/faster_rcnn_r50_fpn_1x_voc0712_20220320_192712-54bef0f3.pth \
#     --output-file faster_rcnn_r50_fpn_1x_psvoc.onnx \
#     --shape 1000 600 \
#     --verify

# # yolov3 coco
# python tools/deployment/pytorch2onnx.py \
#     configs/yolo/yolov3_d53_320_273e_coco.py \
#     models/yolov3_d53_320_273e_coco-421362b6.pth \
#     --output-file yolo_coco.onnx \
#     --shape 320 
#     # --verify

# yolo psvoc
# PYTHONPATH='.':$PYTHONPATH \
# python tools/deployment/pytorch2onnx.py \
#     configs/parking_slot/yolov3_psvoc.py \
#     work_dirs/yolo_1x_psvoc0712/latest.pth \
#     --output-file yolo_ps.onnx \
#     --shape 416 \
#     --simplify 
    # --input-img test.jpg 
    # --skip-postprocess

# polygraphy surgeon sanitize yolo_ps.onnx \
#     --fold-constants \
#     -o yolo_ps_sim.onnx

# yolo psvoc
PYTHONPATH='.':$PYTHONPATH \
python tools/deployment/pytorch2onnx.py \
    configs/parking_slot/yolox_psvoc.py \
    work_dirs/yolox_psvoc/latest.pth \
    --output-file yolox.onnx \
    --shape 416 \
    --simplify 
    # --input-img test.jpg 
    # --skip-postprocess
