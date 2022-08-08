PYTHONPATH='.':$PYTHONPATH \
python tools/deployment/pytorch2onnx.py \
    configs/parking_slot/yolov3_psvoc.py \
    work_dirs/yolov3_psvoc/latest.pth \
    --output-file models/yolov3_psvoc.onnx \
    --shape 416 \
    --simplify \
    --skip-postprocess

# polygraphy surgeon sanitize yolo_ps.onnx \
#     --fold-constants \
#     -o yolo_ps_sim.onnx

# yolo psvoc
# PYTHONPATH='.':$PYTHONPATH \
# python tools/deployment/pytorch2onnx.py \
#     configs/parking_slot/yolox_psvoc.py \
#     work_dirs/yolox_psvoc/latest.pth \
#     --output-file yolox.onnx \
#     --shape 416 \
#     --simplify 
#     # --input-img test.jpg 
#     # --skip-postprocess
