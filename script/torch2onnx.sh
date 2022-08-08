PYTHONPATH='.':$PYTHONPATH \
python tools/deployment/pytorch2onnx.py \
    configs/parking_slot/yolov3_psvoc.py \
    work_dirs/yolov3_psvoc/latest.pth \
    --output-file models/yolov3_psvoc.onnx \
    --shape 416 \
    --simplify 
    # --input-img test.jpg 
    # --skip-postprocess

# polygraphy surgeon sanitize models/yolov3_psvoc.onnx \
#     --fold-constants \
#     -o models/yolov3_psvoc.onnx
