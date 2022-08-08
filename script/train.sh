PYTHONPATH='.':$PYTHONPATH  mim train mmdet \
        configs/parking_slot/yolov3_psvoc.py \
        --gpus 4  \
        --launcher pytorch 
