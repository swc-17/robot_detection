
# distributed
#     --launcher pytorch \
# yolo psvoc


# export PYTHONPATH='.':$PYTHONPATH  
# mim test mmdet \
# 	configs/parking_slot/yolov3_psvoc.py \
# 	--gpus 1  \
# 	--checkpoint work_dirs/yolo_1x_psvoc0712/latest.pth \
# 	--eval mAP
	#--show-dir demo

export PYTHONPATH='.':$PYTHONPATH  
mim test mmdet \
	configs/parking_slot/yolox_psvoc.py \
	--gpus 4  \
	--checkpoint work_dirs/yolox_nano_8x8_300e_coco/latest.pth \
	--launcher pytorch \
	--eval mAP
