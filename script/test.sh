export PYTHONPATH='.':$PYTHONPATH  
mim test mmdet \
	configs/parking_slot/yolov3_psvoc.py \
	--gpus 4  \
	--checkpoint work_dirs/yolov3_psvoc/latest.pth \
	--eval mAP \
	--launcher pytorch
	#--show-dir demo
