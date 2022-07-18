# faster-rcnn ps
#PYTHONPATH='.':$PYTHONPATH  mim test mmdet \
#    configs/pascal_voc/faster_rcnn_r50_fpn_1x_psvoc0712.py \
#    --gpus 1 \
#    --checkpoint models/faster_rcnn_r50_fpn_1x_psvoc.pth \
#    --eval mAP 
    # --show-dir show

# faster-rcnn voc
# PYTHONPATH='.':$PYTHONPATH  mim test mmdet \
#     configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py \
#     --gpus 1 \
#     --checkpoint models/faster_rcnn_r50_fpn_1x_voc0712_20220320_192712-54bef0f3.pth \
#     --eval mAP
#     --show-dir show

# yolov3 coco
# PYTHONPATH='.':$PYTHONPATH  mim test mmdet \
#     configs/yolo/yolov3_d53_320_273e_coco.py \
#     --gpus 1 \
#     --checkpoint models/yolov3_d53_320_273e_coco-421362b6.pth \
#     --eval bbox

# distributed
#     --launcher pytorch \
# yolo psvoc


export PYTHONPATH='.':$PYTHONPATH  
mim test mmdet \
	configs/pascal_voc/yolo_1x_psvoc0712.py \
	--gpus 1  \
	--checkpoint work_dirs/yolo_1x_psvoc0712/latest.pth \
	--eval mAP
	#--show-dir demo

