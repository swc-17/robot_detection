# yolox ps
 PYTHONPATH='.':$PYTHONPATH  mim train mmdet \
         configs/parking_slot/yolox_psvoc.py \
         --gpus 4  \
         --launcher pytorch 

# yolo ps
# PYTHONPATH='.':$PYTHONPATH  mim train mmdet \
#         configs/pascal_voc/yolo_1x_psvoc0712.py \
#         --gpus 4  \
#         --launcher pytorch 

# yolov3 coco
# PYTHONPATH='.':$PYTHONPATH  mim train mmdet \
#         configs/yolo/yolov3_d53_320_273e_coco.py \
#         --gpus 4  \
#         --launcher pytorch 

# faster-rcnn voc
# PYTHONPATH='.':$PYTHONPATH  mim train mmdet \
#         configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py \
#         --gpus 4  \
#         --launcher pytorch 

# faster-rcnn ps
#  PYTHONPATH='.':$PYTHONPATH  mim train mmdet \
#          configs/pascal_voc/faster_rcnn_r50_fpn_1x_psvoc0712.py \
#          --gpus 4  \
#          --launcher pytorch 

# faster-rcnn voc single_gpu
#PYTHONPATH='.':$PYTHONPATH  mim train mmdet \
#        configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py 

# faster-rcnn ps single gpu
# PYTHONPATH='.':$PYTHONPATH  mim train mmdet \
#         configs/pascal_voc/faster_rcnn_r50_fpn_1x_psvoc0712.py 
