# yolov3 coco
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py \
       configs/yolo/yolov3_d53_320_273e_coco.py \
       models/yolov3_d53_320_273e_coco-421362b6.pth \
       --launcher pytorch

# faster-rcnn voc
# python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py \
#        configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py \
#        models/faster_rcnn_r50_fpn_1x_voc0712_20220320_192712-54bef0f3.pth \
#        --launcher pytorch