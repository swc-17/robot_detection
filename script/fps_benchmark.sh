# yolov3 coco
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py \
       configs/yolo/yolov3_d53_320_273e_coco.py \
       models/yolov3_d53_320_273e_coco-421362b6.pth \
       --launcher pytorch

