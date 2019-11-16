#!/bin/bash
# voc
python scripts/voc_all_map.py --data_file cfg/voc.yolov3-giou-38.data --cfg_file cfg/yolov3-voc.yolov3-giou-38.cfg --weights_folder backup/yolov3-giou-38/
python scripts/voc_all_map.py --data_file cfg/voc.yolov3-giou-39.data --cfg_file cfg/yolov3-voc.yolov3-giou-39.cfg --weights_folder backup/yolov3-giou-39/
python scripts/voc_all_map.py --data_file cfg/voc.yolov3-giou-40.data --cfg_file cfg/yolov3-voc.yolov3-giou-40.cfg --weights_folder backup/yolov3-giou-40/

python scripts/voc_all_map.py --data_file cfg/coco.coco-iou-1.25.data --cfg_file cfg/yolov3.coco-iou-1.25.cfg --weights_folder backup/coco-iou-1.25 --lib_folder lib
python scripts/voc_all_map.py --data_file cfg/coco.coco-giou-1.25.data --cfg_file cfg/yolov3.coco-giou-1.25.cfg --weights_folder backup/coco-giou-1.25  --lib_folder lib --gpu_id 1
python scripts/voc_all_map.py --data_file cfg/coco.coco-baseline2.data --cfg_file cfg/yolov3.coco-baseline2.cfg --weights_folder backup/coco-baseline2  --lib_folder lib

#coco
python scripts/coco_all_map.py --data_file cfg/coco.coco-baseline4.data --cfg_file cfg/yolov3.coco-baseline4.cfg --weights_folder backup/coco-baseline4/

python scripts/coco_all_map.py --data_file cfg/coco-giou-4.data --cfg_file cfg/yolov3.coco-giou-4.cfg --weights_folder backup/coco-giou-4
