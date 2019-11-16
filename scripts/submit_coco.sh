#!/bin/bash
set -x
set -e
set -o pipefail

PRETRAINED_WEIGHTS=datasets/pretrained/yolov3-608.weights
PRETRAINED_ZIP=coco_results.pretrained608.zip
BASELINE_ITERATION=492000
BASELINE_ZIP=coco_results.json.baseline4.${BASELINE_ITERATION}.zip
IOU_ITERATION=470000
IOU_ZIPFILE=coco_results.json.iou-14.${IOU_ITERATION}.zip
GIOU_ITERATION=final
GIOU_RUN=12
GIOU_ZIPFILE=coco_results.json.giou-${GIOU_RUN}.${GIOU_ITERATION}.zip

## pretrained weights
#  test-dev2018
#./darknet detector valid cfg/coco.coco-pretrained-608.data cfg/yolov3.coco-pretrained-608.cfg ${PRETRAINED_WEIGHTS}
#pushd results
#zip $PRETRAINED_ZIP coco_results.json
#popd
#echo "scp capri24:src/nn/darknet/results/${PRETRAINED_ZIP} /tmp/"

## baseline
#  test-dev2018
./darknet detector valid cfg/testdev2018.bbox.coco-baseline4.data cfg/yolov3.coco-baseline4.cfg backup/coco-baseline4/yolov3_${BASELINE_ITERATION}.weights
pushd results
zip $BASELINE_ZIP coco_results.json
popd

## iou-14
#  test-dev2018
./darknet detector valid cfg/testdev2018.bbox.coco-iou-14.data cfg/yolov3.coco-iou-14.cfg backup/coco-iou-14/yolov3_${IOU_ITERATION}.weights
pushd results
zip $IOU_ZIPFILE coco_results.json
popd

## giou
#  test-dev2018
#./darknet detector valid cfg/testdev2018.bbox.coco-giou-${GIOU_RUN}.data cfg/yolov3.coco-giou-${GIOU_RUN}.cfg backup/coco-giou-${GIOU_RUN}/yolov3_${GIOU_ITERATION}.weights
./darknet detector valid cfg/testdev2018.bbox.coco-giou-${GIOU_RUN}.data cfg/yolov3.coco-giou-${GIOU_RUN}.cfg backup/coco-giou-${GIOU_RUN}/yolov3_${GIOU_ITERATION}.weights
pushd results
zip $GIOU_ZIPFILE coco_results.json
popd

set +x
echo "scp capri24:src/nn/darknet/results/${BASELINE_ZIP} /tmp/"
echo "scp capri24:src/nn/darknet/results/${IOU_ZIPFILE} /tmp/"
echo "scp capri24:src/nn/darknet/results/${GIOU_ZIPFILE} /tmp/"

