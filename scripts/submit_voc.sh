#!/bin/bash
set -x

DTYPE=test
rm results/results/VOC2012/Main/*

set -e
set -o pipefail
# baseline
#TGZNAME=$DTYPE-baseline2
#./darknet detector valid cfg/voc.yolov3-baseline2.data cfg/yolov3-voc.yolov3-baseline2.cfg backup/yolov3-baseline2/yolov3-voc_50000.weights

## baseline test
##./darknet detector test cfg/voc.yolov3-baseline2.data cfg/yolov3-voc.yolov3-baseline2.cfg backup/yolov3-baseline2/yolov3-voc_50000.weights /scr/ntsoi/darknet/datasets/voc/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg

## giou (maybe overfitting)
#TGZNAME=$DTYPE-giou-40-81000
#./darknet detector valid cfg/voc.yolov3-giou-40.data cfg/yolov3-voc.yolov3-giou-40.cfg backup/yolov3-giou-40/yolov3-voc_81000.weights

# giou at 51000
ITERATION=51000
TGZNAME=$DTYPE-giou-40-$ITERATION
./darknet detector valid cfg/voc.yolov3-giou-40.data cfg/yolov3-voc.yolov3-giou-40.cfg backup/yolov3-giou-40/yolov3-voc_$ITERATION.weights

## giou test
##./darknet detector test cfg/voc.yolov3-giou-40.data cfg/yolov3-voc.yolov3-giou-40.cfg backup/yolov3-giou-40/yolov3-voc_81000.weights /scr/ntsoi/darknet/datasets/voc/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg


#./darknet detector test cfg/voc.yolov3-giou-40.data cfg/yolov3-voc.yolov3-giou-40.cfg backup/yolov3-giou-40/yolov3-voc_81000.weights /scr/ntsoi/darknet/datasets/voc/VOCdevkit/VOC2012/JPEGImages/2008_000002.jpg

pushd results
mkdir -p results/VOC2012/Main
mv comp4_det_test_aeroplane.txt results/VOC2012/Main/comp4_det_${DTYPE}_aeroplane.txt
mv comp4_det_test_bicycle.txt results/VOC2012/Main/comp4_det_${DTYPE}_bicycle.txt
mv comp4_det_test_bird.txt results/VOC2012/Main/comp4_det_${DTYPE}_bird.txt
mv comp4_det_test_boat.txt results/VOC2012/Main/comp4_det_${DTYPE}_boat.txt
mv comp4_det_test_bottle.txt results/VOC2012/Main/comp4_det_${DTYPE}_bottle.txt
mv comp4_det_test_bus.txt results/VOC2012/Main/comp4_det_${DTYPE}_bus.txt
mv comp4_det_test_car.txt results/VOC2012/Main/comp4_det_${DTYPE}_car.txt
mv comp4_det_test_cat.txt results/VOC2012/Main/comp4_det_${DTYPE}_cat.txt
mv comp4_det_test_chair.txt results/VOC2012/Main/comp4_det_${DTYPE}_chair.txt
mv comp4_det_test_cow.txt results/VOC2012/Main/comp4_det_${DTYPE}_cow.txt
mv comp4_det_test_diningtable.txt results/VOC2012/Main/comp4_det_${DTYPE}_diningtable.txt
mv comp4_det_test_dog.txt results/VOC2012/Main/comp4_det_${DTYPE}_dog.txt
mv comp4_det_test_horse.txt results/VOC2012/Main/comp4_det_${DTYPE}_horse.txt
mv comp4_det_test_motorbike.txt results/VOC2012/Main/comp4_det_${DTYPE}_motorbike.txt
mv comp4_det_test_person.txt results/VOC2012/Main/comp4_det_${DTYPE}_person.txt
mv comp4_det_test_pottedplant.txt results/VOC2012/Main/comp4_det_${DTYPE}_pottedplant.txt
mv comp4_det_test_sheep.txt results/VOC2012/Main/comp4_det_${DTYPE}_sheep.txt
mv comp4_det_test_sofa.txt results/VOC2012/Main/comp4_det_${DTYPE}_sofa.txt
mv comp4_det_test_train.txt results/VOC2012/Main/comp4_det_${DTYPE}_train.txt
mv comp4_det_test_tvmonitor.txt results/VOC2012/Main/comp4_det_${DTYPE}_tvmonitor.txt
tar cvzf results.${TGZNAME}.tgz results
popd
pwd
echo "scp capri24:src/nn/darknet/results/results.${TGZNAME}.tgz /tmp/"
