#!/bin/bash
RESFILE=datasets/voc/val2012.part
echo '' > $RESFILE
cat datasets/voc/VOCdevkit/VOC2012/ImageSets/Main/aeroplane_val.txt | awk '{print $1".jpg"}' >> $RESFILE
cat datasets/voc/VOCdevkit/VOC2012/ImageSets/Main/bicycle_val.txt | awk '{print $1".jpg"}' >> $RESFILE
cat datasets/voc/VOCdevkit/VOC2012/ImageSets/Main/bird_val.txt | awk '{print $1".jpg"}' >> $RESFILE
cat datasets/voc/VOCdevkit/VOC2012/ImageSets/Main/boat_val.txt | awk '{print $1".jpg"}' >> $RESFILE
cat datasets/voc/VOCdevkit/VOC2012/ImageSets/Main/bottle_val.txt | awk '{print $1".jpg"}' >> $RESFILE
cat datasets/voc/VOCdevkit/VOC2012/ImageSets/Main/bus_val.txt | awk '{print $1".jpg"}' >> $RESFILE
cat datasets/voc/VOCdevkit/VOC2012/ImageSets/Main/car_val.txt | awk '{print $1".jpg"}' >> $RESFILE
cat datasets/voc/VOCdevkit/VOC2012/ImageSets/Main/cat_val.txt | awk '{print $1".jpg"}' >> $RESFILE
cat datasets/voc/VOCdevkit/VOC2012/ImageSets/Main/chair_val.txt | awk '{print $1".jpg"}' >> $RESFILE
cat datasets/voc/VOCdevkit/VOC2012/ImageSets/Main/cow_val.txt | awk '{print $1".jpg"}' >> $RESFILE
cat datasets/voc/VOCdevkit/VOC2012/ImageSets/Main/diningtable_val.txt | awk '{print $1".jpg"}' >> $RESFILE
cat datasets/voc/VOCdevkit/VOC2012/ImageSets/Main/dog_val.txt | awk '{print $1".jpg"}' >> $RESFILE
cat datasets/voc/VOCdevkit/VOC2012/ImageSets/Main/horse_val.txt | awk '{print $1".jpg"}' >> $RESFILE
cat datasets/voc/VOCdevkit/VOC2012/ImageSets/Main/motorbike_val.txt | awk '{print $1".jpg"}' >> $RESFILE
cat datasets/voc/VOCdevkit/VOC2012/ImageSets/Main/person_val.txt | awk '{print $1".jpg"}' >> $RESFILE
cat datasets/voc/VOCdevkit/VOC2012/ImageSets/Main/pottedplant_val.txt | awk '{print $1".jpg"}' >> $RESFILE
cat datasets/voc/VOCdevkit/VOC2012/ImageSets/Main/sheep_val.txt | awk '{print $1".jpg"}' >> $RESFILE
cat datasets/voc/VOCdevkit/VOC2012/ImageSets/Main/sofa_val.txt | awk '{print $1".jpg"}' >> $RESFILE
cat datasets/voc/VOCdevkit/VOC2012/ImageSets/Main/train_val.txt | awk '{print $1".jpg"}' >> $RESFILE
cat datasets/voc/VOCdevkit/VOC2012/ImageSets/Main/tvmonitor_val.txt | awk '{print $1".jpg"}' >> $RESFILE

DSPATH=datasets/voc
paste <(awk "{print \"$PWD/datasets/voc/VOCdevkit/VOC2012/JPEGImages/\"}" < $DSPATH/val2012.part) $DSPATH/val2012.part | tr -d '\t' > $DSPATH/val2012.txt

