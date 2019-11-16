#!/bin/bash

# Clone COCO API
git clone https://github.com/pdollar/coco
cd coco

mkdir images
cd images

# Download Images
wget -c https://pjreddie.com/media/files/train2014.zip
wget -c https://pjreddie.com/media/files/val2014.zip
wget -c http://images.cocodataset.org/zips/test2014.zip

# Unzip
unzip -q train2014.zip
unzip -q val2014.zip
unzip -q test2014.zip

cd ..

# Download COCO Metadata
wget -c https://pjreddie.com/media/files/instances_train-val2014.zip
unzip -q instances_train-val2014.zip
wget -c https://pjreddie.com/media/files/coco/5k.part
wget -c https://pjreddie.com/media/files/coco/trainvalno5k.part
wget -c https://pjreddie.com/media/files/coco/labels.tgz
tar xzf labels.tgz
wget -c http://images.cocodataset.org/annotations/image_info_test2014.zip
unzip image_info_test2014.zip

# Set Up Image Lists
ls -l images/test2014/ | awk '{print $(NF)}'| > test2014.part
paste <(awk "{print \"$PWD/images/test2014/\"}" <test2014.part) test2014.part | tr -d '\t' > test2014.txt
paste <(awk "{print \"$PWD\"}" <5k.part) 5k.part | tr -d '\t' > coco_val_5k.txt
paste <(awk "{print \"$PWD\"}" <trainvalno5k.part) trainvalno5k.part | tr -d '\t' > trainvalno5k.txt

#ls -l images/val2014/ | awk '{print $(NF)}'| > val2014.part
#paste <(awk "{print \"$PWD/images/val2014/\"}" <val2014.part) val2014.part | tr -d '\t' > val2014.txt

#ls -l images/val2017/ | awk '{print $(NF)}'| > val2017.part
#paste <(awk "{print \"$PWD/images/val2017/\"}" <val2017.part) val2017.part | tr -d '\t' > val2017.txt
