#!/bin/bash

# Clone COCO API
git clone https://github.com/pdollar/coco
cd coco

mkdir -p images
cd images

# Download Images
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/zips/test2017.zip

# Unzip
unzip -q train2017.zip
unzip -q val2017.zip
unzip -q test2017.zip

cd ..

# Download COCO Metadata
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

# Test image metadata
wget -c http://images.cocodataset.org/annotations/image_info_test2017.zip
unzip image_info_test2017.zip

# Set Up Image Lists
# test
ls -l images/test2017/*.jpg | awk '{print $(NF)}'| > test2017.part
paste <(awk "{print \"$PWD/\"}" <test2017.part) test2017.part | tr -d '\t' > test2017.txt
# test-dev
cat annotations/image_info_test-dev2017.json|jq '.images[]|.coco_url'|awk 'match($0,/test2017\/([^"]*)/) { print substr( $0, RSTART, RLENGTH )}' > testdev2017.part
paste <(awk "{print \"$PWD/images/\"}" <testdev2017.part) testdev2017.part | tr -d '\t' > testdev2017.txt
#paste <(awk "{print \"$PWD\"}" <5k.part) 5k.part | tr -d '\t' > coco_val_5k.txt
#paste <(awk "{print \"$PWD\"}" <trainvalno5k.part) trainvalno5k.part | tr -d '\t' > trainvalno5k.txt


