#!/bin/bash

mkdir datasets/openimages

# get images
pushd datasets/openimages

# create image directories
mkdir train
mkdir validation
mkdir test
mkdir test_challenge_2018

# annotations
mkdir -p labels/train

# download the data
aws s3 --no-sign-request sync s3://open-images-dataset/train train
aws s3 --no-sign-request sync s3://open-images-dataset/validation validation
aws s3 --no-sign-request sync s3://open-images-dataset/test test
aws s3 --no-sign-request sync s3://open-images-dataset/challenge2018 test_challenge_2018
popd

pushd datasets/openimages
# annotations
wget https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv
wget https://storage.googleapis.com/openimages/2018_04/validation/validation-annotations-bbox.csv
wget https://storage.googleapis.com/openimages/2018_04/test/test-annotations-bbox.csv

# ids
wget https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv
wget https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv
wget https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv
popd

# generate lists
pushd datasets/openimages
find train -type f -name '*.jpg' | awk '{print $(NF)}'| > train.part
paste <(awk "{print \"$PWD/\"}" <train.part) train.part | tr -d '\t' > train.cvgl.txt
popd

# generate annotations
scripts/openimages_to_coco_annotations.py \
  --annotations_input_path datasets/openimages/train-annotations-bbox.csv \
  --image_index_input_path datasets/openimages/train-images-boxable-with-rotation.csv \
  --coco_output_path datasets/openimages/train-annotations-bbox.json \
  --darknet_output_path datasets/openimages/labels/train

