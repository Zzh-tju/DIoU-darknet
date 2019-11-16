#!/bin/bash

NETWORK_DIR=/cvgl2/u/ntsoi/src/nn/darknet
PROJECT_DIR=/scr/ntsoi/darknet

mkdir -p $PROJECT_DIR
mkdir -p $PROJECT_DIR/datasets/coco
mkdir -p $PROJECT_DIR/datasets/pretrained
mkdir -p $PROJECT_DIR/backup
mkdir -p $PROJECT_DIR/batch/out
mkdir -p $PROJECT_DIR/backup/$RUN_NAME/*
# copy in latest project dir
rsync --update -raz --progress $NETWORK_DIR/* "$PROJECT_DIR/" --exclude datasets --exclude backup --exclude batch --exclude darkboard --exclude results
# copy in pretrained weights
rsync --update -raz --progress $NETWORK_DIR/datasets/pretrained/* "$PROJECT_DIR/datasets/pretrained/"
# copy in voc dataset
rsync --update -raz --progress $NETWORK_DIR/datasets/voc/* "$PROJECT_DIR/datasets/voc/" --exclude '*.tar'
# copy in coco dataset
rsync --update -raz --progress $NETWORK_DIR/datasets/coco/* "$PROJECT_DIR/datasets/coco/" --exclude '*.zip' --exclude 'coco/*.txt'
# copy in backup, as specified by the RUN_NAME
#rsync --update -raz --progress $NETWORK_DIR/backup/$RUN_NAME/* "$PROJECT_DIR/backup/$RUN_NAME/" --exclude '*.txt'
# adjust paths to images
cd $PROJECT_DIR/datasets/coco/coco
paste <(awk "{print \"$PWD\"}" <5k.part) 5k.part | tr -d '\t' > coco_val_5k.txt
paste <(awk "{print \"$PWD\"}" <trainvalno5k.part) trainvalno5k.part | tr -d '\t' > trainvalno5k.txt
head trainvalno5k.txt

cd $PROJECT_DIR/datasets/voc
cp $PROJECT_DIR/scripts/voc_label.py .
python voc_label.py
cat 2007_train.txt 2007_val.txt 2012_*.txt > train.txt
# adjust paths to images for:
#train  = datasets/voc/train.txt
#valid  = datasets/voc/2007_test.txt
#paste <(awk "{print \"$PWD/\"}" <test2012.part) test2012.part | tr -d '\t' > test2012.txt


