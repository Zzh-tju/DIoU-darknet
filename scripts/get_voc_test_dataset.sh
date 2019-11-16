#!/bin/bash

pushd datasets/voc
# probably wont work, need to login to download (copy curl): http://host.robots.ox.ac.uk:8080/eval/challenges/voc2012/
wget http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2012test.tar
tar xf VOC2012test.tar

#wget https://pjreddie.com/media/files/voc_label.py
#python voc_label.py

cat VOCdevkit/VOC2012/ImageSets/Main/test.txt|awk '{print "VOCdevkit/VOC2012/JPEGImages/"$1".jpg"}'>test2012.part
paste <(awk "{print \"$PWD/\"}" <test2012.part) test2012.part | tr -d '\t' > test2012.txt
popd
