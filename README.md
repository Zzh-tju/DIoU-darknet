# DIoU
Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression (AAAI 2020)

# DC-Darknet

YOLOv3 with DIoU and CIoU losses implemented in Darknet

If you use this work, please consider citing:

```
@article{Zhaohui_Zheng_2020_AAAI,
  author    = {Zhaohui Zheng, Ping Wang, Wei Liu, Jinze Li, Rongguang Ye, Dongwei Ren},
  title     = {Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression},
  booktitle = {The AAAI Conference on Artificial Intelligence (AAAI)},
  month     = {February},
  year      = {2020},
}
```

## Modifications in this repository

This repository contains a YoloV3 implementation of the IoU, GIoU, DIoU and CIoU losses while keeping the code as close to the GDarknet as possible. It is also possible to train with MSE loss as well, see the options below. 

### Losses

The loss can be chosen with the `iou_loss` option in the `.cfg` file and must be specified on each `[yolo]` layer. The valid options are currently: `[iou|giou|diou|ciou|mse]`

```
iou_loss=mse
```

### Normalizers

We also implement a normalizer between the localization and classification loss. These can be specified with the `cls_normalizer` and `iou_normalizer` parameters on the `[yolo]` layers. The default values are `1.0` for both. In our constrained search, the following values appear to work well for the loss.

```
iou_loss=diou
cls_normalizer=1
iou_normalizer=1.0
```
```
iou_loss=ciou
cls_normalizer=1
iou_normalizer=0.5
```
### DIoU-NMS

NMS can be chosen with the `nms_kind` option in the `.cfg` file and must be specified on each `[yolo]` layer. The valid options are currently: `[greedynms|diounms]`

```
nms_kind=greedynms
```
```
nms_kind=diounms
```
Besides that, we also found that for YOLOv3, we introduce beta1 for DIoU-NMS, that is IoU - R_DIoU ^ {beta1}. With this operation, DIoU-NMS can perform better than default beta1=1.0.

In our constrained search, the following values appear to work well for the DIoU-NMS.
```
beta1=0.6
```

While for SSD and Faster R-CNN， beta1 can be 1.0 which is good enough. Of course, beta1=1.0 for YOLOv3 is still better than greedy-NMS.
### Data

#### Augmentation

It has been reported that the custom data augmentation code in the original [Darknet repository](https://github.com/pjreddie/darknet) is a significant bottleneck during training. To this end, we have replaced the data loading and augmentation with the OpenCV implementation in [AlexeyAB's fork](https://github.com/AlexeyAB/darknet).

#### Output Prefix

To enable multiple simultaneous runs of the network, we have added a parameter named `prefix` to the `.data` config file.

This parameter should be set to your run name and will be used in the appropriate places to separate output by prefix per running instance.

## Scripts

A description of the scripts contained in this repository follows.

### Data pre-processing

see: `scripts/get_2017_coco_dataset.sh`

### Evaluation

See `scripts/voc_all_map.py` for VOC evaluation and  `scripts/coco_all_map.py` for COCO evaluation and `scripts/crontab.tmpl` for usage

### sbatch (that we did not use)

At Stanford, we use Slurm to manage the shared resources in our computing clusters

The [batch] directory contains the `sbatch` launch scripts for our cluster. Each script contain the bash commands used to start the network for a given test run.

## Visualization (that we did not use)

We have created a visualization tool, named Darkboard, to plot data generated during training. Though the implementation was quick and dirty, this tool is useful in evaluating network performance.

Details on running Darkboard can be found in the [/darkboard/README.md]() file.

## Pre-trained Models

See the [Workflow](#workflow) and [Evaluation](#evaluation) sections below for details on how to use these files

|Link|Save as local file|cfg used for training|
|--|--|--|
|http://bit.ly/2JZ9uMc|backup/coco-baseline4/yolov3_492000.weights|cfg/runs/coco-baseline4/yolov3.coco-baseline4.cfg|
|http://bit.ly/2TRNa6V|backup/coco-giou-12/yolov3_final.weights|cfg/runs/coco-giou-12/yolov3.coco-giou-12.cfg|

## Workflow

When training the network I used 2 GPUs on one Ubuntu machine.

To make running on various machines easier, I use the `scripts/package_libs.sh` script to pull all dependencies of darknet and place them in a single folder (`lib`).

For each test run I create the following new files:

|file name|purpose|
|---|---|
|cfg/[run name].data|data sources for train and validation data as well as the run prefix setting (which, by convention I always to [run name])|
|cfg/[run name].cfg|network configuration including loss, normalizers and representation|


Note that the `cfg/[run name].cfg` file contains parameters that must be changed when changing the number of GPUs used for training.


```
./darknet detector train cfg/voc-diou.data cfg/voc-diou.cfg darknet53.conv.74 -gpus 0,1
```

I change `cfg/[run name].cfg`, decreasing the `learning_rate` by setting `NEW_RATE = ORIGINAL_RATE * 1/NUMBER_OF_GPUS` and increasing the `burn_in` setting it to `NEW_BURN_IN = ORIGINAL_BURN_IN * NUMBER_OF_GPUS`

So for one GPU, the relevant portion of the `.cfg` file would be:

    learning_rate=0.001
    burn_in=1000

And for two GPUs, the relevant portion of the `.cfg` file would be:

    learning_rate=0.0005
    burn_in=2000

And for four GPUs, the relevant portion of the `.cfg` file would be:

    learning_rate=0.00025
    burn_in=4000

Then, resume the run from a specific iteration's weight file or in the case below, the backup, passing in the GPUs to run with using:

```
./darknet detector train cfg/voc-diou.data cfg/voc-diou.cfg backup/yourpath/voc-diou_30000.weights -gpus 0,1
```
Note that the burn_in must be 0 when you resume training.

## Training YOLO on VOC

Get The Pascal VOC Data:

```
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
```
Then
``` 
python scripts/voc_label.py
```

Put them in the following dir
VOCdevkit
├── VOC2007
    ├──Annotations
    ├──ImageSets
    ├──JPEGImages
    ├──labels
    ├──SegmentationClass
    ├──SegmentationObject
├── VOC2012
    ├──Annotations
    ├──ImageSets
    ├──JPEGImages
    ├──labels
    ├──SegmentationClass
    ├──SegmentationObject
	
Now, yourpath/d-darknet-master/ will have several txt file like this:
2007_test.txt, train.txt

Training set contains 16551 images, and validation set contains 4952 images.

## Evaluation

This repository contains tools for running ongoing evaluation while training the network.

### VOC

Evaluate all weights files in the given `weights_folder` with both the IoU and GIoU metrics using the following script:

    python scripts/voc_all_map.py --data_file cfg/yolov3-voc-lin-1.data --cfg_file cfg/yolov3-voc-lin-1.cfg --weights_folder backup/yolov3-voc-lin-1/

When you finish the training, you can validate it:

```
./darknet detector valid voc-diou.data voc-diou.cfg backup/your_weight_path/your_weight.weights
```
There will be 20 txt files generated in results/.

Then for validation, I mainly use three files: compute_mAP.py, voc_eval.py, map.py

You can put the three in the same directory.

After that I use compute_mAP.py to compute mAP. This will create 10 txt files, each of them contains mAP for 20 classes. You can open voc_eval.py to change the path. Finally, run map.py. This will print the AP in the terminal and calculate the mAP for different threshhold, e.g, AP50, AP75.
AP50, AP55, ..., AP95 will appear at the last line of 10 txt files generated above.

### COCO

Evaluate all weights files in the given `weights_folder` with both the IoU and GIoU metrics using the following script:

    python scripts/coco_all_map.py --data_file cfg/coco-giou-12.data --cfg_file cfg/yolov3.coco-giou-12.cfg --weights_folder backup/coco-giou-12 --lib_folder lib --gpu_id 0 --min_weight_id 20000

See the [scripts/crontab.tmpl]() file for details

Evaluate a specific weights file:

    mkdir -p results/coco-giou-12 && ./darknet detector valid cfg/runs/coco-giou-12/coco-giou-12.data cfg/runs/coco-giou-12/yolov3.coco-giou-12.cfg backup/coco-giou-12/yolov3_final.weights -i 0 -prefix results/coco-giou-12

The detector results are written to `coco_results.json` in the prefix specified above

Now edit `scripts/coco_eval.py` to load the this resulting json file and run the evaluation script:


## TODOs

The described setup requires a shared file system when training and testing across multiple machines. In the absence of this, it would be useful to have some logging service to aggregate logs over a network protocol vs requiring a write to shared disk.

## Acknowledgments

Thank you to the Darknet community for help getting started on this code. Specifically, thanks to [AlexeyAB](https://github.com/AlexeyAB/) for his fork of [Darknet](https://github.com/AlexeyAB/darknet), which has been useful as a reference for understanding the code.

And thank you to the GDarknet for their excellent work. (https://github.com/generalized-iou/g-darknet)


## Original Readme

![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).
