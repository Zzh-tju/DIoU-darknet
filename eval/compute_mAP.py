from voc_eval import voc_eval
for i in range(9):
    thresh = (i*5+50)/100
    print(voc_eval('/home/yourpath/DIoU-Darknet/results/{}.txt', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/Annotations/{}.xml', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'aeroplane', '.', thresh))

    print(voc_eval('/home/yourpath/DIoU-Darknet/results/{}.txt', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/Annotations/{}.xml', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'bicycle', '.', thresh))

    print(voc_eval('/home/yourpath/DIoU-Darknet/results/{}.txt', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/Annotations/{}.xml', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'bird', '.', thresh))

    print(voc_eval('/home/yourpath/DIoU-Darknet/results/{}.txt', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/Annotations/{}.xml', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'boat', '.', thresh))

    print(voc_eval('/home/yourpath/DIoU-Darknet/results/{}.txt', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/Annotations/{}.xml', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'bottle', '.', thresh))

    print(voc_eval('/home/yourpath/DIoU-Darknet/results/{}.txt', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/Annotations/{}.xml', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'bus', '.', thresh))

    print(voc_eval('/home/yourpath/DIoU-Darknet/results/{}.txt', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/Annotations/{}.xml', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'car', '.', thresh))

    print(voc_eval('/home/yourpath/DIoU-Darknet/results/{}.txt', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/Annotations/{}.xml', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'cat', '.', thresh))

    print(voc_eval('/home/yourpath/DIoU-Darknet/results/{}.txt', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/Annotations/{}.xml', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'chair', '.', thresh))

    print(voc_eval('/home/yourpath/DIoU-Darknet/results/{}.txt', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/Annotations/{}.xml', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'cow', '.', thresh))

    print(voc_eval('/home/yourpath/DIoU-Darknet/results/{}.txt', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/Annotations/{}.xml', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'diningtable', '.', thresh))

    print(voc_eval('/home/yourpath/DIoU-Darknet/results/{}.txt', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/Annotations/{}.xml', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'dog', '.', thresh))

    print(voc_eval('/home/yourpath/DIoU-Darknet/results/{}.txt', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/Annotations/{}.xml', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'horse', '.', thresh))

    print(voc_eval('/home/yourpath/DIoU-Darknet/results/{}.txt', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/Annotations/{}.xml', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'motorbike', '.', thresh))

    print(voc_eval('/home/yourpath/DIoU-Darknet/results/{}.txt', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/Annotations/{}.xml', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'person', '.', thresh))

    print(voc_eval('/home/yourpath/DIoU-Darknet/results/{}.txt', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/Annotations/{}.xml', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'pottedplant', '.', thresh))

    print(voc_eval('/home/yourpath/DIoU-Darknet/results/{}.txt', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/Annotations/{}.xml', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'sheep', '.', thresh))

    print(voc_eval('/home/yourpath/DIoU-Darknet/results/{}.txt', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/Annotations/{}.xml', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'sofa', '.', thresh))

    print(voc_eval('/home/yourpath/DIoU-Darknet/results/{}.txt', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/Annotations/{}.xml', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'train', '.', thresh))

    print(voc_eval('/home/yourpath/DIoU-Darknet/results/{}.txt', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/Annotations/{}.xml', '/home/yourpath/DIoU-Darknet/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'tvmonitor', '.', thresh))
