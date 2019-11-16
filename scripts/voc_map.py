#!/usr/bin/env python

import os

import os, sys, argparse
import numpy as np
import cPickle

from voc_eval import voc_eval
from voc_reval import do_python_eval

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate results over mAP range')
    parser.add_argument('--output_dir', dest='output_dir', default='results', type=str)
    parser.add_argument('--weights_path', dest='weights_path', type=str)
    parser.add_argument('--voc_dir', dest='voc_dir', default='datasets/voc/VOCdevkit', type=str)
    parser.add_argument('--year', dest='year', default='2007', type=str)
    parser.add_argument('--image_set', dest='image_set', default='test', type=str)
    parser.add_argument('--skip_detector', dest='skip_detector', action='store_true')
    parser.set_defaults(skip_detector=False)
    parser.add_argument('--classes', dest='class_file', default='data/voc.names', type=str)
    parser.add_argument('--metric', dest='metric', default='iou', type=str)
    parser.add_argument('--cfg_file', dest='cfg_file', default='cfg/yolov3-voc.cfg', type=str)
    parser.add_argument('--data_file', dest='data_file', default='cfg/voc.data', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def main():
    """
    python scripts/voc_map.py --weights_path [path to weights]

    long version:
    python scripts/voc_map.py --year 2007 --classes data/voc.names --image_set test --voc_dir datasets/voc/VOCdevkit --output_dir results --weights_path [path to weights]
    """
    args = parse_args()
    if not args.weights_path:
        raise("you must pass a --weights_path")
    if not args.skip_detector:
        os.system("make")
        os.system("./darknet detector valid {} {} {}".format(args.data_file, args.cfg_file, args.weights_path))

    output_dir = os.path.abspath(args.output_dir)
    with open(args.class_file, 'r') as f:
        lines = f.readlines()

    classes = [t.strip('\n') for t in lines]

    use_giou = (args.metric.lower() == 'giou')
    print('Evaluating detections with {}'.format('gIOU' if use_giou else 'IoU'))
    mAP_analysis = [','.join(['{}IOU'.format('g' if use_giou else ''), 'mAP'])]
    for i in range(50, 100, 5):
        iou_threshold = i / float(100)
        mAP = do_python_eval(args.voc_dir, args.year, args.image_set, classes, output_dir, iou_threshold, use_giou)
        mAP_analysis.append(','.join([str(iou_threshold), str(mAP)]))
    print('\n'.join(mAP_analysis))

if __name__ == '__main__':
    main()
