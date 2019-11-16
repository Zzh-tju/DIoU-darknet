#!/usr/bin/env python

import argparse, os, sys, re
import numpy as np
import cPickle
import glob
import time

from voc_eval import voc_eval
from voc_reval import do_python_eval

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate results over mAP range')

    # required
    parser.add_argument('--weights_folder', dest='weights_folder', type=str)
    parser.add_argument('--data_file', dest='data_file', type=str)
    parser.add_argument('--cfg_file', dest='cfg_file', type=str)

    parser.add_argument('--lib_folder', dest='lib_folder', default='', type=str)
    parser.add_argument('--gpu_id', dest='gpu_id', default='', type=str)
    parser.add_argument('--voc_dir', dest='voc_dir', default='datasets/voc/VOCdevkit', type=str)
    parser.add_argument('--year', dest='year', default='2007', type=str)
    parser.add_argument('--image_set', dest='image_set', default='test', type=str)
    parser.add_argument('--classes', dest='class_file', default='data/voc.names', type=str)
    parser.add_argument('--min_weight_id', dest='min_weight_id', default=700, type=int)
    # does both metrics now
    #parser.add_argument('--metric', dest='metric', default='iou', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def file_to_weight_id(path):
    wid = None
    try:
        m = re.search('(\d+).weights', path)
        if m is None:
            m = re.search('(final)\.weights', path)
        wid = m.group(1)
    except AttributeError as ae:
        print("{}: {}".format(path, ae))
        raise ae
    return wid

def int_or_max(int_or_final):
    if int_or_final == 'final':
        return sys.maxint
    return int(int_or_final)

def main():
    """
    python scripts/voc_all_map.py --weights_folder [folder with weights] --data_file [whatever.data] --cfg_file [whatever.cfg]

    long version:
    python scripts/voc_map.py --year 2007 --classes data/voc.names --image_set test --voc_dir datasets/voc/VOCdevkit --output_dir results --weights_folder [folder with weights]
    """
    now = int(time.time())
    output_path = "results/{}".format(now)
    print("Output to: '{}'".format(output_path))
    os.mkdir(output_path)
    args = parse_args()
    if not args.weights_folder:
        raise("you must pass a --weights_folder")

    all_weights_files = glob.glob(os.path.join(args.weights_folder, '*.weights'))
    #reverse sorted
    all_weights_files = sorted(all_weights_files, lambda a,b: -1 if int_or_max(file_to_weight_id(a)) > int_or_max(file_to_weight_id(b)) else 1)
    print("Processing {}".format('\n'.join(all_weights_files)))

    visited = set()
    map_results_path = os.path.join(args.weights_folder, 'map.txt')
    # skip already visited
    if os.path.isfile(map_results_path):
        with open(map_results_path, 'r') as f:
            for line in f.readlines():
                if len(line.strip()) < 1:
                    continue
                rows = line.split(',')
                # mteric, weight_id
                visited.add((rows[0], int_or_max(rows[1])))
    print("Skipping already visited {}".format(visited))

    for weights_file in all_weights_files:
        weight_id = int_or_max(file_to_weight_id(weights_file))
        if weight_id < args.min_weight_id:
            continue
        if ('iou', weight_id) in visited and ('giou', weight_id) in visited:
            continue
        weights_path = os.path.dirname(weights_file)

        # don't rebuild, inference is always set to the correct batch size now
        #if os.WEXITSTATUS(os.system("make")) != 0:
        #    assert "make failed"
        ldlib = 'LD_LIBRARY_PATH={}'.format(args.lib_folder) if args.lib_folder else ''
        gpu = '-i {}'.format(args.gpu_id) if args.gpu_id else ''
        cmd = "{} ./darknet detector valid {} {} {} {} -prefix {}".format(ldlib, args.data_file, args.cfg_file, weights_file, gpu, output_path)
        print(cmd)
        if os.WEXITSTATUS(os.system(cmd)) != 0:
            assert "{} failed".format(cmd)

        print("./darknet detector complete")
        output_dir = os.path.abspath(output_path)
        with open(args.class_file, 'r') as f:
            lines = f.readlines()
        classes = [t.strip('\n') for t in lines]

        for metric in ['iou', 'giou']:
            if (metric, weight_id) in visited:
                continue
            use_giou = (metric == 'giou')
            one = [metric]
            one.append(weight_id)

            print('Evaluating detections with {}'.format('gIOU' if use_giou else 'IoU'))
            mAP_analysis = [','.join(['{}IOU'.format('g' if use_giou else ''), 'mAP'])]
            mean_map_sum = 0
            mean_map_count = 0
            maps = []
            for i in range(50, 100, 5):
                iou_threshold = i / float(100)
                mAP = do_python_eval(args.voc_dir, args.year, args.image_set, classes, output_dir, iou_threshold, use_giou)
                mean_map_sum += mAP
                mean_map_count += 1
                maps.append(mAP)
                mAP_analysis.append(','.join([str(iou_threshold), str(mAP)]))
            mean_map = mean_map_sum/mean_map_count
            one.append(mean_map)
            one.extend(maps)
            results_path = os.path.join(weights_path, '{}-{}.txt'.format(weight_id, metric))
            print("Writing: '{}' and '{}'".format(results_path, map_results_path))
            with open(results_path, 'w') as f:
                f.write('\n'.join(mAP_analysis))
            # read/write for insertion sort
            # per-line format is ['giou|iou', weight file id, mean, 0.5..0.95 iou]
            reslines = []
            inserted = False
            linetoinsert = ','.join([str(o) for o in one])
            print(linetoinsert)
            if os.path.isfile(map_results_path):
                with open(map_results_path, 'r') as f:
                    for line in f.readlines():
                        if len(line.strip()) < 1:
                            continue
                        cols = line.split(',')
                        if (not inserted) and int(cols[1]) > weight_id:
                            reslines.append(linetoinsert)
                            inserted = True
                        reslines.append(line)
            else:
                reslines.append(linetoinsert)
                inserted = True
            if not inserted:
                reslines.append(linetoinsert)
            with open(map_results_path, 'w') as f:
                f.write('\n'.join([l.strip() for l in reslines]))

if __name__ == '__main__':
    main()
