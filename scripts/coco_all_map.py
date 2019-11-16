#!/usr/bin/env python

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import skimage.io as io
import argparse, os, sys, re
import numpy as np
import cPickle
import glob
import time
import subprocess

MAX_DARKNET_RUNS_PER_RUN = 10

annType = 'bbox'

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
    """
    now = int(time.time())
    args = parse_args()
    if not args.weights_folder:
        raise("you must pass a --weights_folder")
    weights_folder_name = filter(None, args.weights_folder.split('/'))[-1]
    output_path = "results/coco_results_{}".format(weights_folder_name)
    print("Output to: '{}'".format(output_path))
    try:
        os.mkdir(output_path)
    except OSError as ose:
        print("warning: {}".format(ose))

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

    druns = 0
    for i, weights_file in enumerate(all_weights_files):
        weight_id = int_or_max(file_to_weight_id(weights_file))
        if weight_id < args.min_weight_id:
            continue
        if ('iou', weight_id) in visited and ('giou', weight_id) in visited:
            if ('val2017-iou', weight_id) in visited and ('val2017-giou', weight_id) in visited:
                continue
        weights_path = os.path.dirname(weights_file)


        weights_output_paths = dict()
        for year in ['', 'val2017']:
            weights_output_paths[year] = os.path.join(output_path, str(weight_id), year)
            resFile = os.path.join(weights_output_paths[year], 'coco_results.json')
            print("weights output to: '{}'".format(resFile))
            try:
                os.mkdir(weights_output_paths[year])
            except OSError as ose:
                print("warning: {}".format(ose))

            if druns > MAX_DARKNET_RUNS_PER_RUN and year == '':
                print("completed {} runs, no more darknet this time around!".format(druns))
                break
            if os.path.isfile(resFile) and os.path.getsize(resFile) > 0:
                print("skipping generation of populated results file '{}'".format(resFile))
            else:
                druns += 1
                ldlib = 'LD_LIBRARY_PATH={}'.format(args.lib_folder) if args.lib_folder else ''
                gpu = '-i {}'.format(args.gpu_id) if args.gpu_id else ''
                date_file_with_year = "{}{}.data".format(args.data_file.split('.data')[0], (".{}".format(year) if len(year) else year))
                cmd = "{} ./darknet detector valid {} {} {} {} -prefix {}".format(ldlib, date_file_with_year, args.cfg_file, weights_file, gpu, weights_output_paths[year])
                print("running '{}'".format(cmd))
                retval = 0
                callerr = False
                try:
                    retval = subprocess.call(cmd, shell=True)
                except OSError as ose:
                    print("OSError: '{}'".format(ose))
                    callerr = True
                print("{} finished with val {}".format(cmd, retval))
                sys.stdout.flush()
                if retval != 0 or callerr:
                    raise Exception("'{}' failed".format(cmd))

                print("darknet run {}, '{}' complete".format(druns, cmd))
                output_dir = os.path.abspath(weights_output_paths[year])

        if len(weights_output_paths.items()) == 0:
            print("no weights_output_paths, breaking")
            break
        dataDir='datasets/coco/coco'
        annFile = '%s/annotations/instances_minival2014.json'%(dataDir)
        print("loading {}".format(annFile))
        annFileVal2017 = '%s/annotations/instances_val2017.json'%(dataDir)
        print("loading {}".format(annFileVal2017))
        cocoGts={'':COCO(annFile),'val2017':COCO(annFileVal2017)}

        #for metric in ['iou', 'giou']:
        for metric in ['iou', 'giou', 'val2017-iou', 'val2017-giou']:
            if (metric, weight_id) in visited:
                continue
            year = ''
            if 'val2017' in metric:
                year = 'val2017'
            one = [metric]
            one.append(weight_id)

            print('Evaluating detections with {}'.format(metric))
            mAP_analysis = [','.join([metric, 'mAP'])]
            mean_map_sum = 0
            mean_map_count = 0
            maps = []

            try:
                to_load = "{}/coco_results.json".format(weights_output_paths[year])
                print("Results json: {}".format(to_load))
                cocoDt = cocoGts[year].loadRes(to_load)
            except ValueError as ve: 
                print("WARNING: {}".format(ve))
                continue
            imgIds = sorted(cocoDt.getImgIds())
            # comment out for all images
            #imgIds = imgIds[0:100]
            #imgId = imgIds[np.random.randint(100)]

            cocoEval = COCOeval(cocoGts[year],cocoDt,annType)
            
            cocoEval.params.imgIds = imgIds
            gts = cocoEval.cocoGt.loadAnns(cocoEval.cocoGt.getAnnIds(imgIds=imgIds, catIds=cocoEval.params.catIds))
            dts = cocoEval.cocoDt.loadAnns(cocoEval.cocoDt.getAnnIds(imgIds=imgIds, catIds=cocoEval.params.catIds))

            cocoEval.evaluate(metric.split('-')[-1])
            cocoEval.accumulate()
            cocoEval.summarize()

            mAP = cocoEval.stats[0]
            mAP_5 = cocoEval.stats[1]
            mAP_75 = cocoEval.stats[2]
            one = [metric, weight_id, mAP, mAP_5, 0, 0, 0, 0, mAP_75, 0, 0, 0, 0]
            mAP_analysis.append(','.join([str(o) for o in one]))
            results_path = os.path.join(weights_path, '{}-{}.txt'.format(weight_id, metric))
            print("Writing: '{}' and '{}'".format(results_path, map_results_path))
            with open(results_path, 'w') as f:
                f.write('\n'.join(mAP_analysis))

            # read/write for insertion sort
            # per-line format is ['giou|iou', weight file id, mean, 0.5..0.95 iou]
            reslines = []
            inserted = False
            linetoinsert = ','.join([str(o) for o in one])
            print("inserting: {}".format(linetoinsert))
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

