from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
import matplotlib.patheffects as PathEffects
import matplotlib.image as mpimg
import skimage.io as io
import argparse, os, sys, re
import numpy as np
import cPickle
import glob
import time
import subprocess

import iou_utils

DATASETS='/scr/ntsoi/darknet/datasets'
# gt is pink
COLORS= dict(
    mse=[1,1,1],
    iou=[0,0,1],
    giou=[0,1,0],
    white=[1,1,1],
    black=[0,0,0],
)

def show_annotated(imgid, anns_by_loss, gt, img_output_path):
    #import pdb; pdb.set_trace(); 1
    common_categories = set()
    for loss, anns in anns_by_loss.iteritems():
        #c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
        gtcolor = [1.0,.25,.5]
        #import pdb; pdb.set_trace(); 1
        categories = [a['category_id'] for a in anns['dt'] if a['score'] > 0.8]
        common_categories.update(categories)

    common_categories = list(common_categories)
    if len(common_categories) < 1:
        return

    for loss, anns in anns_by_loss.iteritems():
        ax = plt.gca()
        ax.set_autoscale_on(False)
        dtcolor = COLORS[loss] 
        for category_id in common_categories:
            gt_polygons = []
            dt_polygons = []
            color = []
            gta = [a for a in anns['gt'] if a['category_id'] == category_id]
            if len(gta) < 1:
                continue
            gtbbox = gta[0]['bbox']
            #x,y,w,h
            gtpoly = Rectangle((gtbbox[0], gtbbox[1]), gtbbox[2], gtbbox[3])
            gt_polygons.append(gtpoly)
            color.append(COLORS['white'])

            dt = [a for a in anns['dt'] if a['category_id'] == category_id]
            if len(dt) < 1:
                continue
            dt = sorted(dt, key=lambda a: a['score']) 
            dtbbox = dt[0]['bbox']
            #x,y,w,h
            dtpoly = Rectangle((dtbbox[0], dtbbox[1]), dtbbox[2], dtbbox[3])
            dt_polygons.append(dtpoly)
            color.append(COLORS['white'])

            # top, left, bottom, right
            gtbb = gtbbox[1], gtbbox[0], gtbbox[1]+gtbbox[3], gtbbox[0]+gtbbox[2]
            dtbb = dtbbox[1], dtbbox[0], dtbbox[1]+dtbbox[3], dtbbox[0]+dtbbox[2]

            iou = iou_utils.iou(gtbb, dtbb)
            giou = iou_utils.giou(gtbb, dtbb)

            txt = ax.text(0.5, 0.1, "IoU: %.3f, GIoU: %.3f" % (iou, giou), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color=COLORS['white'], family='serif', size=26)
            txt.set_path_effects([PathEffects.withStroke(linewidth=4, foreground='black')])
            
            #p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
            #ax.add_collection(p)
            # gt
            p = PatchCollection(gt_polygons, facecolor='none', edgecolors=color, linewidths=4)
            p.set_linestyle('solid')
            ax.add_collection(p)

            # dt
            p = PatchCollection(dt_polygons, facecolor='none', edgecolors=color, linewidths=4)
            p.set_linestyle('dashed')
            ax.add_collection(p)

            #1 category only
            break

        #import pdb; pdb.set_trace(); 1
        imgfname = gt.imgs[imgid]['file_name']
        _, split, _ = imgfname.split('_')
        src_img = "{}/coco/coco/images/{}/{}".format(DATASETS, split, imgfname)
        img = mpimg.imread(src_img)
        #ax.add_image(img)
        #ax.figure.figimage(img)
        ax.autoscale()
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        #import pdb; pdb.set_trace(); 1
        imgpath = "{}/{}_category-{}-{}.jpg".format(img_output_path, imgid, common_categories[0], loss)
        print("imgpath: {}".format(imgpath))
        ax.figure.savefig(imgpath,dpi=400,bbox_inches='tight',pad_inches=0)
        plt.clf()

parser = argparse.ArgumentParser(description='Evaluate results over mAP range')
parser.add_argument('--lib_folder', dest='lib_folder', default='', type=str)
parser.add_argument('--gpu_id', dest='gpu_id', default='', type=str)
args = parser.parse_args()

configs = dict(
    mse = dict(
        data='cfg/coco.coco-baseline4.data',
        cfg='cfg/yolov3.coco-baseline4.cfg',
        iteration=202124,
        weights='backup/coco-baseline4/yolov3_%d.weights',
    ),
    iou = dict(
        data='cfg/coco-iou-14.data',
        cfg='cfg/yolov3.coco-iou-14.cfg',
        iteration=202000,
        weights='backup/coco-iou-14/yolov3_%d.weights',
    ),
    giou = dict(
        data='cfg/coco-giou-12.data',
        cfg='cfg/yolov3.coco-giou-12.cfg',
        iteration=203000,
        weights='backup/coco-giou-12/yolov3_%d.weights',
    )
)

dataDir='datasets/coco/coco'
prefix = 'instances'
dataType='minival2014'
annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
cocoGt=COCO(annFile)

coco_val_5k = set()
imagepaths='datasets/coco/coco/coco_val_5k.txt'
with open(imagepaths) as f:
    for line in f.readlines():
        try:
            coco_val_5k.add(int(re.search('(\d+)\.jpg', line).group(1)))
        except AttributeError as e:
            print(line, ":", e)

coco_val_5k_subset = list(coco_val_5k)[0:100]

output_prefix = 'results/qualitative'
image_output_path = os.path.join(output_prefix, 'images')
try:
    os.makedirs(image_output_path)
except OSError as ose:
    print("warning: {}".format(ose))
for loss, cfgs in configs.iteritems():
    output_path = os.path.join(output_prefix, loss, str(cfgs['iteration']))
    resFile = "{}/coco_results.json".format(output_path)
    if os.path.isfile(resFile) and os.path.getsize(resFile) > 0:
        print("skipping generation of populated results file '{}'".format(resFile))
    else:
        try:
            os.makedirs(output_path)
        except OSError as ose:
            print("warning: {}".format(ose))
        configs[loss]['output_path'] = output_path
        ldlib = 'LD_LIBRARY_PATH={}'.format(args.lib_folder) if args.lib_folder else ''
        gpu = '-i {}'.format(args.gpu_id) if args.gpu_id else ''
        cmd = "{} ./darknet detector valid {} {} {} {} -prefix {}".format(ldlib, cfgs['data'], cfgs['cfg'], cfgs['weights']%cfgs['iteration'], gpu, output_path)
        print(cmd)
        retval = 0
        callerr = False
        try:
            retval = subprocess.call(cmd, shell=True)
        except OSError as ose:
            print(ose)
            callerr = True
        print("{} finished with val {}".format(cmd, retval))
        if retval != 0 or callerr:
            raise Exception("'{}' failed".format(cmd))

    # load results
    print("loading predicted {}".format(resFile))
    cocoDt=cocoGt.loadRes(resFile)

    # load annotations
    annotations = {}
    for imgid in coco_val_5k_subset:
        annotations[imgid] = dict(
            gt = cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=[imgid])),
            dt = cocoDt.loadAnns(cocoDt.getAnnIds(imgIds=[imgid]))
        )
    configs[loss]['annotations'] = annotations

for imgid in coco_val_5k_subset:
    anns = {}
    for loss, cfgs in configs.iteritems():
        anns[loss] = configs[loss]['annotations'][imgid]
    show_annotated(imgid, anns, cocoGt, image_output_path)
