from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
import matplotlib.patheffects as PathEffects
import matplotlib.image as mpimg
from matplotlib import colors as mcolors
import skimage.io as io
import argparse, os, sys, re
import numpy as np
import cPickle
import glob
import time
import subprocess

import iou_utils

MAX_IMAGES=500
SCORE_THRESHOLD = 0.25
#PLOT_MAX_CATEGORIES = 2
DATASETS='/scr/ntsoi/darknet/datasets'
# gt is pink
COLORS= dict(
    mse=[1,1,1],
    iou=[0,0,1],
    giou=[0,1,0],
    white=[1,1,1],
    black=[0,0,0],
)
color_names = [c[0] for c in mcolors.TABLEAU_COLORS.items()]

colormod = 0

def show_annotated(imgid, anns_by_loss, gt, img_output_path, txt_output_path, network='yolo'):
    global colormod
    #import pdb; pdb.set_trace(); 1
    common_categories = set()
    for loss, anns in anns_by_loss.iteritems():
        #c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
        gtcolor = [1.0,.25,.5]
        #import pdb; pdb.set_trace(); 1
        categories = [a['category_id'] for a in anns['dt']]
        common_categories.update(categories)

    common_categories = list(common_categories)
    if len(common_categories) < 1:
        return

    avggious = dict(mse=False,iou=False,giou=False)
    avgious = dict(mse=False,iou=False,giou=False)
    biggestboxscores = {}
    for loss, anns in anns_by_loss.iteritems():
        avggious[loss] = []
        avgious[loss] = []
        ax = plt.gca()
        ax.set_autoscale_on(False)
        gcolor = []
        dcolor = []
        dtcolor = COLORS[loss] 
        bbox_txt = []

        #for cidx, category_id in enumerate(common_categories):
        gt_polygons = []
        dt_polygons = []
        #gta = [a for a in anns['gt'] if a['category_id'] == category_id]
        gta = anns['gt']
        if len(gta) < 1:
            continue
        gtbboxes = [b['bbox'] for b in gta]
        #dt = [a for a in anns['dt'] if a['category_id'] == category_id]
        dt = anns['dt']
        dtbboxes = [b['bbox'] for b in dt]
        # gious shape(dt, gt)
        gious = np.zeros((len(gtbboxes), len(dtbboxes)))
        for i, g in enumerate(gtbboxes):
            for j, d in enumerate(dtbboxes):
                # top, left, bottom, right
                gtbb = g[1], g[0], g[1]+g[3], g[0]+g[2]
                dtbb = d[1], d[0], d[1]+d[3], d[0]+d[2]
                gious[i,j] = iou_utils.giou(gtbb, dtbb)

        #sortby = np.diagonal(gious)
        #gtbboxes_s = [x for _, x in sorted(zip(sortby,gtbboxes), key=lambda pair: pair[0])]
        #dtbboxes_s = [x for _, x in sorted(zip(sortby,dtbboxes), key=lambda pair: pair[0])]
        
        for i, gtbbox in enumerate(gtbboxes):
            gann = anns['gt'][i]
            category_id = gann['category_id']
            print("category_id: {}".format(category_id))
            # find gt
            dann = anns['dt'][np.argmax(gious[i])]
            dtbbox = dtbboxes[np.argmax(gious[i])]
            #x,y,w,h
            #colormod += 1
            c = color_names[category_id%len(color_names)]

            det_c = c
            # white detection bb if score < threshold
            #det_c = c if dann['score'] >= SCORE_THRESHOLD else 'white'

            gcolor.append(c)
            dcolor.append(det_c)
            gtbb = gtbbox[1], gtbbox[0], gtbbox[1]+gtbbox[3], gtbbox[0]+gtbbox[2]
            gtbbsize = (gtbb[2] - gtbb[0]) * (gtbb[3] - gtbb[1])
            dtbb = dtbbox[1], dtbbox[0], dtbbox[1]+dtbbox[3], dtbbox[0]+dtbbox[2]
            giou = iou_utils.giou(gtbb, dtbb)
            iou = iou_utils.iou(gtbb, dtbb)
            avggious[loss].append(giou)
            avgious[loss].append(iou)
            #if gious.shape[0] > 1:
            #    import pdb; pdb.set_trace(); 1
            bbox_txt.append("BB: {}, category_id: {}, color: {}, giou: {}, iou: {}".format(i, category_id, c, giou, iou))
            #iou = iou_utils.iou(gtbb, dtbb)

            linewidth=5
            category_name = [cat['name'] for cat in gt.dataset['categories'] if cat['id'] == category_id][0]
            score = dann['score']
            if loss not in biggestboxscores:
                biggestboxscores[loss] = None
            if biggestboxscores[loss] is None or biggestboxscores[loss]['box'] < gtbbsize:
                biggestboxscores[loss] = {'box': gtbbsize, 'score': score, 'category': category_name, 'metric': giou}
            print("TEXT COORDINATES: ({}, {}), class: {}, score: {}".format(gtbbox[0], dtbbox[1], category_name, score))
            #ax.text(dtbbox[0], dtbbox[1], category_name, horizontalalignment='left', verticalalignment='bottom', color=c, family='serif', size=10)
            #if dann['score'] <= SCORE_THRESHOLD:
            #    ax.text(dtbbox[2], dtbbox[1], dann['score'], horizontalalignment='left', verticalalignment='bottom', color='white', family='serif', size=10)

            gtpoly = Rectangle((gtbbox[0], gtbbox[1]), gtbbox[2], gtbbox[3])
            gt_polygons.append(gtpoly)
            p = PatchCollection(gt_polygons, facecolor='none', edgecolors='white', linewidths=linewidth, alpha=0.5, zorder=i)
            p.set_linestyle('solid')
            ax.add_collection(p)

            dtpoly = Rectangle((dtbbox[0], dtbbox[1]), dtbbox[2], dtbbox[3])
            dt_polygons.append(dtpoly)
            p = PatchCollection(dt_polygons, facecolor='none', edgecolors=dcolor, linewidths=linewidth, alpha=score, zorder=len(gtbboxes)+i)
            p.set_linestyle('dashed')
            ax.add_collection(p)
            #p = PatchCollection(dt_polygons, facecolor='none', edgecolors=dcolor, linewidths=linewidth, alpha=score)
            #p.set_linestyle('dashed')
            #ax.add_collection(p)


            #txt = ax.text(0.5, 0.1, "IoU: %.3f, GIoU: %.3f" % (iou, giou), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color=COLORS['white'], family='serif', size=26)
            #txt.set_path_effects([PathEffects.withStroke(linewidth=4, foreground='black')])
            
            #p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
            #ax.add_collection(p)
            # gt

            # dt

            #1 category only
            #if cidx >= PLOT_MAX_CATEGORIES:
            #    break

        if avggious['giou'] != False:
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
            # range 0 <-> 2
            mmse = 1+np.mean(avggious['mse'])
            mgiou = 1+np.mean(avggious['giou'])
            miou = 1+np.mean(avggious['iou'])
            scorestr = "%s%s%s"%(str(int(mmse * 100)).zfill(3), str(int(miou * 100)).zfill(3), str(int(mgiou * 100)).zfill(3))
            basename= "{}-id_{}-mse_{}-iou_{}-giou_{}-category-{}-{}-{}".format(scorestr, imgid, mmse, miou, mgiou, common_categories[0], network, loss)
            imgpath = "{}/{}.jpg".format(img_output_path, basename)
            txtpath = "{}/{}.txt".format(txt_output_path, basename)
            bbox_txt.insert(0, "AVG: giou metric: (giou loss: {}, iou loss: {}, mse loss: {}), iou metric: (giou loss: {}, iou loss: {}, mse loss: {})".format(
                np.mean(avggious['giou']),
                np.mean(avggious['iou']),
                np.mean(avggious['mse']),
                np.mean(avgious['giou']),
                np.mean(avgious['iou']),
                np.mean(avgious['mse'])
            ))
            with open(txtpath, 'wb') as f:
                f.write('\n'.join(bbox_txt))
            print("imgpath: {}".format(imgpath))
            ax.figure.savefig(imgpath,dpi=100,bbox_inches='tight',pad_inches=0)
            plt.clf()

    if 'giou' in biggestboxscores and 'iou' in biggestboxscores and 'mse' in biggestboxscores:
        with open(os.path.join(txt_output_path, "scores.csv"), 'ab') as f:
            f.write("{},{},{},{},{},{},{},{},{},{}\n".format(imgid,
                biggestboxscores['giou']['score'],
                biggestboxscores['iou']['score'],
                biggestboxscores['mse']['score'],
                biggestboxscores['giou']['category'],
                biggestboxscores['iou']['category'],
                biggestboxscores['mse']['category'],
                biggestboxscores['giou']['metric'],
                biggestboxscores['iou']['metric'],
                biggestboxscores['mse']['metric']
            ))


parser = argparse.ArgumentParser(description='Evaluate results over mAP range')
parser.add_argument('--lib_folder', dest='lib_folder', default='', type=str)
parser.add_argument('--gpu_id', dest='gpu_id', default='', type=str)
args = parser.parse_args()

configs = dict(
    mse = dict(
        data='cfg/coco.coco-baseline4.data',
        cfg='cfg/yolov3.coco-baseline4.cfg',
        iteration=492000,
        weights='backup/coco-baseline4/yolov3_%d.weights',
    ),
    iou = dict(
        data='cfg/coco-iou-14.data',
        cfg='cfg/yolov3.coco-iou-14.cfg',
        iteration=470000,
        weights='backup/coco-iou-14/yolov3_%d.weights',
    ),
    giou = dict(
        data='cfg/coco-giou-12.data',
        cfg='cfg/yolov3.coco-giou-12.cfg',
        iteration='final',
        weights='backup/coco-giou-12/yolov3_%s.weights',
    )
)

maskrcnnDataDir='results/maskrcnn/coco2014_mask_80k_%s'

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

coco_val_5k_subset = list(coco_val_5k)[0:MAX_IMAGES]

output_prefix = 'results/qualitative'
image_output_path = os.path.join(output_prefix, 'images')
txt_output_path = os.path.join(output_prefix, 'annotations')

maskrcnn_annotations_by_loss = {}
for path in [txt_output_path, image_output_path]:
    try:
        os.makedirs(path)
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
    # load annotations for yolo
    yolo_cocoDt=cocoGt.loadRes(resFile)
    yolo_annotations = {}
    for imgid in coco_val_5k_subset:
        yolo_annotations[imgid] = dict(
            gt = cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=[imgid])),
            dt = [a for a in yolo_cocoDt.loadAnns(yolo_cocoDt.getAnnIds(imgIds=[imgid]))]# if a['score'] > SCORE_THRESHOLD]
        )
    configs[loss]['annotations'] = yolo_annotations

    # load annotations for maskrcnn
    maskrcnn_cocoDt=cocoGt.loadRes("{}/bbox_coco_2014_val_results.json".format(maskrcnnDataDir%(loss if loss is not 'mse' else 'sl1')))
    maskrcnn_annotations = {}
    for imgid in coco_val_5k_subset:
        maskrcnn_annotations[imgid] = dict(
            gt = cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=[imgid])),
            dt = [a for a in maskrcnn_cocoDt.loadAnns(maskrcnn_cocoDt.getAnnIds(imgIds=[imgid]))]# if a['score'] > SCORE_THRESHOLD]
        )
    maskrcnn_annotations_by_loss[loss] = maskrcnn_annotations

for imgid in coco_val_5k_subset:
    yolo_anns = {}
    maskrcnn_anns = {}
    for loss, cfgs in configs.iteritems():
        yolo_anns[loss] = configs[loss]['annotations'][imgid]
        maskrcnn_anns[loss] = maskrcnn_annotations_by_loss[loss][imgid]
    show_annotated(imgid, yolo_anns, cocoGt, image_output_path, txt_output_path, 'yolo')
    show_annotated(imgid, maskrcnn_anns, cocoGt, image_output_path, txt_output_path, 'maskrcnn')
