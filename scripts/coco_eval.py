# per
#  https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
#

#import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import re
#import pylab
#pylab.rcParams['figure.figsize'] = (10.0, 8.0)

annType = ['segm','bbox','keypoints']
# bounding boxes
annType = annType[1]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print 'Running demo for *%s* results.'%(annType)

#initialize COCO ground truth api
dataDir='datasets/coco/coco'

# all instance files:
#instances_val2014.json
#instances_train2014.json
#instances_val2017.json
#instances_train2017.json

dataType='minival2014'
#dataType='val2014'
#dataType='train2014'
#dataType='val2017'
#dataType='train2014'
annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
print("loading gt {}".format(annFile))
cocoGt=COCO(annFile)

#initialize COCO detections api
#fake
#resFile='%s/results/%s_%s_fake%s100_results.json'
#resFile = resFile%(dataDir, prefix, dataType, annType)
# real results
#resFile = 'results/coco_results_coco-giou-12/402008/val2017/coco_results.json'
resFile = 'results/coco_results.pretrained-608.2014val5k.json'
#resFile = 'results/coco_results.json'
print("loading predicted {}".format(resFile))
#resFile = 'results/coco_results.json.score_gt_0_8.valid2014.pretty.json'
cocoDt=cocoGt.loadRes(resFile)

#gtIds = set(cocoGt.getImgIds())
#coco_val_5k = set()
##imagepaths='datasets/coco/coco/trainvalno5k.txt'
#imagepaths='datasets/coco/coco/coco_val_5k.txt'
#with open(imagepaths) as f:
#    for line in f.readlines():
#        try:
#            #import pdb; pdb.set_trace(); 1
#            coco_val_5k.add(int(re.search('(\d+)\.jpg', line).group(1)))
#        except AttributeError as e:
#            print(line, ":", e)
#print("difference: {}".format(len(coco_val_5k.difference(gtIds))))

imgIds=sorted(cocoDt.getImgIds())
# comment out for all images
#imgIds=imgIds[0:100]
#imgId = imgIds[np.random.randint(100)]


#import pdb; pdb.set_trace(); 1
cocoEval = COCOeval(cocoGt,cocoDt,annType)
#cocoEval.params.catIds = categoryIds
cocoEval.params.imgIds = imgIds

gts=cocoEval.cocoGt.loadAnns(cocoEval.cocoGt.getAnnIds(imgIds=imgIds, catIds=cocoEval.params.catIds))
dts=cocoEval.cocoDt.loadAnns(cocoEval.cocoDt.getAnnIds(imgIds=imgIds, catIds=cocoEval.params.catIds))

#by_id={}
#for gt in gts:
#    imgId = gt['image_id']
#    if imgId not in by_id:
#        by_id[imgId] = {'gt': [], 'dt': []}
#    by_id[imgId]['gt'].append(gt['category_id'])
#for dt in dts:
#    imgId = dt['image_id']
#    if imgId not in by_id:
#        by_id[imgId] = {'gt': [], 'dt': []}
#    by_id[imgId]['dt'].append(dt['category_id'])
#misses = 0
#total = 0
#for imgId, d in by_id.iteritems():
#    total += len(d['gt'])
#    if len(d['dt']) <= 0 and len(d['gt']) > 0:
#        misses += len(d['gt'])
#    elif len(d['dt']) > 0 and len(d['gt']) > 0:
#        print("imgId: {}, clsIds: (dt){}|(gt){}".format(imgId, d['dt'], d['gt']))
#        misses += len(set(d['gt']).difference(set(d['dt'])))
#        inters = set(d['dt']).intersection(set(d['gt']))
#        if len(inters) > 0:
#            print("imgId: {}, clsId: {}".format(imgId, list(inters)))
#            #cocoGt.showAnns(cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=[imgId], catIds=list(inters))), False)
#            #cocoDt.showAnns(cocoDt.loadAnns(cocoDt.getAnnIds(imgIds=[imgId], catIds=list(inters))), True)
#            #import pdb; pdb.set_trace(); 1
#print("missed {}/{}".format(misses, total))

#for imgId in imgIds:
#    for catId in cocoEval.params.catIds:
#        gt = cocoEval._gts[imgId,catId]
#        dt = cocoEval._dts[imgId,catId]
#        if len(gt) and len(dt):
#            print("Found! gt:{}, dt:{}".format(gt, dt))
#import pdb; pdb.set_trace(); 1
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

