import sys
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

assert len(sys.argv) == 3, 'Argument mismatch: {iou/giou} {result.json}'
_, metric, pred_f = sys.argv
assert metric in ('iou', 'giou'), metric + ' not a valid eval method'
assert os.path.isfile(pred_f), pred_f + ' not a valid file'

annType = 'bbox'
prefix = 'person_keypoints' if annType == 'keypoints' else 'instances'
print('Running demo for *%s* results.' % (annType))

dataDir = 'data/coco'

dataType = 'val2017'
annFile = '%s/annotations/%s_%s.json' % (dataDir, prefix, dataType)
print("loading gt {}".format(annFile))
cocoGt = COCO(annFile)

print("loading predicted {}".format(pred_f))
cocoDt = cocoGt.loadRes(pred_f)

imgIds = sorted(cocoDt.getImgIds())

cocoEval = COCOeval(cocoGt, cocoDt, annType)
cocoEval.params.imgIds = imgIds

gts = cocoEval.cocoGt.loadAnns(
    cocoEval.cocoGt.getAnnIds(imgIds=imgIds, catIds=cocoEval.params.catIds))
dts = cocoEval.cocoDt.loadAnns(
    cocoEval.cocoDt.getAnnIds(imgIds=imgIds, catIds=cocoEval.params.catIds))

cocoEval.evaluate_bboxreg(metric=metric)
# evaluate_bboxreg modifies some parameters in `cocoEval`.
# Do not run anything else (e.g. `cocoEval.evaluate()`) after this method call.
# Initialize a new COCOeval instance if you want to evaluate AP
