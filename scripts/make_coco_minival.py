import json, os

dataDir='datasets/coco/coco'
prefix = 'instances'

dataType='val2014'
annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)

outDataList='coco_val_5k'
outImageFile = '%s/%s.txt'%(dataDir,outDataList)
outDataType='minival2014'
outAnnFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,outDataType)

image_names = set()
with open(outImageFile, 'rb') as images:
    for image in images.readlines():
        image_names.add(os.path.basename(image).strip())

print("looking for {} images".format(len(image_names)))

outd = None
annotation_images = {}
with open(annFile, 'rb') as injson:
    d = json.load(injson)
    outd = d
    for image in d['images']:
        annotation_images[image['file_name']] = image

outd['images'] = []
for image_name in list(image_names):
    outd['images'].append(annotation_images[image_name])

print("found {} of {}".format(len(outd['images']), len(image_names)))

with open(outAnnFile, 'w') as outjson:
    outjson.write(json.dumps(outd))

print("wrote {}".format(outAnnFile))
