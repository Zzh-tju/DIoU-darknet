%% Demo demonstrating the algorithm result formats for COCO

%cd 'scripts';
addpath('cocoapi/MatlabAPI');
mex('CFLAGS=\$CFLAGS -Wall -std=c99','-largeArrayDims','cocoapi/MatlabAPI/private/maskApiMex.c','cocoapi/common/maskApi.c','-Icocoapi/common/','-outdir','cocoapi/MatlabAPI/private');

%% select results type for demo (either bbox or segm)
type = {'segm','bbox','keypoints'}; type = type{2}; % specify type here
dataType='val2014';
fprintf('Running demo for *%s* results.\n\n',type);

%% initialize COCO ground truth api
dataDir='../datasets/coco/coco'; prefix='instances';
%if(strcmp(type,'keypoints')), prefix='person_keypoints'; end
annFile=sprintf('%s/annotations/%s_%s.json',dataDir,prefix,dataType);
cocoGt=CocoApi(annFile);

%% initialize COCO detections api
%resFile='%s/results/%s_%s_fake%s100_results.json';
%resFile=sprintf(resFile,dataDir,prefix,dataType,type);
resFile='../results/coco_results.json';
cocoDt=cocoGt.loadRes(resFile);

%% visialuze gt and dt side by side
imgIds=sort(cocoGt.getImgIds()); imgIds=imgIds(1:100);
imgId = imgIds(randi(100)); img = cocoGt.loadImgs(imgId);
I = imread(sprintf('%s/images/val2014/%s',dataDir,img.file_name));
f1 = figure(1);
subplot(1,2,1); imagesc(I); axis('image'); axis off;
annIds = cocoGt.getAnnIds('imgIds',imgId); title('ground truth')
anns = cocoGt.loadAnns(annIds); cocoGt.showAnns(anns);
subplot(1,2,2); imagesc(I); axis('image'); axis off;
annIds = cocoDt.getAnnIds('imgIds',imgId); title('results')
anns = cocoDt.loadAnns(annIds); cocoDt.showAnns(anns);

saveas(f1, sprintf('analyze/%s.png',img.file_name));

%% load raw JSON and show exact format for results
fprintf('results structure have the following format:\n');
res = gason(fileread(resFile)); disp(res)

%% the following command can be used to save the results back to disk
if(0), f=fopen(resFile,'w'); fwrite(f,gason(res)); fclose(f); end

%% run COCO evaluation code (see CocoEval.m)
cocoEval=CocoEval(cocoGt,cocoDt,type);
cocoEval.params.imgIds=imgIds;
cocoEval.evaluate();
cocoEval.accumulate();
cocoEval.summarize();

%% generate Derek Hoiem style analyis of false positives (slow)
if(0), cocoEval.analyze(); end
