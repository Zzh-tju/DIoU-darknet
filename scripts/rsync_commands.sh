# copy out latest results
rsync --update -raz --progress /raid/scratch/ntsoi/darknet/backup/coco-giou-4 ~/src/nn/darknet/backup/

# copy in latest project dir
rsync --update -raz --progress $HOME/src/nn/darknet/* /raid/scratch/ntsoi/darknet/ --exclude datasets --exclude backup --exclude batch --exclude darkboard --exclude results

rsync --update -raz --progress /raid/scratch/ntsoi/darknet/backup/coco-giou-4 ~/src/nn/darknet/backup/

rsync --update -raz --progress ~/.pyenv /cvgl2/u/ntsoi/

rsync --update -raz --progress * /scr/ntsoi/mot/* --exclude datasets
