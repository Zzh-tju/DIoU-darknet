#!/bin/bash

# Usage:
#
#   scripts/build_run.sh [src] [dst]
#
# e.g.
#
#   scripts/build_run.sh yolov3-voc-lin-2 yolov3-voc-lin-3
#
#  will copy yolov3-voc-lin-2 to the yolov3-voc-lin-3 config
#
# TODO: some default config if [src] is not specified

set -o nounset  # exit if trying to use an uninitialized var
set -o errexit  # exit if any program fails
set -o pipefail # exit if any program in a pipeline fails, also
set -x          # debug mode

# This file's directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
# cfg/runs Folder
RUNS_PATH="$( cd $DIR/../cfg/runs >/dev/null && pwd )"

SRC=$1
DST=$2

if [ -z "$1" ] && [ -z "$2" ]; then
  echo "Usage:"
  echo "  $0 [src] [dst]"
  printf "Where [src] is the run name in the ${RUNS_PATH}, e.g. one of:\n$(ls $RUNS_PATH)"
  echo "and [dst] is the new run name"
fi

SRC_PATH=$RUNS_PATH/$SRC
DST_PATH=$RUNS_PATH/$DST

find_n_replace_all () {
	find . -type f -exec gsed -i "s/$1/$2/g" {} \;
}

# COPY
cp -r $SRC_PATH $DST_PATH

pushd $DST_PATH
  # SWAP OUT CONFIG NAMES -- NOT NECESSARY, switch file names to just "cfg", "data" and "run.sbatch"
  #mv $SRC.cfg $DST.cfg
  #mv $SRC.data $DST.data
  #mv $SRC.sbatch $DST.sbatch
  find_n_replace_all $SRC $DST
popd

echo "Created '${DST}' from '${DST}' be sure to set the correct number of GPUs in the ${DST_PATH}.cfg file"
