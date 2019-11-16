#!/bin/bash

# MUST BE RUN FROM AFS

ROOTDIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )/../")"
PROJECT=$(basename "$ROOTDIR")
BACKUP=$1

if test -z "$BACKUP" 
then
  echo "Pass name where backup/[name]/ as first param, to limit copy. Copying ALL (except tmp/batch/out/results)!"
  rsync --update -raz --progress "$ROOTDIR" "/scr/ntsoi/" --exclude darkboard/tmp --exclude batch/out --exclude results --exclude darkboard/db
else
  echo "copying $BACKUP"
  rsync --update -raz --progress "$ROOTDIR" "/scr/ntsoi/" --exclude datasets --exclude backup --exclude batch --exclude darkboard --exclude results
  mkdir -p "/scr/ntsoi/$PROJECT/backup"
  rsync --update -raz --progress "$ROOTDIR/backup/$BACKUP" "/scr/ntsoi/$PROJECT/backup/"
fi

