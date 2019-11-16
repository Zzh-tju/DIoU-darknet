#!/bin/bash

# MUST BE RUN FROM AFS

ROOTDIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )/../")"
PROJECT=$(basename "$ROOTDIR")
BACKUP=$1

if test -z "$BACKUP" 
then
  echo "Pass name where backup/[name]/ as first param"
  exit -1
else
  echo "copying $BACKUP"
  rsync --update -raz --progress "/scr/ntsoi/$PROJECT/backup/$BACKUP" "$ROOTDIR/backup/"
fi

