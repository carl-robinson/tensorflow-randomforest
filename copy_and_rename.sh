#!/bin/bash
  programname=$0

function usage {
  echo "This script copies each python file in 'source' directory and all subdirectories"
  echo "to the destination directory, renaming them so their filenames are prefixed with"
  echo "their original path, but with '.' separating the directories instead of '/'"
  echo "usage: $programname source_directory destination_directory    (no trailing slashes)"
  exit 1
}

if [ ${#@} == 0 ]; then
  usage
fi


SOURCE=$1
DEST=$2
cd $SOURCE
#CURR1=`pwd`
#echo "PWD = ${CURR1}"

for f in $(find . -type f -name '*.py')
do
#    echo "${f}"
    BASENAME=$(basename "$f")
    DIR=$(dirname "$f")
#    echo "${DIR}"
#    echo "${DIR}/${BASENAME}"
    CONC="${DIR}/${BASENAME}"
    SLASH="/"
    DOT="."
    DOTTED="${CONC//$SLASH/$DOT}"
#    echo "${DOTTED}"

    LEN=${#DOTTED}
    TRUNC=${DOTTED:2:$LEN}
#    echo "${TRUNC}"
    OUT="${DEST}/${TRUNC}"
#    echo "${OUT}"

    `cp $f $OUT`
done

cd $DEST
#CURR2=`pwd`
#echo "DEST = ${DEST}"
#echo "PWD = ${CURR2}"
find . -name '*.py' -exec pydoc -w {} \;