#!/bin/bash

gifname=${1:-training.gif} #if full path is not provided, then save in $indir
indir=${2:-plots/vf_seq}
delay=${3:-2} #less = faster
nfiles=${4:-0} #0: defined from directory
batch=10 # number of images done at once. To avoid memory overflow
echo Delay: $delay
cd "$indir"
files=(*png)
mkdir -p "$(dirname "$gifname")"

## Compute number of files to process
if [[ $nfiles -eq 0 ]]; then
  # Initialize from directory
  nfiles=${#files[@]}
else
  nfiles=$(($nfiles < ${#files[@]} ? $nfiles:${#files[@]}))
fi
echo "Processing $nfiles files from: $indir"

## Read the array in batches of $batch
for (( i=0; $i<$nfiles; i+=$batch )); do
  ## Convert this batch
  #echo ${files[@]:$i:$batch}
  convert -delay $delay -loop 0 "${files[@]:$i:$batch}" animated.$(printf %05d $i).gif
done

# Add pause on the last
convert -delay 500 -loop 0 "${files[$((nfiles - 1))]}" animated.$(printf %05d $((i+1))).gif

## Now, merge them into a single file
convert  animated.*.gif "$gifname"
if [[ $? -eq 0 ]]; then
  rm -rf animated.*.gif
fi
