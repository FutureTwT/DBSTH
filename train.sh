#!/bin/bash
set -e
METHOD='DBSTH'
DATA='flickr' # 'nuswide' / 'coco'
GPU_ID=0
bits=(16 32 64 128) 

for i in ${bits[*]}; do
  echo "**********Start ${METHOD} algorithm**********"
  CUDA_VISIBLE_DEVICES=$GPU_ID python train.py --nbit $i --dataset $DATA 
  echo "**********End ${METHOD} algorithm**********"

  echo "**********Start ${METHOD} evaluate**********"
  cd matlab
  matlab -nojvm -nodesktop -r "curve($i, '$DATA'); quit;"
  cd ..
  echo "**********End ${METHOD} evaluate**********"
done

