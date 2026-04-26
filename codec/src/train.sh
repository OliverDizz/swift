#!/bin/bash

if (( $# != 1 )); then
    echo "Usage: ./train.sh [0-2], e.g. ./train.sh 2"
    exit
fi
hier=$1

modeldir=model/sem_test_edge

train="data/train"
eval="data/eval"
train_mv="data/train_mv"
eval_mv="data/eval_mv"

train_masks="data/train_masks"
eval_masks="data/eval_masks"

train_edge="data/train_edge"
eval_edge="data/eval_edge"

if [[ ${hier} == "0" ]]; then
  distance1=6
  distance2=6
  bits=16
  encoder_fuse_level=1
  decoder_fuse_level=1
elif [[ ${hier} == "1" ]]; then
  distance1=3
  distance2=3
  bits=16
  encoder_fuse_level=2
  decoder_fuse_level=3
elif [[ ${hier} == "2" ]]; then
  distance1=1
  distance2=2
  bits=8
  encoder_fuse_level=1
  decoder_fuse_level=1
else
  echo "Usage: ./train.sh [0-2], e.g. ./train.sh 2"
  exit
fi

# Execute Python Training
python -u train.py \
  --train ${train} \
  --eval ${eval} \
  --train-mv ${train_mv} \
  --eval-mv ${eval_mv} \
  --train-masks ${train_masks} \
  --eval-masks ${eval_masks} \
  --train-edges ${train_edge} \
  --eval-edges ${eval_edge} \
  --encoder-fuse-level ${encoder_fuse_level} \
  --decoder-fuse-level ${decoder_fuse_level} \
  --v-compress --warp --stack --fuse-encoder \
  --bits ${bits} \
  --distance1 ${distance1} \
  --distance2 ${distance2} \
  --max-train-iters 10000 \
  --checkpoint-iters 10000 \
  --eval-iters 5000 \
  --model-dir ${modeldir} \
  --batch-size 2 \
  --gpus 0,1 \
  --schedule "5000,10000,20000,30000,40000"

  # Full training iters
  #--max-train-iters 50000 \
  #--checkpoint-iters 5000 \
  #--eval-iters 1000 \