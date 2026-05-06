#!/bin/bash

if (( $# != 1 )); then
    echo "Usage: ./train.sh [0-2], e.g. ./train.sh 2"
    exit
fi
hier=$1

# This is the directory where your checkpoints AND tensorboard logs will go
modeldir=model/full_sem_test_2

train="data/train"
eval="data/eval"
train_mv="data/train_mv"
eval_mv="data/eval_mv"

train_masks="data/train_masks"
eval_masks="data/eval_masks"

train_edges="data/train_edges"
eval_edges="data/eval_edges"

# --- Curriculum Learning Threshold ---
# Iterations 0 to 30k: Semantic & Edge base layers only.
# Iterations 30k to 50k: Visual enhancement layers.
phase1_iters=15000 
max_iters=50000

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

echo "================================================="
echo "Starting SVC Training (Semantic + Visual Layers)"
echo "Hierarchy Level: ${hier}"
echo "Model Directory: ${modeldir}"
echo "-------------------------------------------------"
echo "Execution Plan:"
echo "  -> Phase 1 (Base): 0 - ${phase1_iters} iters"
echo "  -> Phase 2 (Visual): ${phase1_iters} - ${max_iters} iters"
echo "================================================="

# Execute Python Training
python -u train.py \
  --train ${train} \
  --eval ${eval} \
  --train-mv ${train_mv} \
  --eval-mv ${eval_mv} \
  --train-masks ${train_masks} \
  --eval-masks ${eval_masks} \
  --train-edges ${train_edges} \
  --eval-edges ${eval_edges} \
  --encoder-fuse-level ${encoder_fuse_level} \
  --decoder-fuse-level ${decoder_fuse_level} \
  --phase1-iters ${phase1_iters} \
  --max-train-iters ${max_iters} \
  --v-compress --warp --stack --fuse-encoder \
  --bits ${bits} \
  --distance1 ${distance1} \
  --distance2 ${distance2} \
  --lr 0.0004 \
  --model-dir ${modeldir} \
  --batch-size 2 \
  --gpus 0,1 \
  --checkpoint-iters 25000 \
  --eval-iters 50000