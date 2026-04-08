#!/bin/bash

if (( $# < 2 )); then
    echo "Usage: ./inference.sh [hier: 0-2] [load-iter] [iterations (optional, default: 16)]"
    echo "Example: ./inference.sh 2 10000 16"
    exit 1
fi

hier=$1
load_iter=$2
iterations=${3:-16} # Defaults to 16 if not provided

modeldir="model/test"
train="data/train"  # Dummy path required by parser
eval="data/eval"
train_mv="data/train_mv"
eval_mv="data/eval_mv"

# Set network architecture parameters based on hierarchy level
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
  echo "Error: Hierarchy level must be 0, 1, or 2."
  exit 1
fi

echo "================================================="
echo "Running Swift Inference"
echo "Hierarchy Level: ${hier} (Distances: ${distance1}/${distance2})"
echo "Loading Checkpoint: Iteration ${load_iter}"
echo "Decoding Layers (Iterations): ${iterations}"
echo "================================================="

python -u inference.py \
  --train ${train} \
  --eval ${eval} \
  --train-mv ${train_mv} \
  --eval-mv ${eval_mv} \
  --encoder-fuse-level ${encoder_fuse_level} \
  --decoder-fuse-level ${decoder_fuse_level} \
  --v-compress --warp --stack --fuse-encoder \
  --bits ${bits} \
  --distance1 ${distance1} \
  --distance2 ${distance2} \
  --model-dir ${modeldir} \
  --load-model-name "demo" \
  --load-iter ${load_iter} \
  --iterations ${iterations}