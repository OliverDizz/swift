#!/bin/bash

if (( $# < 2 )); then
    echo "Usage: ./inference.sh [hier: 0-2] [load-iter] [iterations (optional, default: 10)] [full_video (optional, 0 or 1, default: 0)]"
    echo "Example (Single Batch): ./inference.sh 0 2200 10 0"
    echo "Example (Full Video):   ./inference.sh 0 2200 10 1"
    exit 1
fi

hier=$1
load_iter=$2
iterations=${3:-10} # Defaults to 10 (matching your training setup)
full_video_param=${4:-0} # Defaults to 0 (Single Batch)

# Process the full video flag
full_video_flag=""
if [[ "$full_video_param" == "1" ]]; then
    full_video_flag="--full_video"
    mode_string="Full Video Processing"
else
    mode_string="Single Batch Only"
fi

# --- MODIFIED: Point to the new semantic model directory ---
modeldir="model/full_sem_test"
#modeldir="baseline_model/baseline"

train="data/train"  # Dummy path required by parser
eval="data/eval"
train_mv="data/train_mv"
eval_mv="data/eval_mv"

# --- NEW: Define Mask and Edge Directories ---
train_masks="data/train_masks"
eval_masks="data/eval_masks"
train_edges="data/train_edges"
eval_edges="data/eval_edges"

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
echo "Running Swift Inference (Semantic & Edge Base Layers)"
echo "Hierarchy Level: ${hier} (Distances: ${distance1}/${distance2})"
echo "Loading Checkpoint: Iteration ${load_iter}"
echo "Decoding Layers (Iterations): ${iterations}"
echo "Execution Mode: ${mode_string}"
echo "================================================="

python -u inference.py \
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
  --v-compress --warp --stack --fuse-encoder \
  --bits ${bits} \
  --distance1 ${distance1} \
  --distance2 ${distance2} \
  --model-dir ${modeldir} \
  --load-model-name "demo" \
  --load-iter ${load_iter} \
  --iterations ${iterations} \
  ${full_video_flag}