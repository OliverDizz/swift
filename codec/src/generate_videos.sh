#!/bin/bash

# Default to sem_improved_model if no argument is provided
INPUT_DIR=${1:-"inference_results/baseline_hier_2"}
OUTPUT_DIR="video_results/baseline_hier_2"

# Create the video results folder if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "================================================="
echo "Generating Videos from Frames"
echo "Input Directory: $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "================================================="

# Check if the input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory '$INPUT_DIR' not found!"
    exit 1
fi

# Loop through every sub-directory in the input directory
for folder in "$INPUT_DIR"/*/; do
    # Remove the trailing slash to get the clean folder name
    folder=${folder%/}
    folder_name=$(basename "$folder")

    # Skip folders that contain the word "difference"
    if [[ "$folder_name" == *difference* ]]; then
        echo "Skipping difference folder: $folder_name..."
        continue
    fi

    echo "Processing: $folder_name..."

    # Run FFmpeg to compile the PNGs into an MP4
    # -y: Overwrite without asking
    # -hide_banner -loglevel error: Keeps the terminal output clean
    ffmpeg -y -framerate 30 -pattern_type glob -i "$folder/*.png" \
           -c:v libx264 -pix_fmt yuv420p \
           -hide_banner -loglevel error \
           "$OUTPUT_DIR/${folder_name}.mp4"

    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        echo "  -> Saved: $OUTPUT_DIR/${folder_name}.mp4"
    else
        echo "  -> WARNING: Failed to process $folder_name (Are there .png files inside?)"
    fi
done

echo "================================================="
echo "All videos successfully generated in ./$OUTPUT_DIR/"