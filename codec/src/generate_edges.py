import cv2
import os
import numpy as np
from tqdm import tqdm

def generate_edge_maps(image_root, mask_root, output_root, low_thresh=100, high_thresh=200):
    """
    Generates Canny edge maps, correctly mapping 4-digit 1-indexed image filenames
    to 5-digit 0-indexed mask filenames, while preserving sub-directory structures.
    """
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # Iterate over video categories (bluesky, flower, foreman, etc.)
    for category in os.listdir(image_root):
        img_category_dir = os.path.join(image_root, category)
        
        # Skip if it's not a directory
        if not os.path.isdir(img_category_dir):
            continue
            
        out_category_dir = os.path.join(output_root, category)
        if not os.path.exists(out_category_dir):
            os.makedirs(out_category_dir)
            
        # Process each frame in the category
        for filename in tqdm(os.listdir(img_category_dir), desc=f"Processing {category}"):
            if not filename.endswith('.png'):
                continue

            # 1. Parse the image index (e.g., extracts 1 from "frame_0001.png")
            try:
                img_idx = int(filename.split('_')[1].split('.')[0])
            except ValueError:
                continue

            # 2. Map to the mask filename (subtract 1 and use 5 digits)
            # frame_0001.png -> frame_00000.png
            mask_idx = img_idx - 1
            mask_filename = f"frame_{mask_idx:05d}.png"
            
            img_path = os.path.join(img_category_dir, filename)
            mask_path = os.path.join(mask_root, category, mask_filename)
            
            # We save the edge map using the ORIGINAL image name (4-digits) 
            # so the dataloader can easily fetch 'frame_0001.png' across all folders
            save_path = os.path.join(out_category_dir, filename)

            # 3. Load image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Convert to grayscale and blur slightly to reduce noise
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Generate raw Canny edges
            edges = cv2.Canny(blurred, low_thresh, high_thresh)

            # 4. Apply the Semantic Mask
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Safety check in case resolutions differ slightly
                    if mask.shape != edges.shape:
                        mask = cv2.resize(mask, (edges.shape[1], edges.shape[0]))
                    
                    # Mask out the background edges
                    edges = cv2.bitwise_and(edges, edges, mask=mask)
                else:
                    print(f"\nWarning: Mask is unreadable at {mask_path}")
            else:
                print(f"\nWarning: Mask file not found at {mask_path}")

            # 5. Save the final edge map
            cv2.imwrite(save_path, edges)

if __name__ == "__main__":
    # Assuming script is run from /nfs/home/olidiz18/git/swift/codec/src/
    base_dir = "data"
    
    # Process both train and eval datasets
    for split in ['train', 'eval']:
        img_dir = os.path.join(base_dir, split)
        mask_dir = os.path.join(base_dir, f"{split}_masks")
        edge_dir = os.path.join(base_dir, f"{split}_edge")
        
        if os.path.exists(img_dir) and os.path.exists(mask_dir):
            print(f"\n--- Generating {split} edge maps ---")
            print(f"Outputting to: {edge_dir}")
            generate_edge_maps(img_dir, mask_dir, edge_dir)
        else:
            print(f"\nSkipping {split}: could not find {img_dir} or {mask_dir}")