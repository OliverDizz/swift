import cv2
import os
import numpy as np
from tqdm import tqdm

def generate_edge_maps(image_root, mask_root, output_root, sigma=0.4, test_mode=False):
    """
    Generates balanced Auto-Canny edge maps.
    Includes Foreground-only thresholding, moderate CLAHE texture boosting, and bilateral filtering.
    """
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    for category in os.listdir(image_root):
        # If testing, skip everything until we hit the 'silent' folder
        if test_mode and category != 'silent':
            continue

        img_category_dir = os.path.join(image_root, category)
        
        if not os.path.isdir(img_category_dir):
            continue
            
        out_category_dir = os.path.join(output_root, category)
        if not os.path.exists(out_category_dir):
            os.makedirs(out_category_dir)
            
        for filename in tqdm(os.listdir(img_category_dir), desc=f"Processing {category}"):
            if not filename.endswith('.png'):
                continue

            try:
                img_idx = int(filename.split('_')[1].split('.')[0])
            except ValueError:
                continue

            mask_idx = img_idx - 1
            mask_filename = f"frame_{mask_idx:05d}.png"
            
            img_path = os.path.join(img_category_dir, filename)
            mask_path = os.path.join(mask_root, category, mask_filename)
            save_path = os.path.join(out_category_dir, filename)

            img = cv2.imread(img_path)
            if img is None:
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 1. Apply the Semantic Mask FIRST
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    if mask.shape != gray.shape:
                        mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]))
                    gray = cv2.bitwise_and(gray, gray, mask=mask)
                else:
                    print(f"\nWarning: Mask is unreadable at {mask_path}")
            else:
                print(f"\nWarning: Mask file not found at {mask_path}")

            # 2. Moderate Texture Boost (Toned down from 2.0 to 1.2)
            # Prevents over-amplifying micro-noise into hard edges
            clahe = cv2.createCLAHE(clipLimit=0.9, tileGridSize=(8, 8))
            gray_boosted = clahe.apply(gray)

            # 3. Bilateral Filter (Increased from 25 to 40)
            # Smooths out more of the flat noise while preserving the CLAHE-boosted textures
            blurred = cv2.bilateralFilter(gray_boosted, d=7, sigmaColor=40, sigmaSpace=40)
            
            # 4. Smart Auto-Canny
            foreground_pixels = blurred[blurred > 0]
            if len(foreground_pixels) > 0:
                v = np.median(foreground_pixels)
            else:
                v = 0 
                
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            edges = cv2.Canny(blurred, lower, upper, L2gradient=True)

            # 5. Morphological Dilation (Currently disabled)
            #kernel = np.ones((2, 2), np.uint8)
            #edges = cv2.dilate(edges, kernel, iterations=1)

            cv2.imwrite(save_path, edges)

            if test_mode:
                print(f"\n[TEST MODE ACTIVE] Processed and saved 1 image: {save_path}")
                print(f"-> Foreground Median: {v}")
                print(f"-> Calculated Auto-Canny Thresholds: Lower={lower}, Upper={upper}")
                return

if __name__ == "__main__":
    base_dir = "data"
    
    # Set to True to verify the new texture balance on eval/silent
    TEST_MODE = False 
    
    # Target the eval split exclusively if we are testing
    splits = ['eval'] if TEST_MODE else ['train', 'eval']

    for split in splits:
        img_dir = os.path.join(base_dir, split)
        mask_dir = os.path.join(base_dir, f"{split}_masks")
        edge_dir = os.path.join(base_dir, f"{split}_edge")
        
        if os.path.exists(img_dir) and os.path.exists(mask_dir):
            print(f"\n--- Generating {split} edge maps ---")
            # sigma=0.4 is a good balance here with the lowered CLAHE
            generate_edge_maps(img_dir, mask_dir, edge_dir, sigma=40, test_mode=TEST_MODE)
        else:
            print(f"\nSkipping {split}: could not find {img_dir} or {mask_dir}")