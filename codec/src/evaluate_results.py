import os
import glob
import cv2
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
from pytorch_msssim import ms_ssim

def get_rates_from_csv(csv_path, num_frames, width, height, fps=30.0):
    """
    Parses the latent_code_sizes.csv file and calculates both the cumulative bitrate (kbps)
    and the Bits Per Pixel (BPP) for each layer.
    """
    bitrates = {}
    bpps = {}
    duration_sec = num_frames / fps
    total_pixels = num_frames * width * height
    
    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            layer_id = row['Layer']
            cum_kb = float(row['Cumulative_Kilobytes'])
            
            # Bitrate (kbps) = (Kilobytes * 8) / Duration in seconds
            bitrate_kbps = (cum_kb * 8) / duration_sec
            
            # Bits Per Pixel = (Kilobytes * 1024 bytes * 8 bits) / Total Pixels
            bpp = (cum_kb * 8192) / total_pixels
            
            bitrates[layer_id] = bitrate_kbps
            bpps[layer_id] = bpp
            
    return bitrates, bpps

def evaluate_image_directories(ref_dir, dist_dir, mask_dir):
    """
    Calculates average PSNR and MS-SSIM (both Global and Semantic Masked) 
    over all PNG frames between the original and reconstructed directories.
    """
    ref_images = sorted(glob.glob(os.path.join(ref_dir, "*.png")))
    dist_images = sorted(glob.glob(os.path.join(dist_dir, "*.png")))

    if len(ref_images) != len(dist_images) or len(ref_images) == 0:
        print(f"Warning: Frame count mismatch or missing images! Ref: {len(ref_images)}, Dist: {len(dist_images)}")
        return 0.0, 0.0, 0.0, 0.0

    psnr_vals, msssim_vals = [], []
    masked_psnr_vals, masked_msssim_vals = [], []

    for ref_path, dist_path in zip(ref_images, dist_images):
        img_ref = cv2.imread(ref_path)
        img_dist = cv2.imread(dist_path)
        
        # Load the corresponding mask for this frame
        mask_path = os.path.join(mask_dir, os.path.basename(ref_path))
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img_ref is None or img_dist is None or mask_img is None:
            continue

        # Create a binary mask (1 for semantic object, 0 for background)
        mask_bin = (mask_img > 127).astype(np.uint8)

        # Extract Luma (Y) channel for standard video coding metrics
        y_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2YUV)[:, :, 0]
        y_dist = cv2.cvtColor(img_dist, cv2.COLOR_BGR2YUV)[:, :, 0]

        # ---------------------------------------------------------
        # 1. GLOBAL METRICS
        # ---------------------------------------------------------
        frame_psnr = psnr(y_ref, y_dist, data_range=255)
        psnr_vals.append(frame_psnr)

        t_ref = torch.from_numpy(y_ref).float().unsqueeze(0).unsqueeze(0) / 255.0
        t_dist = torch.from_numpy(y_dist).float().unsqueeze(0).unsqueeze(0) / 255.0

        frame_msssim = ms_ssim(t_dist, t_ref, data_range=1.0, size_average=True)
        msssim_vals.append(frame_msssim.item())

        # ---------------------------------------------------------
        # 2. SEMANTIC MASKED METRICS (Region of Interest)
        # ---------------------------------------------------------
        # Masked PSNR Calculation
        diff = y_ref.astype(np.float32) - y_dist.astype(np.float32)
        # Calculate MSE only where the mask is 1
        masked_pixel_count = max(np.sum(mask_bin), 1) # Prevent division by zero
        mse_masked = np.sum((diff ** 2) * mask_bin) / masked_pixel_count
        
        if mse_masked == 0:
            frame_masked_psnr = 100.0 # Perfect match
        else:
            frame_masked_psnr = 10 * np.log10((255 ** 2) / mse_masked)
        masked_psnr_vals.append(frame_masked_psnr)

        # Masked MS-SSIM Calculation
        # We apply the mask directly to the images before converting to tensors
        y_ref_m = y_ref * mask_bin
        y_dist_m = y_dist * mask_bin
        
        t_ref_m = torch.from_numpy(y_ref_m).float().unsqueeze(0).unsqueeze(0) / 255.0
        t_dist_m = torch.from_numpy(y_dist_m).float().unsqueeze(0).unsqueeze(0) / 255.0

        frame_masked_msssim = ms_ssim(t_dist_m, t_ref_m, data_range=1.0, size_average=True)
        masked_msssim_vals.append(frame_masked_msssim.item())

    return np.mean(psnr_vals), np.mean(msssim_vals), np.mean(masked_psnr_vals), np.mean(masked_msssim_vals)

def process_directory(dir_name, mask_dir):
    """
    Processes a model's output directory, parses its CSV, and evaluates 
    the original PNGs against every visual enhancement layer.
    """
    base_path = os.path.join("inference_results", dir_name)
    original_dir = os.path.join(base_path, "original")
    csv_path = os.path.join(base_path, "latent_code_sizes.csv")
    
    if not os.path.exists(original_dir) or not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing original frames or CSV in {base_path}")

    # Read original images to get count and dimensions
    original_images = sorted(glob.glob(os.path.join(original_dir, "*.png")))
    num_frames = len(original_images)
    
    # Read the first image to extract width and height dynamically
    sample_img = cv2.imread(original_images[0])
    height, width = sample_img.shape[:2]
    
    # Load cumulative kbps and BPP for every layer
    bitrates, bpps = get_rates_from_csv(csv_path, num_frames, width, height)

    # Search specifically for reconstructed PNG folders (ignoring difference maps and base masks)
    search_pattern = os.path.join(base_path, "layer_*_reconstructed")
    layer_dirs = sorted(glob.glob(search_pattern))

    results = {
        'layer_ids': [], 'psnr': [], 'msssim': [], 
        'masked_psnr': [], 'masked_msssim': [], 
        'bitrate': [], 'bpp': []
    }

    for layer_dir in layer_dirs:
        folder_name = os.path.basename(layer_dir)
        layer_num = folder_name.split('_')[1]

        print(f"  -> Evaluating: {folder_name} ...")
        avg_psnr, avg_msssim, avg_mpsnr, avg_mmsssim = evaluate_image_directories(original_dir, layer_dir, mask_dir)
        
        # Match the folder layer number to the CSV layer number
        if layer_num in bitrates:
            kbps = bitrates[layer_num]
            bpp = bpps[layer_num]
        else:
            print(f"     Warning: Layer {layer_num} not found in CSV. Skipping.")
            continue
            
        results['layer_ids'].append(layer_num)
        results['psnr'].append(avg_psnr)
        results['msssim'].append(avg_msssim)
        results['masked_psnr'].append(avg_mpsnr)
        results['masked_msssim'].append(avg_mmsssim)
        results['bitrate'].append(kbps)
        results['bpp'].append(bpp)

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate RD Curves for layered codecs.")
    parser.add_argument('--log_scale', action='store_true', help='Use logarithmic scale for the X-axis.')
    parser.add_argument('--use_bpp', action='store_true', help='Use Bits Per Pixel (BPP) for the X-axis instead of Bitrate (kbps).')
    args = parser.parse_args()

    dir_baseline = "baseline_hier_2"
    dir_sem = "sem_improved_model_2"
    
    # Define the shared mask directory generated by the semantic model
    shared_mask_dir = os.path.join("inference_results", dir_sem, "layer_00_semantic_mask")
    if not os.path.exists(shared_mask_dir):
        raise FileNotFoundError(f"Semantic mask directory not found: {shared_mask_dir}")

    print(f"--- Processing {dir_baseline} ---")
    res_baseline = process_directory(dir_baseline, shared_mask_dir)

    print(f"\n--- Processing {dir_sem} ---")
    res_sem = process_directory(dir_sem, shared_mask_dir)

    # --- DYNAMIC AXIS & METRIC SELECTION LOGIC ---
    x_metric = 'bpp' if args.use_bpp else 'bitrate'
    x_label_base = 'Bits Per Pixel (BPP)' if args.use_bpp else 'Bitrate (kbps)'
    x_label = x_label_base + (' (Log Scale)' if args.log_scale else '')

    all_x = res_baseline[x_metric] + res_sem[x_metric]
    
    if args.log_scale:
        x_min = min(all_x) * 0.85
        x_max = max(all_x) * 1.15
    else:
        margin = (max(all_x) - min(all_x)) * 0.05
        x_min = min(all_x) - margin
        x_max = max(all_x) + margin

    # Create a 2x2 grid for Global and Masked plots
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # Apply log scale if requested
    if args.log_scale:
        for ax in [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]:
            ax.set_xscale('log')
    
    # ---------------------------------------------------------
    # PLOT 1: Global PSNR
    # ---------------------------------------------------------
    axs[0, 0].plot(res_baseline[x_metric], res_baseline['psnr'], marker='o', label='Baseline Model', color='blue')
    axs[0, 0].plot(res_sem[x_metric], res_sem['psnr'], marker='s', label='Semantic Model', color='red')
    axs[0, 0].set_title('Global Rate-Distortion: Y-PSNR')
    axs[0, 0].set_xlabel(x_label)
    axs[0, 0].set_ylabel('Y-PSNR (dB)')
    axs[0, 0].set_xlim(x_min, x_max)
    axs[0, 0].grid(True, which="both", ls="--", alpha=0.7)
    axs[0, 0].legend()

    # ---------------------------------------------------------
    # PLOT 2: Global MS-SSIM
    # ---------------------------------------------------------
    axs[0, 1].plot(res_baseline[x_metric], res_baseline['msssim'], marker='o', label='Baseline Model', color='blue')
    axs[0, 1].plot(res_sem[x_metric], res_sem['msssim'], marker='s', label='Semantic Model', color='red')
    axs[0, 1].set_title('Global Rate-Distortion: MS-SSIM')
    axs[0, 1].set_xlabel(x_label)
    axs[0, 1].set_ylabel('MS-SSIM')
    axs[0, 1].set_xlim(x_min, x_max)
    axs[0, 1].grid(True, which="both", ls="--", alpha=0.7)
    axs[0, 1].legend()

    # ---------------------------------------------------------
    # PLOT 3: Semantic Masked PSNR
    # ---------------------------------------------------------
    axs[1, 0].plot(res_baseline[x_metric], res_baseline['masked_psnr'], marker='o', label='Baseline Model', color='blue', linestyle='--')
    axs[1, 0].plot(res_sem[x_metric], res_sem['masked_psnr'], marker='s', label='Semantic Model', color='red', linestyle='--')
    axs[1, 0].set_title('Semantic Object Quality: Masked Y-PSNR')
    axs[1, 0].set_xlabel(x_label)
    axs[1, 0].set_ylabel('Masked Y-PSNR (dB)')
    axs[1, 0].set_xlim(x_min, x_max)
    axs[1, 0].grid(True, which="both", ls="--", alpha=0.7)
    axs[1, 0].legend()

    # ---------------------------------------------------------
    # PLOT 4: Semantic Masked MS-SSIM
    # ---------------------------------------------------------
    axs[1, 1].plot(res_baseline[x_metric], res_baseline['masked_msssim'], marker='o', label='Baseline Model', color='blue', linestyle='--')
    axs[1, 1].plot(res_sem[x_metric], res_sem['masked_msssim'], marker='s', label='Semantic Model', color='red', linestyle='--')
    axs[1, 1].set_title('Semantic Object Quality: Masked MS-SSIM')
    axs[1, 1].set_xlabel(x_label)
    axs[1, 1].set_ylabel('Masked MS-SSIM')
    axs[1, 1].set_xlim(x_min, x_max)
    axs[1, 1].grid(True, which="both", ls="--", alpha=0.7)
    axs[1, 1].legend()

    plt.tight_layout()
    
    suffix = ("_bpp" if args.use_bpp else "") + ("_log" if args.log_scale else "")
    plot_filename = f"inference_results/rd_curves_results_masked{suffix}.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"\nEvaluation complete. Comprehensive RD Curves saved to {plot_filename}")

    # ---------------------------------------------------------
    # PLOT 5 - Bitrate Offset Analysis (Only run for default mode)
    # ---------------------------------------------------------
    if not args.log_scale and not args.use_bpp:
        min_len = min(len(res_sem['bitrate']), len(res_baseline['bitrate']))
        # Calculate difference: Semantic rate - Baseline rate
        rate_offsets = [res_sem['bitrate'][i] - res_baseline['bitrate'][i] for i in range(min_len)]
        layers = range(1, min_len + 1)

        fig_offset, ax_off = plt.subplots(figsize=(8, 5))
        ax_off.bar(layers, rate_offsets, color='orange', alpha=0.7, label='Bitrate Difference')
        
        ax_off.set_title('Bitrate Difference: Semantic vs Baseline')
        ax_off.set_xlabel('Visual Layer Iteration')
        ax_off.set_ylabel('Bitrate Difference (kbps)')
        ax_off.set_xticks(layers)
        ax_off.grid(axis='y', linestyle='--', alpha=0.6)
        ax_off.legend()

        plt.tight_layout()
        overhead_filename = "inference_results/bitrate_overhead_analysis.png"
        plt.savefig(overhead_filename, dpi=300)
        print(f"Bitrate offset analysis saved to {overhead_filename}")

if __name__ == "__main__":
    main()