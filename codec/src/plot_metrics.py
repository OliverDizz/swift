import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from metric import psnr, msssim

def calculate_rd_data(results_dir, is_semantic=False, bits=8):
    orig_files = glob.glob(os.path.join(results_dir, "*_original.png"))
    if not orig_files: return None
    
    orig_path = orig_files[0]
    base_name = os.path.basename(orig_path).replace("_original.png", "")
    img_orig = cv2.cvtColor(cv2.imread(orig_path), cv2.COLOR_BGR2RGB)

    bpp_list, psnr_list, ssim_list = [], [], []

    # Baseline: layers 01-10 | Semantic: layers 02-11
    start_layer = 2 if is_semantic else 1
    end_layer = 12 if is_semantic else 11

    for layer_idx in range(start_layer, end_layer):
        layer_str = str(layer_idx).zfill(2)
        recon_path = os.path.join(results_dir, f"{base_name}_layer_{layer_str}_reconstructed.png")
        
        if not os.path.exists(recon_path): continue
            
        img_recon = cv2.cvtColor(cv2.imread(recon_path), cv2.COLOR_BGR2RGB)
        
        # Calculate BPP
        # Semantic layer 02 (iter 1) = Mask(1) + Edge(1) + Visual(1) = 3 units
        # Baseline layer 01 (iter 1) = Visual(1) = 1 unit
        units = (layer_idx + 1) if is_semantic else layer_idx
        bpp = (units * bits) / 256.0
        
        bpp_list.append(bpp)
        psnr_list.append(psnr(img_orig, img_recon))
        ssim_list.append(msssim(img_orig, img_recon))

    return bpp_list, psnr_list, ssim_list

def plot_rd_comparison(base_data, sem_data, bits):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot PSNR vs BPP
    ax1.plot(base_data[0], base_data[1], 'o--', color='gray', label='Baseline (SWIFT)')
    ax1.plot(sem_data[0], sem_data[1], 'o-', color='navy', label='Semantic 3-Layer')
    ax1.set_xlabel('Bits Per Pixel (BPP)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax1.set_title('Rate-Distortion: PSNR', fontsize=14)
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax1.legend()

    # Plot MS-SSIM vs BPP
    ax2.plot(base_data[0], base_data[2], 's--', color='gray', label='Baseline (SWIFT)')
    ax2.plot(sem_data[0], sem_data[2], 's-', color='darkred', label='Semantic 3-Layer')
    ax2.set_xlabel('Bits Per Pixel (BPP)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MS-SSIM', fontsize=12, fontweight='bold')
    ax2.set_title('Rate-Distortion: MS-SSIM', fontsize=14)
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    ax2.legend()

    plt.suptitle(f"Standard RD Curve Analysis (Bottleneck bits={bits})", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("inference_results/rd_curve_comparison.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    BITS_VAL = 8 # Match your training --bits arg
    
    print("Calculating Baseline Metrics...")
    base_res = calculate_rd_data("inference_results/baseline", is_semantic=False, bits=BITS_VAL)
    
    print("Calculating Semantic Metrics...")
    sem_res = calculate_rd_data("inference_results/semantic", is_semantic=True, bits=BITS_VAL)
    
    if base_res and sem_res:
        plot_rd_comparison(base_res, sem_res, BITS_VAL)