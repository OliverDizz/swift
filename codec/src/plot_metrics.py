import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import the exact evaluation metrics from your SWIFT codebase
from metric import psnr, msssim

def calculate_layer_metrics(results_dir="inference_results"):
    # 1. Find the original ground truth frame
    orig_files = glob.glob(os.path.join(results_dir, "*_original.png"))
    if not orig_files:
        print(f"No original image found in {results_dir}")
        return

    orig_path = orig_files[0]
    base_name = os.path.basename(orig_path).replace("_original.png", "")
    print(f"Processing sequence: {base_name}")

    # Load original image and convert BGR to RGB
    img_orig = cv2.imread(orig_path)
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

    layers = []
    psnr_scores = []
    msssim_scores = []

    # 2. Loop through the visual enhancement layers (02 to 11)
    # Note: Layer 00 is Semantic, Layer 01 is Edge, so visual pixels start at 02
    for i in range(2, 12):
        layer_str = str(i).zfill(2)
        recon_path = os.path.join(results_dir, f"{base_name}_layer_{layer_str}_reconstructed.png")
        
        if not os.path.exists(recon_path):
            continue
            
        # Load reconstructed image
        img_recon = cv2.imread(recon_path)
        img_recon = cv2.cvtColor(img_recon, cv2.COLOR_BGR2RGB)
        
        # 3. Calculate Metrics
        # Your metric.py expects numpy arrays in (H, W, C) format with values 0-255
        val_psnr = psnr(img_orig, img_recon)
        val_msssim = msssim(img_orig, img_recon)
        
        layers.append(i - 1)  # Shift x-axis so the first visual iteration is "1"
        psnr_scores.append(val_psnr)
        msssim_scores.append(val_msssim)
        
        print(f"Visual Iteration {i-1} | PSNR: {val_psnr:.2f} dB | MS-SSIM: {val_msssim:.4f}")

    # 4. Plot the results
    create_dual_axis_plot(layers, psnr_scores, msssim_scores, base_name, results_dir)

def create_dual_axis_plot(layers, psnr_scores, msssim_scores, base_name, results_dir):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Visual Enhancement Layer (Iteration)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('PSNR (dB)', color=color1, fontsize=12, fontweight='bold')
    line1 = ax1.plot(layers, psnr_scores, color=color1, marker='o', linewidth=2, label='PSNR')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()  
    color2 = 'tab:red'
    ax2.set_ylabel('MS-SSIM', color=color2, fontsize=12, fontweight='bold')
    line2 = ax2.plot(layers, msssim_scores, color=color2, marker='s', linewidth=2, label='MS-SSIM')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Combine legends from both axes
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', fontsize=12)

    plt.title(f"Reconstruction Quality per Visual Layer\n({base_name})", fontsize=14, fontweight='bold')
    
    # Ensure integer ticks on the X-axis
    plt.xticks(layers)
    
    # Save the plot
    plot_path = os.path.join(results_dir, f"{base_name}_quality_metrics_plot.png")
    fig.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"\nPlot saved successfully to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    calculate_layer_metrics()