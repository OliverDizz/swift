import os
import torch
import numpy as np
import argparse
import csv

# Import specific components from your codebase
from train_options import parser
from util import get_models, eval_forward, save_numpy_array_as_image
import network
from dataset import get_loader

# --- ADDED: Entropy Calculation Function (Type-Safe) ---
def calculate_entropy_bits(data_input):
    """
    Calculates the theoretical minimum bits required to transmit this tensor/array
    using Shannon Entropy, simulating a perfect entropy encoder.
    """
    # --- FIXED: Handle both PyTorch tensors and NumPy arrays safely ---
    if hasattr(data_input, 'detach'):
        # It's a PyTorch tensor, move to CPU and convert to numpy
        data = data_input.detach().cpu().numpy().flatten()
    else:
        # It's already a NumPy array or list, just flatten it
        data = np.array(data_input).flatten()
    
    # Count occurrences of each unique value
    _, counts = np.unique(data, return_counts=True)
    
    # Calculate probabilities
    probabilities = counts / len(data)
    
    # Shannon Entropy formula: H = -sum(p * log2(p))
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    # Total bits is the entropy per symbol times the number of symbols
    return entropy * len(data)

def main():
    parser.add_argument('--full_video', action='store_true', 
                        help='Process all frames in the eval dataset to compile into a video.')
    
    parser.add_argument('--out_dir', type=str, default='inference_results/default_run',
                        help='Directory to save the exported frames.')
    
    args = parser.parse_args()
    
    gpus = [int(gpu) for gpu in args.gpus.split(',')] if hasattr(args, 'gpus') and args.gpus else []
    primary_device = torch.device(f"cuda:{gpus[0]}" if len(gpus) > 0 and torch.cuda.is_available() else "cpu")

    print("Initializing models...")
    # 1. Initialize visual models
    encoder, binarizer, decoder, unet = get_models(
        args=args, v_compress=args.v_compress,
        bits=args.bits,
        encoder_fuse_level=args.encoder_fuse_level,
        decoder_fuse_level=args.decoder_fuse_level
    )
    
    d2 = network.DecoderCell2(
        v_compress=args.v_compress, 
        shrink=args.shrink,
        bits=args.bits, 
        fuse_level=args.decoder_fuse_level
    ).to(primary_device)

    SEMANTIC_BITS = 4

    # PASS 0: Initialize Semantic Base Layer Models
    semantic_encoder = network.SemanticEncoder(in_channels=1).to(primary_device)
    semantic_binarizer = network.SemanticBinarizer(bits=SEMANTIC_BITS).to(primary_device)
    semantic_decoder = network.SemanticDecoder(out_channels=1, bits=SEMANTIC_BITS).to(primary_device)

    # PASS 1: Initialize Structural Edge Layer Models
    edge_encoder = network.EdgeEncoder(in_channels=1).to(primary_device)
    edge_binarizer = network.EdgeBinarizer(bits=SEMANTIC_BITS).to(primary_device)
    edge_decoder = network.EdgeDecoder(out_channels=1, bits=SEMANTIC_BITS).to(primary_device)

    nets = [encoder, binarizer, decoder, d2, 
            semantic_encoder, semantic_binarizer, semantic_decoder,
            edge_encoder, edge_binarizer, edge_decoder]
    
    names = ['encoder', 'binarizer', 'decoder', 'd2', 
             'semantic_encoder', 'semantic_binarizer', 'semantic_decoder',
             'edge_encoder', 'edge_binarizer', 'edge_decoder']
    
    if unet is not None:
        nets.append(unet)
        names.append('unet')
        
    # Set to evaluation mode
    for net in nets:
        if net is not None:
            net.eval()

    # 2. Load Checkpoints
    print(f"Loading weights from {args.model_dir} (Iter: {args.load_iter})...")
    for net_idx, net in enumerate(nets):
        if net is not None:
            name = names[net_idx]
            checkpoint_path = '{}/{}_{}_{:08d}.pth'.format(
                args.model_dir, args.load_model_name, name, args.load_iter)
            
            if os.path.exists(checkpoint_path):
                print(f"  -> Loading {name}")
                net.load_state_dict(torch.load(checkpoint_path))
            else:
                print(f"  -> WARNING: Checkpoint not found: {checkpoint_path}")

    # 3. Load evaluation dataset
    print("\nLoading evaluation data...")
    eval_loader = get_loader(
        is_train=False, 
        root=args.eval, 
        mv_dir=args.eval_mv, 
        mask_dir=args.eval_masks,
        edge_dir=args.eval_edges,
        args=args
    )
    
    out_dir = args.out_dir
    
    # --- MODIFIED: Latent size counters (now tracking true entropy bits) ---
    total_bits_semantic = 0.0
    total_bits_edge = 0.0
    total_bits_visual = [] # Will hold a list of bit counts for each visual layer
    
    # 4. Run Inference Loop
    print(f"Running inference... (Full Video Mode: {args.full_video})")
    
    frame_counter = 0
    with torch.no_grad():
        for batch_idx, (batch, ctx_frames, filenames, masks, edges) in enumerate(eval_loader):
            batch = batch.to(primary_device)
            masks = masks.to(primary_device)
            edges = edges.to(primary_device)

            # eval_forward handles Pass 2-N
            original, out_imgs, out_imgs_ee1, out_imgs_ee2, out_imgs_ee3, out_imgs_ee4, losses, codes = eval_forward(
                nets, (batch, ctx_frames, masks, edges), args)
                
            # Manual extraction of Pass 0 (Semantic)
            sem_encoded = semantic_encoder(masks)
            sem_codes = semantic_binarizer(sem_encoded)
            reconstructed_semantics = semantic_decoder(sem_codes)

            # Manual extraction of Pass 1 (Structural)
            edge_encoded = edge_encoder(edges)
            edge_codes = edge_binarizer(edge_encoded)
            reconstructed_edges = edge_decoder(edge_codes)

            # --- MODIFIED: Calculate actual transmitted size using Shannon Entropy ---
            # Instead of .numel(), we evaluate how well the data compresses.
            total_bits_semantic += calculate_entropy_bits(sem_codes)
            total_bits_edge += calculate_entropy_bits(edge_codes)
            
            num_layers = out_imgs.shape[0]
            if not total_bits_visual:
                total_bits_visual = [0.0] * num_layers
            
            # Extract the actual binarized codes for the visual passes and calculate their entropy
            for i in range(num_layers):
                # --- FIXED: Layers are the first dimension, so we just use codes[i] ---
                layer_codes = codes[i]
                total_bits_visual[i] += calculate_entropy_bits(layer_codes)

            # 5. Extract and Save Results for each item in the batch
            batch_size = batch.shape[0]
            
            for b_idx in range(batch_size):
                original_frame = original[b_idx]           
                base_name = os.path.basename(filenames[b_idx]).split('.')[0] if len(filenames) > b_idx else f"frame_{frame_counter:04d}"
                
                folders = [
                    f"{out_dir}/original",
                    f"{out_dir}/layer_00_semantic_mask",
                    f"{out_dir}/layer_01_structural_edge"
                ]
                for i in range(num_layers):
                    layer_num = str(i + 2).zfill(2)
                    folders.append(f"{out_dir}/layer_{layer_num}_reconstructed")
                    folders.append(f"{out_dir}/layer_{layer_num}_difference")
                
                for f in folders:
                    os.makedirs(f, exist_ok=True)

                # Save original
                save_numpy_array_as_image(f"{out_dir}/original/{base_name}.png", original_frame)
                
                # Save Pass 0 (Semantic)
                recon_mask_img = reconstructed_semantics[b_idx].cpu().numpy()
                recon_mask_img = np.repeat(recon_mask_img, 3, axis=0)
                save_numpy_array_as_image(f"{out_dir}/layer_00_semantic_mask/{base_name}.png", recon_mask_img)

                # Save Pass 1 (Edge)
                recon_edge_img = reconstructed_edges[b_idx].cpu().numpy()
                recon_edge_img = np.repeat(recon_edge_img, 3, axis=0)
                save_numpy_array_as_image(f"{out_dir}/layer_01_structural_edge/{base_name}.png", recon_edge_img)
                
                # Save Pass 2-N (Visual Enhancements)
                for i in range(num_layers):
                    current_layer_reconstruction = out_imgs[i, b_idx]
                    
                    difference_map = np.abs(original_frame - current_layer_reconstruction)
                    difference_map_enhanced = np.clip(difference_map * 5.0, 0, 1)
                    
                    layer_num = str(i + 2).zfill(2) 
                    
                    save_numpy_array_as_image(f"{out_dir}/layer_{layer_num}_reconstructed/{base_name}.png", current_layer_reconstruction)
                    save_numpy_array_as_image(f"{out_dir}/layer_{layer_num}_difference/{base_name}.png", difference_map_enhanced)
                
                frame_counter += 1

            if not args.full_video:
                print("Single batch processed. Run with --full_video to process the entire sequence.")
                break
            
            if batch_idx % 10 == 0:
                print(f"Processed {frame_counter} frames...")

    print(f"\nSuccess! Frames saved to ./{out_dir}/")
    
    if args.full_video:
        csv_path = os.path.join(out_dir, "latent_code_sizes.csv")
        print(f"\nExporting true latent code bitrates to {csv_path}...")
        
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Layer', 'Layer_Type', 'Total_Entropy_Bits', 'Total_Kilobytes', 'Cumulative_Kilobytes'])
            
            cumulative_kb = 0.0
            
            # Write Semantic Layer
            kb = total_bits_semantic / 8192.0 # (bits / 8) / 1024
            cumulative_kb += kb
            writer.writerow(['00', 'Semantic_Mask', f"{total_bits_semantic:.2f}", f"{kb:.2f}", f"{cumulative_kb:.2f}"])
            
            # Write Edge Layer
            kb = total_bits_edge / 8192.0
            cumulative_kb += kb
            writer.writerow(['01', 'Structural_Edge', f"{total_bits_edge:.2f}", f"{kb:.2f}", f"{cumulative_kb:.2f}"])
            
            # Write Visual Layers
            for i in range(num_layers):
                kb = total_bits_visual[i] / 8192.0
                cumulative_kb += kb
                layer_num = str(i + 2).zfill(2)
                writer.writerow([layer_num, f'Visual_Enhancement_{i}', f"{total_bits_visual[i]:.2f}", f"{kb:.2f}", f"{cumulative_kb:.2f}"])

        print("\nTo combine these into a video, you can use the generate_videos.sh script:")
        print(f"./generate_videos.sh {out_dir}")

if __name__ == '__main__':
    main()