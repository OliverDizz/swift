import os
import torch
import numpy as np
import argparse

# Import specific components from your codebase
from train_options import parser
from util import get_models, eval_forward, save_numpy_array_as_image
import network
from dataset import get_loader

def main():
    # --- NEW: Add custom argument for full video processing ---
    parser.add_argument('--full_video', action='store_true', 
                        help='Process all frames in the eval dataset to compile into a video.')
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

    # --- MODIFIED: Ultra-low bitrate setup for base layers (must match train.py) ---
    SEMANTIC_BITS = 4

    # PASS 0: Initialize Semantic Base Layer Models
    semantic_encoder = network.SemanticEncoder(in_channels=1).to(primary_device)
    semantic_binarizer = network.SemanticBinarizer(bits=SEMANTIC_BITS).to(primary_device)
    semantic_decoder = network.SemanticDecoder(out_channels=1, bits=SEMANTIC_BITS).to(primary_device)

    # PASS 1: Initialize Structural Edge Layer Models
    edge_encoder = network.SemanticEncoder(in_channels=1).to(primary_device)
    edge_binarizer = network.SemanticBinarizer(bits=SEMANTIC_BITS).to(primary_device)
    edge_decoder = network.SemanticDecoder(out_channels=1, bits=SEMANTIC_BITS).to(primary_device)

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
    
    out_dir = "inference_results/test_old_sem"
    
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

            # 5. Extract and Save Results for each item in the batch
            num_layers = out_imgs.shape[0]
            batch_size = batch.shape[0]
            
            for b_idx in range(batch_size):
                original_frame = original[b_idx]           
                # Use the original filename or fallback to sequence count
                base_name = os.path.basename(filenames[b_idx]).split('.')[0] if len(filenames) > b_idx else f"frame_{frame_counter:04d}"
                
                # --- Organize folders for easy video generation ---
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

            # If not generating a full video, break after the first batch
            if not args.full_video:
                print("Single batch processed. Run with --full_video to process the entire sequence.")
                break
            
            if batch_idx % 10 == 0:
                print(f"Processed {frame_counter} frames...")

    print(f"\nSuccess! Frames saved to ./{out_dir}/")
    if args.full_video:
        print("\nTo combine these into a video, you can use ffmpeg in the terminal:")
        print(f"ffmpeg -framerate 30 -pattern_type glob -i '{out_dir}/layer_02_reconstructed/*.png' -c:v libx264 -pix_fmt yuv420p output_layer_02.mp4")

if __name__ == '__main__':
    main()