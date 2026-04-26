import os
import torch
import numpy as np

# Import specific components from your codebase
from train_options import parser
from util import get_models, eval_forward, save_numpy_array_as_image
import network
from dataset import get_loader

def main():
    # Parse arguments (assumes you will pass similar args as train.sh)
    args = parser.parse_args()
    
    print("Initializing models...")
    # 1. Initialize models
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
    ).cuda()

    # --- NEW: Initialize Semantic Base Layer Models ---
    semantic_encoder = network.SemanticEncoder(in_channels=1).cuda()
    semantic_binarizer = network.Binarizer(bits=args.bits).cuda()
    semantic_decoder = network.SemanticDecoder(out_channels=1, bits=args.bits).cuda()

    # --- MODIFIED: Add semantic networks to the tracking list ---
    nets = [encoder, binarizer, decoder, d2, semantic_encoder, semantic_binarizer, semantic_decoder]
    names = ['encoder', 'binarizer', 'decoder', 'd2', 'semantic_encoder', 'semantic_binarizer', 'semantic_decoder']
    
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

    # 3. Load a single batch from the evaluation dataset
    print("\nLoading evaluation data...")
    eval_loader = get_loader(
        is_train=False, 
        root=args.eval, 
        mv_dir=args.eval_mv, 
        mask_dir=args.eval_masks,
        args=args
    )
    
    # --- MODIFIED: Unpack the masks ---
    batch, ctx_frames, filenames, masks = next(iter(eval_loader))
    batch = batch.cuda()
    masks = masks.cuda()

    # 4. Run Inference
    print("Running inference forward pass...")
    with torch.no_grad():
        # eval_forward handles the iterative visual compression
        # We pass the full `nets` list so it can access all models
        original, out_imgs, out_imgs_ee1, out_imgs_ee2, out_imgs_ee3, out_imgs_ee4, losses, codes = eval_forward(
            nets, (batch, ctx_frames, masks), args)
            
        # --- NEW: Run Pass 0 (Semantic) manually to extract the mask image for saving ---
        sem_encoded = semantic_encoder(masks)
        sem_codes = semantic_binarizer(sem_encoded)
        reconstructed_semantics = semantic_decoder(sem_codes)

    # 5. Extract and Save Results
    print("\nProcessing and saving images...")
    
    original_frame = original[0]           
    base_name = os.path.basename(filenames[0]).split('.')[0]
    out_dir = "inference_results"
    os.makedirs(out_dir, exist_ok=True)
    
    # Save the original reference frame
    save_numpy_array_as_image(f"{out_dir}/{base_name}_original.png", original_frame)
    
    # --- FIXED: Save the Semantic Mask outputs (Layer 0) ---
    # We keep the shape as [C, H, W] because save_numpy_array_as_image will transpose it.
    # masks[0] is [1, H, W]. We repeat across axis 0 to make it [3, H, W]
    gt_mask_img = masks[0].cpu().numpy()
    gt_mask_img = np.repeat(gt_mask_img, 3, axis=0) 
    
    recon_mask_img = reconstructed_semantics[0].cpu().numpy()
    recon_mask_img = np.repeat(recon_mask_img, 3, axis=0)
    
    save_numpy_array_as_image(f"{out_dir}/{base_name}_original_mask.png", gt_mask_img)
    save_numpy_array_as_image(f"{out_dir}/{base_name}_layer_00_semantic_mask.png", recon_mask_img)
    
    # --- VISUAL LAYERS ---
    # Loop through every single visual iteration (layers 1 to N)
    num_layers = out_imgs.shape[0]
    for i in range(num_layers):
        current_layer_reconstruction = out_imgs[i, 0]
        
        # Calculate difference map for this specific layer
        difference_map = np.abs(original_frame - current_layer_reconstruction)
        difference_map_enhanced = np.clip(difference_map * 5.0, 0, 1)
        
        # Save the reconstruction and difference map for this layer
        layer_num = str(i + 1).zfill(2) # Formats as 01, 02, 03...
        save_numpy_array_as_image(f"{out_dir}/{base_name}_layer_{layer_num}_reconstructed.png", current_layer_reconstruction)
        save_numpy_array_as_image(f"{out_dir}/{base_name}_layer_{layer_num}_difference.png", difference_map_enhanced)
        
    print(f"Success! {num_layers} visual layers and 1 semantic layer saved to ./{out_dir}/")

if __name__ == '__main__':
    main()