import argparse
import os
import time

import numpy as np

import torch
import torch.utils.data as data

from util import eval_forward, evaluate, evaluate_psnr, get_models, set_eval, save_numpy_array_as_image
from torchvision import transforms
from dataset import get_loader


def save_codes(name, codes):
  print(codes)
  codes = (codes.astype(np.int8) + 1) // 2
  export = np.packbits(codes.reshape(-1))
  np.savez_compressed(
      name + '.codes',
      shape=codes.shape,
      codes=export)


def save_output_images(name, ex_imgs):
  for i, img in enumerate(ex_imgs):
    save_numpy_array_as_image(
      '%s_iter%02d.png' % (name, i + 1),
      img
    )


def finish_batch(args, filenames, original, out_imgs,
                 losses, code_batch, output_suffix):

  all_losses, all_msssim, all_psnr = [], [], []
  for ex_idx, filename in enumerate(filenames):
      filename = filename.split('/')[-1]
      if args.save_codes:
        save_codes(
          os.path.join(args.out_dir, output_suffix, 'codes', filename),
          code_batch[:, ex_idx, :, :, :]
        )

      if args.save_out_img:
        save_output_images(
          os.path.join(args.out_dir, output_suffix, 'images', filename),
          out_imgs[:, ex_idx, :, :, :]
        )

      msssim, psnr = evaluate(
        original[None, ex_idx],
        [out_img[None, ex_idx] for out_img in out_imgs])

      all_losses.append(losses)
      all_msssim.append(msssim)
      all_psnr.append(psnr)

  return all_losses, all_msssim, all_psnr


def get_psnr(args, filenames, original, out_imgs):

  all_psnr = []
  for ex_idx, filename in enumerate(filenames):

      psnr = evaluate_psnr(
        original[None, ex_idx],
        [out_img[None, ex_idx] for out_img in out_imgs])

      all_psnr.append(psnr)

  return all_psnr


# --- NEW: Function to evaluate quality specifically in semantic regions ---
def get_semantic_psnr(args, filenames, original, out_imgs, masks):
    all_psnr = []
    
    # Convert mask tensor to numpy array (Batch, C, H, W)
    masks_np = masks.cpu().numpy()
    
    for ex_idx, filename in enumerate(filenames):
        # Extract the mask for the current image
        mask = masks_np[ex_idx]
        
        # Multiply the original image by the mask to black out the background
        orig_masked = original[ex_idx] * mask
        
        psnr_list = []
        for out_img in out_imgs:
            # Mask the reconstructed image
            out_img_masked = out_img[ex_idx] * mask
            
            # Compute MSE ONLY over the masked area to avoid division by empty space
            mask_area = np.mean(mask) + 1e-8
            mse = np.mean((orig_masked - out_img_masked) ** 2) / mask_area
            
            if mse < 1e-10:
                psnr = 100.0
            else:
                psnr = 20 * np.log10(1.0) - 10.0 * np.log10(mse)
                
            psnr_list.append(psnr)

        all_psnr.append(psnr_list)

    return all_psnr


def run_eval(model, eval_loader, args, output_suffix=''):

  for sub_dir in ['codes', 'images']:
    cur_eval_dir = os.path.join(args.out_dir, output_suffix, sub_dir)
    if not os.path.exists(cur_eval_dir):
      print("Creating directory %s." % cur_eval_dir)
      os.makedirs(cur_eval_dir)

  all_losses, all_msssim, all_psnr = [], [], []
  all_psnr_ee1, all_psnr_ee2, all_psnr_ee3, all_psnr_ee4 = [], [], [], []
  all_psnr_semantic = [] # --- NEW Tracker ---

  # FIX: Determine the primary device dynamically using args.gpus
  gpus = [int(gpu) for gpu in args.gpus.split(',')] if hasattr(args, 'gpus') and args.gpus else []
  primary_device = torch.device(f"cuda:{gpus[0]}" if len(gpus) > 0 and torch.cuda.is_available() else "cpu")

  start_time = time.time()
  
  # --- THE FIX: Add `masks` to the dataloader unpacking ---
  for i, (batch, ctx_frames, filenames, masks) in enumerate(eval_loader):

      with torch.no_grad():
          # FIX: Replaced .cuda() with .to(primary_device) to avoid DataParallel mismatch crashes
          batch = batch.to(primary_device)
          masks = masks.to(primary_device) # --- NEW: Move masks to GPU ---

          # --- MODIFIED: Pass masks to eval_forward ---
          original, out_imgs, out_imgs_ee1, out_imgs_ee2, out_imgs_ee3, out_imgs_ee4, losses, code_batch = eval_forward(
                  model, (batch, ctx_frames, masks), args)

          losses, msssim, psnr = finish_batch(
                  args, filenames, original, out_imgs,
                  losses, code_batch, output_suffix)

          psnr_ee1 = get_psnr(args, filenames, original, out_imgs_ee1)
          psnr_ee2 = get_psnr(args, filenames, original, out_imgs_ee2)
          psnr_ee3 = get_psnr(args, filenames, original, out_imgs_ee3)
          psnr_ee4 = get_psnr(args, filenames, original, out_imgs_ee4)

          # --- NEW: Calculate Semantic PSNR ---
          psnr_semantic = get_semantic_psnr(args, filenames, original, out_imgs, masks)

          all_losses += losses
          all_msssim += msssim
          all_psnr += psnr

          all_psnr_ee1 += psnr_ee1
          all_psnr_ee2 += psnr_ee2
          all_psnr_ee3 += psnr_ee3
          all_psnr_ee4 += psnr_ee4
          all_psnr_semantic += psnr_semantic # --- NEW ---

      if i % 10 == 0:
        print('\tevaluating iter %d (%f seconds)...' % (
          i, time.time() - start_time))

  # --- MODIFIED: Return Semantic PSNR metric as the 8th value in the tuple ---
  return (np.array(all_losses).mean(axis=0),
          np.array(all_msssim).mean(axis=0),
          np.array(all_psnr).mean(axis=0),
          np.array(all_psnr_ee1).mean(axis=0),
          np.array(all_psnr_ee2).mean(axis=0),
          np.array(all_psnr_ee3).mean(axis=0),
          np.array(all_psnr_ee4).mean(axis=0),
          np.array(all_psnr_semantic).mean(axis=0) # --- NEW ---
          )