import argparse
import os
import time

import numpy as np

import torch
import torch.utils.data as data
import torch.nn.functional as F

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

# --- NEW: Independent Base Layer Evaluation Metrics (SVC Requirement) ---
def calculate_iou_dice(pred, target, threshold=0.5):
    """Calculates Intersection over Union (IoU) and Dice Score for binary masks."""
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    
    intersection = (pred_bin * target_bin).sum(dim=[1, 2, 3])
    union = pred_bin.sum(dim=[1, 2, 3]) + target_bin.sum(dim=[1, 2, 3]) - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2. * intersection + 1e-6) / (pred_bin.sum(dim=[1, 2, 3]) + target_bin.sum(dim=[1, 2, 3]) + 1e-6)
    
    return iou.mean().item(), dice.mean().item()


def run_eval(model, eval_loader, args, output_suffix=''):

  for sub_dir in ['codes', 'images']:
    cur_eval_dir = os.path.join(args.out_dir, output_suffix, sub_dir)
    if not os.path.exists(cur_eval_dir):
      print("Creating directory %s." % cur_eval_dir)
      os.makedirs(cur_eval_dir)

  all_losses, all_msssim, all_psnr = [], [], []
  all_psnr_ee1, all_psnr_ee2, all_psnr_ee3, all_psnr_ee4 = [], [], [], []
  all_psnr_semantic = [] 
  
  # --- NEW Trackers for Base Layer Fidelity ---
  all_mask_iou, all_mask_dice = [], []
  all_edge_iou, all_edge_dice = [], []

  # FIX: Determine the primary device dynamically using args.gpus
  gpus = [int(gpu) for gpu in args.gpus.split(',')] if hasattr(args, 'gpus') and args.gpus else []
  primary_device = torch.device(f"cuda:{gpus[0]}" if len(gpus) > 0 and torch.cuda.is_available() else "cpu")

  # Extract base layer models from the `nets` list passed from train.py
  # nets = [encoder, binarizer, decoder, d2, sem_enc, sem_bin, sem_dec, edge_enc, edge_bin, edge_dec, unet]
  sem_enc, sem_bin, sem_dec = model[4], model[5], model[6]
  edge_enc, edge_bin, edge_dec = model[7], model[8], model[9]

  start_time = time.time()
  
  for i, (batch, ctx_frames, filenames, masks, edges) in enumerate(eval_loader):

      with torch.no_grad():
          batch = batch.to(primary_device)
          masks = masks.to(primary_device) 
          edges = edges.to(primary_device) 

          # ---------------------------------------------------------
          # NEW: INDEPENDENT BASE LAYER EVALUATION
          # Simulate the LEO receiver decoding ONLY the semantic/edge stream
          # ---------------------------------------------------------
          rec_masks = sem_dec(sem_bin(sem_enc(masks)))
          rec_edges = edge_dec(edge_bin(edge_enc(edges)))

          m_iou, m_dice = calculate_iou_dice(rec_masks, masks)
          e_iou, e_dice = calculate_iou_dice(rec_edges, edges)

          all_mask_iou.append(m_iou)
          all_mask_dice.append(m_dice)
          all_edge_iou.append(e_iou)
          all_edge_dice.append(e_dice)

          # ---------------------------------------------------------
          # VISUAL ENHANCEMENT EVALUATION
          # ---------------------------------------------------------
          original, out_imgs, out_imgs_ee1, out_imgs_ee2, out_imgs_ee3, out_imgs_ee4, losses, code_batch = eval_forward(
                  model, (batch, ctx_frames, masks, edges), args)

          losses, msssim, psnr = finish_batch(
                  args, filenames, original, out_imgs,
                  losses, code_batch, output_suffix)

          psnr_ee1 = get_psnr(args, filenames, original, out_imgs_ee1)
          psnr_ee2 = get_psnr(args, filenames, original, out_imgs_ee2)
          psnr_ee3 = get_psnr(args, filenames, original, out_imgs_ee3)
          psnr_ee4 = get_psnr(args, filenames, original, out_imgs_ee4)

          # Calculate Semantic PSNR
          psnr_semantic = get_semantic_psnr(args, filenames, original, out_imgs, masks)

          all_losses += losses
          all_msssim += msssim
          all_psnr += psnr

          all_psnr_ee1 += psnr_ee1
          all_psnr_ee2 += psnr_ee2
          all_psnr_ee3 += psnr_ee3
          all_psnr_ee4 += psnr_ee4
          all_psnr_semantic += psnr_semantic 

      if i % 10 == 0:
        print('\tevaluating iter %d (%f seconds)... | Mask IoU: %.4f | Edge IoU: %.4f | Sem PSNR: %.2f' % (
          i, time.time() - start_time, np.mean(all_mask_iou), np.mean(all_edge_iou), np.mean(all_psnr_semantic)))

  # Return all metrics, appending the new Base Layer structural metrics at the end
  return (np.array(all_losses).mean(axis=0),
          np.array(all_msssim).mean(axis=0),
          np.array(all_psnr).mean(axis=0),
          np.array(all_psnr_ee1).mean(axis=0),
          np.array(all_psnr_ee2).mean(axis=0),
          np.array(all_psnr_ee3).mean(axis=0),
          np.array(all_psnr_ee4).mean(axis=0),
          np.array(all_psnr_semantic).mean(axis=0),
          # --- NEW: Base layer specific metrics ---
          np.mean(all_mask_iou),
          np.mean(all_mask_dice),
          np.mean(all_edge_iou),
          np.mean(all_edge_dice)
          )