import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS

from dataset import get_loader
from evaluate import run_eval
from train_options import parser
from util import get_models, init_lstm, set_train, set_eval, init_d2
from util import prepare_inputs, forward_ctx

import network

import code

args = parser.parse_args()
print(args)

# --- HELPER TO ENSURE TENSORS ARE ON CORRECT DEVICE ---
def move_states_to_device(states, device):
    """Recursively moves tuples of hidden states to the target device."""
    if isinstance(states, torch.Tensor):
        return states.to(device)
    elif isinstance(states, (tuple, list)):
        return type(states)(move_states_to_device(s, device) for s in states)
    return states

# --- ENCAPSULATE THE RNN LOOP WITH SEMANTIC BASE LAYER ---
class CodecLoop(nn.Module):
    """
    Wraps the iterative compression loop into a single PyTorch Module.
    Includes Pass 0: Semantic Base Layer.
    """
    def __init__(self, encoder, binarizer, decoder, d2, 
                 semantic_encoder, semantic_binarizer, semantic_decoder, args):
        super().__init__()
        self.encoder = encoder
        self.binarizer = binarizer
        self.decoder = decoder
        self.d2 = d2
        
        # --- NEW: Semantic Network Components ---
        self.semantic_encoder = semantic_encoder
        self.semantic_binarizer = semantic_binarizer
        self.semantic_decoder = semantic_decoder
        
        self.args = args

    def forward(self, res, frame1, frame2, warped_unet_output1, warped_unet_output2, semantic_gt):
        batch_size, _, height, width = res.size()
        device = res.device
        in_img = res

        # ---------------------------------------------------------
        # PASS 0: SEMANTIC BASE LAYER
        # ---------------------------------------------------------
        # Encode the ground truth mask into its own bitstream
        sem_encoded = self.semantic_encoder(semantic_gt)
        semantic_codes = self.semantic_binarizer(sem_encoded)
        
        # Decode to reconstruct the semantics
        reconstructed_semantics = self.semantic_decoder(semantic_codes)
        
        # Calculate semantic reconstruction loss (L1 loss on the mask)
        semantic_loss = (semantic_gt - reconstructed_semantics).abs().mean(dim=[1, 2, 3])

        # ---------------------------------------------------------
        # PASS 1-N: VISUAL ENHANCEMENT LAYERS
        # ---------------------------------------------------------
        # Initialize hidden states LOCALLY on whichever GPU this chunk landed on
        lstm_states = init_lstm(batch_size, height, width, self.args)
        (encoder_h_1, encoder_h_2, encoder_h_3,
         decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4) = move_states_to_device(lstm_states, device)

        out_img = torch.zeros(1, 3, height, width, device=device) + 0.5
        b, d, h, w = batch_size, self.args.bits, height//16, width//16
        code_arr = [torch.zeros(b, d, h, w, device=device) for _ in range(self.args.iterations)]

        losses = []
        rec2_losses = []
        ee1_losses, ee2_losses, ee3_losses, ee4_losses = [], [], [], []

        for i in range(self.args.iterations):
            # --- MODIFIED: Inject reconstructed semantics into the visual encoder ---
            if self.args.v_compress and self.args.stack:
                encoder_input = torch.cat([frame1, res, frame2, reconstructed_semantics], dim=1)
            else:
                encoder_input = torch.cat([res, reconstructed_semantics], dim=1)

            encoded, encoder_h_1, encoder_h_2, encoder_h_3 = self.encoder(
                encoder_input, encoder_h_1, encoder_h_2, encoder_h_3,
                warped_unet_output1, warped_unet_output2)

            codes = self.binarizer(encoded)
            code_arr[i] = codes

            output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = self.decoder(
                codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4,
                warped_unet_output1, warped_unet_output2)

            res = res - output
            out_img = out_img + output.data
            
            losses.append(res.abs().mean(dim=[1, 2, 3]))

            d2_states = init_d2(batch_size, height, width, self.args)
            (d2_h_1, d2_h_2, d2_h_3, d2_h_4) = move_states_to_device(d2_states, device)

            codes_d2 = torch.stack(code_arr, dim=1).reshape(b, -1, h, w)
            
            (output_d2, out_ee1, out_ee2, out_ee3, out_ee4, d2_h_1, d2_h_2, d2_h_3, d2_h_4) = self.d2(
                    codes_d2, d2_h_1, d2_h_2, d2_h_3, d2_h_4,
                    warped_unet_output1, warped_unet_output2)

            rec2_losses.append((in_img - output_d2).abs().mean(dim=[1, 2, 3]))
            ee1_losses.append((in_img - out_ee1).abs().mean(dim=[1, 2, 3]))
            ee2_losses.append((in_img - out_ee2).abs().mean(dim=[1, 2, 3]))
            ee3_losses.append((in_img - out_ee3).abs().mean(dim=[1, 2, 3]))
            ee4_losses.append((in_img - out_ee4).abs().mean(dim=[1, 2, 3]))

        # Average over the iterations
        rec1_loss = torch.stack(losses).mean(dim=0)
        rec2_loss = torch.stack(rec2_losses).mean(dim=0)
        ee1_loss = torch.stack(ee1_losses).mean(dim=0)
        ee2_loss = torch.stack(ee2_losses).mean(dim=0)
        ee3_loss = torch.stack(ee3_losses).mean(dim=0)
        ee4_loss = torch.stack(ee4_losses).mean(dim=0)
        
        # --- NEW: Return semantic loss ---
        return rec1_loss, rec2_loss, ee1_loss, ee2_loss, ee3_loss, ee4_loss, semantic_loss

############### Data ###############
train_loader = get_loader(
  is_train=True,
  root=args.train, 
  mv_dir=args.train_mv,
  mask_dir=args.train_masks, # <-- NEW
  args=args
)

def get_eval_loaders():
  eval_loaders = {
    'VTL': get_loader(
        is_train=False,
        root=args.eval, 
        mv_dir=args.eval_mv,
        mask_dir=args.eval_masks, # <-- NEW
        args=args),
  }
  return eval_loaders

############### Model ###############
encoder, binarizer, decoder, unet = get_models(
  args=args, v_compress=args.v_compress,
  bits=args.bits,
  encoder_fuse_level=args.encoder_fuse_level,
  decoder_fuse_level=args.decoder_fuse_level)

# Setup Devices
gpus = [int(gpu) for gpu in args.gpus.split(',')]
primary_device = torch.device(f"cuda:{gpus[0]}" if len(gpus) > 0 and torch.cuda.is_available() else "cpu")

d2 = network.DecoderCell2(v_compress=args.v_compress, shrink=args.shrink,bits=args.bits,fuse_level=args.decoder_fuse_level).to(primary_device)

# --- NEW: Initialize Semantic Models ---
semantic_encoder = network.SemanticEncoder(in_channels=1).to(primary_device)
semantic_binarizer = network.Binarizer(bits=args.bits).to(primary_device) 
semantic_decoder = network.SemanticDecoder(out_channels=1, bits=args.bits).to(primary_device)

# --- DP INITIALIZATION ---
if len(gpus) > 1:
    print("Using GPUs {}.".format(gpus))
    if unet is not None:
        unet = nn.DataParallel(unet, device_ids=gpus)
    
    # Wrap the extended inner loop
    codec_loop = nn.DataParallel(
        CodecLoop(encoder, binarizer, decoder, d2, 
                  semantic_encoder, semantic_binarizer, semantic_decoder, args), 
        device_ids=gpus)
else:
    codec_loop = CodecLoop(encoder, binarizer, decoder, d2, 
                           semantic_encoder, semantic_binarizer, semantic_decoder, args).to(primary_device)

# We track the un-wrapped networks for the optimizer and saving checkpoints natively
nets = [encoder, binarizer, decoder, d2, semantic_encoder, semantic_binarizer, semantic_decoder]
if unet is not None:
    nets.append(unet)

print(nets)

params = [{'params': net.parameters()} for net in nets]

solver = optim.Adam(
    params,
    lr=args.lr)

milestones = [int(s) for s in args.schedule.split(',')]
scheduler = LS.CosineAnnealingLR(solver, T_max=args.max_train_iters, eta_min=1e-7)

if not os.path.exists(args.model_dir):
  print("Creating directory %s." % args.model_dir)
  os.makedirs(args.model_dir)

############### Checkpoints ###############
def resume(model_name, index):
  # --- MODIFIED: Added semantic names ---
  names = ['encoder', 'binarizer', 'decoder', 'd2', 'semantic_encoder', 'semantic_binarizer', 'semantic_decoder', 'unet']

  for net_idx, net in enumerate(nets):
    if net is not None:
      name = names[net_idx]
      checkpoint_path = '{}/{}_{}_{:08d}.pth'.format(
          args.model_dir, model_name,
          name, index)

      if os.path.exists(checkpoint_path):
          print('Loading %s from %s...' % (name, checkpoint_path))
          net.load_state_dict(torch.load(checkpoint_path))
      else:
          print('Skipping %s, checkpoint not found.' % name)

def save(index):
  # --- MODIFIED: Added semantic names ---
  names = ['encoder', 'binarizer', 'decoder', 'd2', 'semantic_encoder', 'semantic_binarizer', 'semantic_decoder', 'unet']

  for net_idx, net in enumerate(nets):
    if net is not None:
      state_dict = net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict()
      torch.save(state_dict,
                 '{}/{}_{}_{:08d}.pth'.format(
                   args.model_dir, args.save_model_name,
                   names[net_idx], index))

############### Training ###############

train_iter = 0
just_resumed = False
if args.load_model_name:
    print('Loading %s@iter %d' % (args.load_model_name,
                                  args.load_iter))

    resume(args.load_model_name, args.load_iter)
    train_iter = args.load_iter
    scheduler.last_epoch = train_iter - 1
    just_resumed = True


while True:

   # --- MODIFIED: Unpack masks from the dataloader ---
    for batch, (crops, ctx_frames, _, masks) in enumerate(train_loader):
        train_iter += 1

        if train_iter > args.max_train_iters:
          break

        batch_t0 = time.time()

        solver.zero_grad()

        # --- THE FIX: Use torch.cat to collapse the crops into the batch dimension ---
        if isinstance(masks, list) or isinstance(masks, tuple):
            masks = torch.cat(masks, dim=0).to(primary_device)
        else:
            masks = masks.to(primary_device)

        # Forward U-net
        if args.v_compress:
            unet_output1, unet_output2 = forward_ctx(unet, ctx_frames)
        else:
            unet_output1 = torch.zeros(args.batch_size).to(primary_device)
            unet_output2 = torch.zeros(args.batch_size).to(primary_device)

        res, frame1, frame2, warped_unet_output1, warped_unet_output2 = prepare_inputs(
            crops, args, unet_output1, unet_output2)

        bp_t0 = time.time()
        
        # --- MODIFIED: Pass masks to the codec loop ---
        (rec1_batch, rec2_batch, ee1_batch, 
         ee2_batch, ee3_batch, ee4_batch, semantic_batch) = codec_loop(
            res, frame1, frame2, tuple(warped_unet_output1), tuple(warped_unet_output2), masks
        )

        bp_t1 = time.time()

        # Average the loss vectors
        rec1_loss = rec1_batch.mean()
        rec2_loss = rec2_batch.mean()
        ee1_loss = ee1_batch.mean()
        ee2_loss = ee2_batch.mean()
        ee3_loss = ee3_batch.mean()
        ee4_loss = ee4_batch.mean()
        
        # --- NEW: Average Semantic Loss ---
        semantic_loss = semantic_batch.mean()

        # --- MODIFIED: Include semantic_loss in the backprop calculation ---
        # Note: You can adjust the weight (0.5) based on empirical results
        loss = (rec1_loss+rec2_loss)*0.5 + (ee1_loss+ee2_loss+ee3_loss+ee4_loss)*0.25 + (semantic_loss * 0.5)
        loss.backward()

        for net in nets:
            if net is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)

        solver.step()
        scheduler.step()

        batch_t1 = time.time()

        if train_iter % 10 == 0:
            print(
                '[TRAIN] Iter[{}]; LR: {}; Losses [Sem: {:.6f}; Rec1: {:.6f}; Rec2: {:.6f}; EE1: {:.6f}; EE2: {:.6f}]; Backprop: {:.4f} sec; Batch: {:.4f} sec'.
                format(train_iter,
                       scheduler.get_last_lr()[0],
                       semantic_loss.item(),
                       rec1_loss.item(),
                       rec2_loss.item(),
                       ee1_loss.item(),
                       ee2_loss.item(),
                       bp_t1 - bp_t0,
                       batch_t1 - batch_t0))

        if train_iter % args.checkpoint_iters == 0:
            save(train_iter)

        if just_resumed or train_iter % args.eval_iters == 0:
            print('Start evaluation...')
            
            # --- MODIFIED: Restored and updated the evaluation block ---
            set_eval(nets)

            eval_loaders = get_eval_loaders()
            for eval_name, eval_loader in eval_loaders.items():
                eval_begin = time.time()
                
                # Unpack all 8 returns from run_eval, including the new psnr_semantic
                eval_loss, mssim, psnr, psnr_ee1, psnr_ee2, psnr_ee3, psnr_ee4, psnr_semantic = run_eval(
                    nets, eval_loader, args, output_suffix='iter%d' % train_iter)

                print('Evaluation @iter %d done in %d secs' % (
                    train_iter, time.time() - eval_begin))
                print('%s Loss    : ' % eval_name
                      + '\t'.join(['%.5f' % el for el in eval_loss.tolist()]))
                print('%s MS-SSIM : ' % eval_name
                      + '\t'.join(['%.5f' % el for el in mssim.tolist()]))
                print('%s PSNR    : ' % eval_name
                      + '\t'.join(['%.5f' % el for el in psnr.tolist()]))

                print('%s EE1 PSNR: ' % eval_name
                      + '\t'.join(['%.5f' % el for el in psnr_ee1.tolist()]))
                print('%s EE2 PSNR: ' % eval_name
                      + '\t'.join(['%.5f' % el for el in psnr_ee2.tolist()]))
                print('%s EE3 PSNR: ' % eval_name
                      + '\t'.join(['%.5f' % el for el in psnr_ee3.tolist()]))
                print('%s EE4 PSNR: ' % eval_name
                      + '\t'.join(['%.5f' % el for el in psnr_ee4.tolist()]))
                      
                # Print the new Semantic PSNR metric
                print('%s Semantic PSNR: ' % eval_name
                      + '\t'.join(['%.5f' % el for el in psnr_semantic.tolist()]))

            set_train(nets)
            just_resumed = False

    if train_iter > args.max_train_iters:
      print('Training done.')
      break