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

def move_states_to_device(states, device):
    """Recursively moves tuples of hidden states to the target device."""
    if isinstance(states, torch.Tensor):
        return states.to(device)
    elif isinstance(states, (tuple, list)):
        return type(states)(move_states_to_device(s, device) for s in states)
    return states

class CodecLoop(nn.Module):
    """
    Wraps the iterative compression loop into a single PyTorch Module.
    Includes Pass 0: Semantic Base Layer and Pass 1: Structural Edge Layer.
    """
    def __init__(self, encoder, binarizer, decoder, d2, 
                 semantic_encoder, semantic_binarizer, semantic_decoder,
                 edge_encoder, edge_binarizer, edge_decoder, args): # <-- NEW: Edge nets
        super().__init__()
        self.encoder = encoder
        self.binarizer = binarizer
        self.decoder = decoder
        self.d2 = d2
        
        self.semantic_encoder = semantic_encoder
        self.semantic_binarizer = semantic_binarizer
        self.semantic_decoder = semantic_decoder
        
        # --- NEW: Edge Network Components ---
        self.edge_encoder = edge_encoder
        self.edge_binarizer = edge_binarizer
        self.edge_decoder = edge_decoder
        
        self.args = args

    # --- MODIFIED: Added edge_gt to inputs ---
    def forward(self, res, frame1, frame2, warped_unet_output1, warped_unet_output2, semantic_gt, edge_gt):
        batch_size, _, height, width = res.size()
        device = res.device
        in_img = res

        # ---------------------------------------------------------
        # PASS 0: SEMANTIC BASE LAYER (Masks)
        # ---------------------------------------------------------
        sem_encoded = self.semantic_encoder(semantic_gt)
        semantic_codes = self.semantic_binarizer(sem_encoded)
        reconstructed_semantics = self.semantic_decoder(semantic_codes)
        semantic_loss = (semantic_gt - reconstructed_semantics).abs().mean(dim=[1, 2, 3])

        # ---------------------------------------------------------
        # PASS 1: STRUCTURAL ENHANCEMENT LAYER (Edges)
        # ---------------------------------------------------------
        edge_encoded = self.edge_encoder(edge_gt)
        edge_codes = self.edge_binarizer(edge_encoded)
        reconstructed_edges = self.edge_decoder(edge_codes)
        edge_loss = (edge_gt - reconstructed_edges).abs().mean(dim=[1, 2, 3])

        # ---------------------------------------------------------
        # PASS 2-N: VISUAL ENHANCEMENT LAYERS (Pixels)
        # ---------------------------------------------------------
        lstm_states = init_lstm(batch_size, height, width, self.args)
        (encoder_h_1, encoder_h_2, encoder_h_3,
         decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4) = move_states_to_device(lstm_states, device)

        out_img = torch.zeros(1, 3, height, width, device=device) + 0.5
        b, d, h, w = batch_size, self.args.bits, height//16, width//16
        code_arr = [torch.zeros(b, d, h, w, device=device) for _ in range(self.args.iterations)]

        losses, rec2_losses = [], []
        ee1_losses, ee2_losses, ee3_losses, ee4_losses = [], [], [], []

        for i in range(self.args.iterations):
            # --- MODIFIED: Inject BOTH semantics and edges into visual encoder ---
            if self.args.v_compress and self.args.stack:
                encoder_input = torch.cat([frame1, res, frame2, reconstructed_semantics, reconstructed_edges], dim=1)
            else:
                encoder_input = torch.cat([res, reconstructed_semantics, reconstructed_edges], dim=1)

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

        rec1_loss = torch.stack(losses).mean(dim=0)
        rec2_loss = torch.stack(rec2_losses).mean(dim=0)
        ee1_loss = torch.stack(ee1_losses).mean(dim=0)
        ee2_loss = torch.stack(ee2_losses).mean(dim=0)
        ee3_loss = torch.stack(ee3_losses).mean(dim=0)
        ee4_loss = torch.stack(ee4_losses).mean(dim=0)
        
        # --- MODIFIED: Return edge_loss ---
        return rec1_loss, rec2_loss, ee1_loss, ee2_loss, ee3_loss, ee4_loss, semantic_loss, edge_loss

############### Data ###############
train_loader = get_loader(
  is_train=True,
  root=args.train, 
  mv_dir=args.train_mv,
  mask_dir=args.train_masks, 
  edge_dir=args.train_edges, # <-- NEW
  args=args
)

def get_eval_loaders():
  eval_loaders = {
    'VTL': get_loader(
        is_train=False,
        root=args.eval, 
        mv_dir=args.eval_mv,
        mask_dir=args.eval_masks,
        edge_dir=args.eval_edges, # <-- NEW
        args=args),
  }
  return eval_loaders

############### Model ###############
encoder, binarizer, decoder, unet = get_models(
  args=args, v_compress=args.v_compress,
  bits=args.bits,
  encoder_fuse_level=args.encoder_fuse_level,
  decoder_fuse_level=args.decoder_fuse_level)

gpus = [int(gpu) for gpu in args.gpus.split(',')]
primary_device = torch.device(f"cuda:{gpus[0]}" if len(gpus) > 0 and torch.cuda.is_available() else "cpu")

d2 = network.DecoderCell2(v_compress=args.v_compress, shrink=args.shrink,bits=args.bits,fuse_level=args.decoder_fuse_level).to(primary_device)

semantic_encoder = network.SemanticEncoder(in_channels=1).to(primary_device)
semantic_binarizer = network.Binarizer(bits=args.bits).to(primary_device) 
semantic_decoder = network.SemanticDecoder(out_channels=1, bits=args.bits).to(primary_device)

# --- NEW: Initialize Edge Models ---
edge_encoder = network.SemanticEncoder(in_channels=1).to(primary_device)
edge_binarizer = network.Binarizer(bits=args.bits).to(primary_device) 
edge_decoder = network.SemanticDecoder(out_channels=1, bits=args.bits).to(primary_device)

if len(gpus) > 1:
    print("Using GPUs {}.".format(gpus))
    if unet is not None:
        unet = nn.DataParallel(unet, device_ids=gpus)
    
    codec_loop = nn.DataParallel(
        CodecLoop(encoder, binarizer, decoder, d2, 
                  semantic_encoder, semantic_binarizer, semantic_decoder, 
                  edge_encoder, edge_binarizer, edge_decoder, args), # <-- NEW
        device_ids=gpus)
else:
    codec_loop = CodecLoop(encoder, binarizer, decoder, d2, 
                           semantic_encoder, semantic_binarizer, semantic_decoder, 
                           edge_encoder, edge_binarizer, edge_decoder, args).to(primary_device)

# --- MODIFIED: Added edge nets ---
nets = [encoder, binarizer, decoder, d2, semantic_encoder, semantic_binarizer, semantic_decoder, edge_encoder, edge_binarizer, edge_decoder]
if unet is not None:
    nets.append(unet)

params = [{'params': net.parameters()} for net in nets]

solver = optim.Adam(params, lr=args.lr)

milestones = [int(s) for s in args.schedule.split(',')]
scheduler = LS.CosineAnnealingLR(solver, T_max=args.max_train_iters, eta_min=1e-7)

if not os.path.exists(args.model_dir):
  os.makedirs(args.model_dir)

############### Checkpoints ###############
def resume(model_name, index):
  # --- MODIFIED ---
  names = ['encoder', 'binarizer', 'decoder', 'd2', 'semantic_encoder', 'semantic_binarizer', 'semantic_decoder', 'edge_encoder', 'edge_binarizer', 'edge_decoder', 'unet']

  for net_idx, net in enumerate(nets):
    if net is not None:
      name = names[net_idx]
      checkpoint_path = '{}/{}_{}_{:08d}.pth'.format(args.model_dir, model_name, name, index)
      if os.path.exists(checkpoint_path):
          print('Loading %s from %s...' % (name, checkpoint_path))
          net.load_state_dict(torch.load(checkpoint_path))
      else:
          print('Skipping %s, checkpoint not found.' % name)

def save(index):
  # --- MODIFIED ---
  names = ['encoder', 'binarizer', 'decoder', 'd2', 'semantic_encoder', 'semantic_binarizer', 'semantic_decoder', 'edge_encoder', 'edge_binarizer', 'edge_decoder', 'unet']

  for net_idx, net in enumerate(nets):
    if net is not None:
      state_dict = net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict()
      torch.save(state_dict, '{}/{}_{}_{:08d}.pth'.format(args.model_dir, args.save_model_name, names[net_idx], index))

############### Training ###############

train_iter = 0
just_resumed = False
if args.load_model_name:
    resume(args.load_model_name, args.load_iter)
    train_iter = args.load_iter
    scheduler.last_epoch = train_iter - 1
    just_resumed = True


while True:
    # --- MODIFIED: Unpack edges from dataloader ---
    for batch, (crops, ctx_frames, _, masks, edges) in enumerate(train_loader):
        train_iter += 1

        if train_iter > args.max_train_iters: break

        batch_t0 = time.time()
        solver.zero_grad()

        if isinstance(masks, (list, tuple)):
            masks = torch.cat(masks, dim=0).to(primary_device)
            edges = torch.cat(edges, dim=0).to(primary_device) # <-- NEW
        else:
            masks = masks.to(primary_device)
            edges = edges.to(primary_device) # <-- NEW

        if args.v_compress:
            unet_output1, unet_output2 = forward_ctx(unet, ctx_frames)
        else:
            unet_output1, unet_output2 = torch.zeros(args.batch_size).to(primary_device), torch.zeros(args.batch_size).to(primary_device)

        res, frame1, frame2, warped_unet_output1, warped_unet_output2 = prepare_inputs(crops, args, unet_output1, unet_output2)

        bp_t0 = time.time()
        
        # --- MODIFIED: Pass edges to codec loop, unpack edge_batch ---
        (rec1_batch, rec2_batch, ee1_batch, 
         ee2_batch, ee3_batch, ee4_batch, semantic_batch, edge_batch) = codec_loop(
            res, frame1, frame2, tuple(warped_unet_output1), tuple(warped_unet_output2), masks, edges
        )

        bp_t1 = time.time()

        rec1_loss = rec1_batch.mean()
        rec2_loss = rec2_batch.mean()
        ee1_loss = ee1_batch.mean()
        ee2_loss = ee2_batch.mean()
        ee3_loss = ee3_batch.mean()
        ee4_loss = ee4_batch.mean()
        semantic_loss = semantic_batch.mean()
        edge_loss = edge_batch.mean() # <-- NEW

        # --- MODIFIED: Include edge_loss in backprop ---
        loss = (rec1_loss+rec2_loss)*0.5 + (ee1_loss+ee2_loss+ee3_loss+ee4_loss)*0.25 + (semantic_loss * 0.5) + (edge_loss * 0.5)
        loss.backward()

        for net in nets:
            if net is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)

        solver.step()
        scheduler.step()

        batch_t1 = time.time()

        if train_iter % 10 == 0:
            print(
                '[TRAIN] Iter[{}]; LR: {}; Losses [Sem: {:.4f}; Edge: {:.4f}; Rec1: {:.4f}; Rec2: {:.4f}; EE1: {:.4f}]; Batch: {:.4f} s'.
                format(train_iter, scheduler.get_last_lr()[0], semantic_loss.item(), edge_loss.item(), rec1_loss.item(), rec2_loss.item(), ee1_loss.item(), batch_t1 - batch_t0))

        if train_iter % args.checkpoint_iters == 0: save(train_iter)

        if just_resumed or train_iter % args.eval_iters == 0:
            # ... (Eval block remains similar, ensure run_eval returns match if you modified it)
            set_eval(nets)
            eval_loaders = get_eval_loaders()
            for eval_name, eval_loader in eval_loaders.items():
                run_eval(nets, eval_loader, args, output_suffix='iter%d' % train_iter)
            set_train(nets)
            just_resumed = False

    if train_iter > args.max_train_iters:
      print('Training done.')
      break