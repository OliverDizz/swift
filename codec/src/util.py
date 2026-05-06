from collections import namedtuple
import cv2
import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.nn as nn

import network
from metric import msssim, psnr
from unet import UNet

def get_models(args, v_compress, bits, encoder_fuse_level, decoder_fuse_level, device='cuda'):
    encoder = network.EncoderCell(
        v_compress=v_compress,
        stack=args.stack,
        fuse_encoder=args.fuse_encoder,
        fuse_level=encoder_fuse_level
    ).to(device)

    binarizer = network.Binarizer(bits).to(device)

    decoder = network.DecoderCell(
        v_compress=v_compress, shrink=args.shrink,
        bits=bits,
        fuse_level=decoder_fuse_level
    ).to(device)

    if v_compress:
        unet = UNet(3, args.shrink).to(device)
    else:
        unet = None

    return encoder, binarizer, decoder, unet

def get_identity_grid(size, device='cuda'):
    # Replaced hardcoded .cuda() with dynamic device
    id_mat = torch.FloatTensor([[1, 0, 0, 0, 1, 0]] * size[0]).view(-1, 2, 3).to(device)
    return F.affine_grid(id_mat, size, align_corners=True)

def transpose_to_grid(frame2):
    return frame2.permute(0, 2, 3, 1) # More efficient than multiple transposes

def get_id_grids(size, device='cuda'):
    batch_size, _, height, width = size
    id_grid_4 = get_identity_grid(torch.Size([batch_size, 32, height//2, width//2]), device)
    id_grid_3 = get_identity_grid(torch.Size([batch_size, 32, height//4, width//4]), device)
    id_grid_2 = get_identity_grid(torch.Size([batch_size, 32, height//8, width//8]), device)
    return id_grid_4, id_grid_3, id_grid_2

def get_large_id_grid(size, device='cuda'):
    batch_size, _, height, width = size
    return get_identity_grid(torch.Size([batch_size, 32, height, width]), device)

down_sample = nn.AvgPool2d(2, stride=2)

def get_flows(flow):
    flow_4 = down_sample(flow)
    flow_3 = down_sample(flow_4)
    flow_2 = down_sample(flow_3)

    flow_4 = transpose_to_grid(flow_4)
    flow_3 = transpose_to_grid(flow_3)
    flow_2 = transpose_to_grid(flow_2)

    final_grid_4 = flow_4 + 0.5
    final_grid_3 = flow_3 + 0.5
    final_grid_2 = flow_2 + 0.5

    return [final_grid_4, final_grid_3, final_grid_2]

def prepare_batch(batch, v_compress, warp):
    res = batch - 0.5
    flows = []
    frame1, frame2 = None, None
    if v_compress:
        if warp:
            assert res.size(1) == 13
            flow_1 = res[:, 9:11]
            flow_2 = res[:, 11:13]
            flows.append(get_flows(flow_1))
            flows.append(get_flows(flow_2))

        frame1 = res[:, :3]
        frame2 = res[:, 6:9]
        res = res[:, 3:6]
    return res, frame1, frame2, flows

def set_eval(models):
    for m in models:
        if m is not None:
            m.eval()

def set_train(models):
    for m in models:
        if m is not None:
            m.train()

def prepare_inputs(crops, args, unet_output1, unet_output2, device='cuda'):
    data_arr, frame1_arr, frame2_arr = [], [], []
    warped_unet_output1, warped_unet_output2 = [], []

    for crop_idx, data in enumerate(crops):
        patches = data.to(device)

        res, frame1, frame2, flows = prepare_batch(patches, args.v_compress, args.warp)
        data_arr.append(res)
        frame1_arr.append(frame1)
        frame2_arr.append(frame2)

        if args.v_compress and args.warp:
            wuo1, wuo2 = warp_unet_outputs(flows, unet_output1, unet_output2)
            warped_unet_output1.append(wuo1)
            warped_unet_output2.append(wuo2)

    res = torch.cat(data_arr, dim=0)
    frame1 = torch.cat(frame1_arr, dim=0)
    frame2 = torch.cat(frame2_arr, dim=0)
    warped_unet_output1 = [torch.cat(wuos, dim=0) for wuos in zip(*warped_unet_output1)]
    warped_unet_output2 = [torch.cat(wuos, dim=0) for wuos in zip(*warped_unet_output2)]

    return res, frame1, frame2, warped_unet_output1, warped_unet_output2

def forward_ctx(unet, ctx_frames, device='cuda'):
    ctx_frames = ctx_frames.to(device) - 0.5
    frame1, frame2 = ctx_frames[:, :3], ctx_frames[:, 3:]
    unet_output1, unet_output2 = [], []

    unet_outputs = unet(torch.cat([frame1, frame2], dim=0))
    for u_out in unet_outputs:
        u_out1, u_out2 = u_out.chunk(2, dim=0)
        unet_output1.append(u_out1)
        unet_output2.append(u_out2)

    return unet_output1, unet_output2

def warp_unet_outputs(flows, unet_output1, unet_output2):
    [grid_1_4, grid_1_3, grid_1_2] = flows[0]
    [grid_2_4, grid_2_3, grid_2_2] = flows[1]
    warped_unet_output1, warped_unet_output2 = [], []

    warped_unet_output1.append(F.grid_sample(unet_output1[0], grid_1_2, padding_mode='border', align_corners=True))
    warped_unet_output2.append(F.grid_sample(unet_output2[0], grid_2_2, padding_mode='border', align_corners=True))

    warped_unet_output1.append(F.grid_sample(unet_output1[1], grid_1_3, padding_mode='border', align_corners=True))
    warped_unet_output2.append(F.grid_sample(unet_output2[1], grid_2_3, padding_mode='border', align_corners=True))

    warped_unet_output1.append(F.grid_sample(unet_output1[2], grid_1_4, padding_mode='border', align_corners=True))
    warped_unet_output2.append(F.grid_sample(unet_output2[2], grid_2_4, padding_mode='border', align_corners=True))

    return warped_unet_output1, warped_unet_output2

def init_lstm(batch_size, height, width, args, device='cuda'):
    def make_hidden(channels, h, w):
        return (torch.zeros(batch_size, channels, h, w).to(device),
                torch.zeros(batch_size, channels, h, w).to(device))

    h1 = make_hidden(256, height // 4, width // 4)
    h2 = make_hidden(512, height // 8, width // 8)
    h3 = make_hidden(512, height // 16, width // 16)
    
    d1 = make_hidden(512, height // 16, width // 16)
    d2 = make_hidden(512, height // 8, width // 8)
    d3 = make_hidden(256, height // 4, width // 4)
    d4 = make_hidden(128, height // 2, width // 2)

    return (h1, h2, h3, d1, d2, d3, d4)

def init_d2(batch_size, height, width, args, device='cuda'):
    def make_hidden(channels, h, w):
        return (torch.zeros(batch_size, channels, h, w).to(device),
                torch.zeros(batch_size, channels, h, w).to(device))

    return (make_hidden(512, height // 16, width // 16),
            make_hidden(512, height // 8, width // 8),
            make_hidden(256, height // 4, width // 4),
            make_hidden(128, height // 2, width // 2))

def save_numpy_array_as_image(filename, arr):
    img = np.squeeze(arr * 255.0).astype(np.uint8).transpose(1, 2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)

# --- MODIFIED: Adjusted signature to accept inputs tuple including edges ---
def eval_forward(model, inputs, args):
    batch, ctx_frames, masks, edges = inputs # <-- NEW: Extract edges
    cooked_batch = prepare_batch(
        batch, args.v_compress, args.warp)

    with torch.no_grad():
        return forward_model(
            model=model,
            cooked_batch=cooked_batch,
            ctx_frames=ctx_frames,
            masks=masks,
            edges=edges, # <-- NEW: Pass edges down to the network
            args=args,
            v_compress=args.v_compress,
            iterations=args.iterations,
            encoder_fuse_level=args.encoder_fuse_level,
            decoder_fuse_level=args.decoder_fuse_level)

def prepare_unet_output(unet, unet_input, flows, warp):
    unet_output1, unet_output2 = [], []
    
    unet_outputs = unet(unet_input)
    
    for u_out in unet_outputs:
        u_out1, u_out2 = u_out.chunk(2, dim=0)
        unet_output1.append(u_out1)
        unet_output2.append(u_out2)
        
    if warp:
        unet_output1, unet_output2 = warp_unet_outputs(
            flows, unet_output1, unet_output2)
            
    return unet_output1, unet_output2

# --- MODIFIED: Added `edges` parameter and Pass 1 logic ---
def forward_model(model, cooked_batch, ctx_frames, masks, edges, args, v_compress,
                  iterations, encoder_fuse_level, decoder_fuse_level):
    
    # Unpack visual, semantic, and new edge models
    encoder, binarizer, decoder, d2, semantic_encoder, semantic_binarizer, semantic_decoder = model[:7]
    edge_encoder, edge_binarizer, edge_decoder = model[7:10] # <-- NEW: Extract Edge Nets
    unet = model[10] if len(model) > 10 else None # <-- MODIFIED: Shifted index
    
    res, _, _, flows = cooked_batch
    in_img = res

    # Infer device dynamically to support DataParallel evaluations safely
    device = res.device

    ctx_frames = ctx_frames.to(device) - 0.5
    frame1 = ctx_frames[:, :3]
    frame2 = ctx_frames[:, 3:]
    
    # Ensure masks and edges are on the correct device
    masks = masks.to(device)
    edges = edges.to(device) # <-- NEW: Device mapping

    # ---------------------------------------------------------
    # PASS 0: SEMANTIC BASE LAYER (Masks)
    # ---------------------------------------------------------
    sem_encoded = semantic_encoder(masks)
    sem_codes = semantic_binarizer(sem_encoded)
    reconstructed_semantics = semantic_decoder(sem_codes)

    # ---------------------------------------------------------
    # PASS 1: STRUCTURAL ENHANCEMENT LAYER (Edges)
    # ---------------------------------------------------------
    edge_encoded = edge_encoder(edges) # <-- NEW
    edge_codes = edge_binarizer(edge_encoded) # <-- NEW
    reconstructed_edges = edge_decoder(edge_codes) # <-- NEW

    batch_size, _, height, width = res.size()
    
    (encoder_h_1, encoder_h_2, encoder_h_3,
     decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4) = init_lstm(batch_size,
                                                                      height,
                                                                      width,
                                                                      args,
                                                                      device=device)

    original = res.data.cpu().numpy() + 0.5

    # Note: Intentionally kept on CPU (no .to(device)) to match original behavior. 
    # Adding CPU output tensors to this avoids memory bloat.
    out_img = torch.zeros(1, 3, height, width) + 0.5
    out_imgs = []
    out_imgs_ee1, out_imgs_ee2, out_imgs_ee3, out_imgs_ee4 = [], [], [], []
    losses = []

    enc_unet_output1, enc_unet_output2 = [], []
    dec_unet_output1, dec_unet_output2 = [], []
    
    if v_compress:
        dec_unet_output1, dec_unet_output2 = prepare_unet_output(
            unet, torch.cat([frame1, frame2], dim=0), flows, warp=args.warp)

        # Make explicitly mutable list copies if modifying elements below
        enc_unet_output1 = list(dec_unet_output1)
        enc_unet_output2 = list(dec_unet_output2)

        for jj in range(3 - max(encoder_fuse_level, decoder_fuse_level)):
            enc_unet_output1[jj] = None
            enc_unet_output2[jj] = None
            dec_unet_output1[jj] = None
            dec_unet_output2[jj] = None

    codes = []
    b, d, h, w = batch_size, args.bits, height // 16, width // 16
    code_arr = [torch.zeros(b, d, h, w).to(device) for _ in range(args.iterations)]

    # ---------------------------------------------------------
    # PASS 2-N: VISUAL ITERATIONS
    # ---------------------------------------------------------
    for i in range(iterations):
        
        # --- MODIFIED: Concatenate BOTH semantics and edges to encoder input ---
        if args.v_compress and args.stack:
            encoder_input = torch.cat([frame1, res, frame2, reconstructed_semantics, reconstructed_edges], dim=1) # <-- NEW: reconstructed_edges
        else:
            encoder_input = torch.cat([res, reconstructed_semantics, reconstructed_edges], dim=1) # <-- NEW: reconstructed_edges

        # FIX: Cast list arguments to tuples to satisfy DataParallel scatter requirements
        encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
            encoder_input, encoder_h_1, encoder_h_2, encoder_h_3,
            tuple(enc_unet_output1), tuple(enc_unet_output2))

        code = binarizer(encoded)
        code_arr[i] = code

        # FIX: Cast list arguments to tuples
        output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
            code, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4,
            tuple(dec_unet_output1), tuple(dec_unet_output2))

        res = res - output
        
        (d2_h_1, d2_h_2, d2_h_3, d2_h_4) = init_d2(batch_size, height, width, args, device=device)
        code_d2 = torch.stack(code_arr, dim=1).reshape(b, -1, h, w)

        # FIX: Cast list arguments to tuples
        (output_d2, out_ee1, out_ee2, out_ee3, out_ee4, d2_h_1, d2_h_2, d2_h_3, d2_h_4) = d2(
                code_d2, d2_h_1, d2_h_2, d2_h_3, d2_h_4,
                tuple(dec_unet_output1), tuple(dec_unet_output2))
        
        
        codes.append(code.data.cpu().numpy())

        out_img_new = out_img + output_d2.data.cpu()
        out_img_np = out_img_new.numpy().clip(0, 1)
        out_imgs.append(out_img_np)
        
        losses.append(float((in_img - output_d2).abs().mean().item()))

        out_imgs_ee1.append((out_img + out_ee1.data.cpu()).numpy().clip(0, 1))
        out_imgs_ee2.append((out_img + out_ee2.data.cpu()).numpy().clip(0, 1))
        out_imgs_ee3.append((out_img + out_ee3.data.cpu()).numpy().clip(0, 1))
        out_imgs_ee4.append((out_img + out_ee4.data.cpu()).numpy().clip(0, 1))

    return original, np.array(out_imgs), np.array(out_imgs_ee1), np.array(out_imgs_ee2), \
           np.array(out_imgs_ee3), np.array(out_imgs_ee4), np.array(losses), np.array(codes)

def evaluate(original, out_imgs):
    ms_ssims = np.array([get_ms_ssim(original, out_img) for out_img in out_imgs])
    psnrs    = np.array([   get_psnr(original, out_img) for out_img in out_imgs])

    return ms_ssims, psnrs

def evaluate_psnr(original, out_imgs):
    psnrs    = np.array([   get_psnr(original, out_img) for out_img in out_imgs])
    return psnrs

def evaluate_all(original, out_imgs):
    all_msssim, all_psnr = [], []
    for j in range(original.shape[0]):
        msssim, psnr = evaluate(
            original[None, j],
            [out_img[None, j] for out_img in out_imgs])
        all_msssim.append(msssim)
        all_psnr.append(psnr)

    return all_msssim, all_psnr

def as_img_array(image):
    image = image.clip(0, 1) * 255.0
    return image.astype(np.uint8).transpose(0, 2, 3, 1)

def get_ms_ssim(original, compared):
    return msssim(as_img_array(original), as_img_array(compared))

def get_psnr(original, compared):
    return psnr(as_img_array(original), as_img_array(compared))

def save_torch_array_as_image(filename, arr):
    img = np.squeeze(arr.cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
    if len(img.shape) == 3: 
        img = img.transpose(1, 2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)