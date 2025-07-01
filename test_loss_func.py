import os
import torch
import h5py
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from pathlib import Path

import imageio
# from simple_ocio import ToneMapper

import lpips

from PIL import Image

loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization


def convert_for_lpips(hdr_img, if_tone_mapping=False):
    
    if if_tone_mapping:
        # # tone-mapping 𝑐𝑙𝑎𝑚𝑝(log 𝐼/log 2, 0, 1) with torch tensor
        # ldr_img = torch.clamp(torch.log2(hdr_img), 0, 1)

        # tone-mapping 𝐼^(1/2.2) with torch tensor
        ldr_img = torch.pow(hdr_img, 1/2.2)
        imageio.v3.imwrite("tmp.exr", ldr_img.cpu().numpy().astype(np.float32))
    else:
        ldr_img = hdr_img
    
    ldr_img = ldr_img.unsqueeze(0).permute(0, 3, 1, 2)
    print(ldr_img.shape)
    return ldr_img

def compute_loss(pred_images, gt_images, loss_type='l1'):
    """Compute loss between predicted and ground truth images"""
    if loss_type == 'l1':
        return F.l1_loss(pred_images, gt_images)
    elif loss_type == 'l2':
        return F.mse_loss(pred_images, gt_images)
    elif loss_type == 'smooth_l1':
        return F.smooth_l1_loss(pred_images, gt_images)
    elif loss_type == 'lpips_alex':
        return loss_fn_alex(pred_images, gt_images)
    elif loss_type == 'lpips_vgg':
        return loss_fn_vgg(pred_images, gt_images)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

gt_images = None
h5_file = "F:/projects/renderformer/renderformertest/tmp/data/7.h5"
with h5py.File(h5_file, 'r') as f:
    triangles = torch.from_numpy(
        np.array(f['triangles']).astype(np.float32)
    )
    num_tris = triangles.shape[0]
    texture = torch.from_numpy(
        np.array(f['texture']).astype(np.float32)
    )
    mask = torch.ones(num_tris, dtype=torch.bool)
    vn = torch.from_numpy(np.array(f['vn']).astype(np.float32))
    c2w = torch.from_numpy(np.array(f['c2w']).astype(np.float32))
    fov = torch.from_numpy(np.array(f['fov']).astype(np.float32))
    
    # Load ground truth images if available
    if 'gt_img' in f:
        gt_images = torch.from_numpy(
            np.array(f['gt_img']).astype(np.float32)
        )
    else:
        gt_images = None

# pred_images_exr_file = "F:/projects/renderformer/renderformertest/output/testcubesubdivise/0_view_0.exr"
pred_images_exr_file = "F:/projects/renderformer/renderformertest/output/data/7_view_0.exr"
pred_images = torch.from_numpy(imageio.v3.imread(pred_images_exr_file).astype(np.float32))

# gt_images_exr_file = "F:/projects/renderformer/renderformertest/tmp/testcubesubdivise/0.exr"
gt_images_exr_file = "F:/projects/renderformer/renderformertest/tmp/data/7.exr"
gt_images = torch.from_numpy(imageio.v3.imread(gt_images_exr_file).astype(np.float32))[..., :3]

print(gt_images.shape, gt_images.min(), gt_images.max())
print(pred_images.shape, pred_images.min(), pred_images.max())

loss = compute_loss(pred_images, gt_images, 'l1')
print("l1", loss)

loss = compute_loss(pred_images, gt_images, 'l2')
print("l2", loss)

loss = compute_loss(convert_for_lpips(pred_images), convert_for_lpips(gt_images), 'lpips_vgg')
print("lpips_vgg", loss)

loss = compute_loss(convert_for_lpips(pred_images), convert_for_lpips(gt_images), 'lpips_alex')
print("lpips_alex", loss)

loss = compute_loss(convert_for_lpips(pred_images, if_tone_mapping=True), convert_for_lpips(gt_images, if_tone_mapping=True), 'lpips_vgg')
print("lpips_vgg with gamma encoding", loss)

loss = compute_loss(convert_for_lpips(pred_images, if_tone_mapping=True), convert_for_lpips(gt_images, if_tone_mapping=True), 'lpips_alex')
print("lpips_alex with gamma encoding", loss)