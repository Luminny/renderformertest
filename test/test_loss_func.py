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

def convert_normal_for_lpips(img):
    # clip to [0, 1]
    img = torch.clamp(img, 0, 1)
    # normalize to [-1, 1]
    img = (img - 0.5) * 2
    img = img.unsqueeze(0).permute(0, 3, 1, 2)
    print(img.shape)
    return img

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
    elif loss_type == 'l1_w_lpips_alex':
        return F.l1_loss(pred_images, gt_images) + 0.05 * loss_fn_alex(convert_normal_for_lpips(pred_images), convert_normal_for_lpips(gt_images))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# pred_images_exr_file = "F:/projects/renderformer/renderformertest/output/testcubesubdivise/0_view_0.exr"
# pred_images_exr_file = "F:/projects/renderformer/renderformertest/output/data/7_view_0.exr"
pred_images_exr_file = "/home/luminyang/workspaces/renderformer/training/0701base/04/000074a334c541878360457c672b6c2e_00_view_0.exr"
pred_images = torch.from_numpy(imageio.v3.imread(pred_images_exr_file).astype(np.float32))

# gt_images_exr_file = "F:/projects/renderformer/renderformertest/tmp/testcubesubdivise/0.exr"
# gt_images_exr_file = "F:/projects/renderformer/renderformertest/tmp/data/7.exr"
gt_images_exr_file = "/home/luminyang/workspaces/renderformer/traindata/000-000/000-000/000074a334c541878360457c672b6c2e_00.exr"
gt_images = torch.from_numpy(imageio.v3.imread(gt_images_exr_file).astype(np.float32))[..., :3]

print(gt_images.shape, gt_images.min(), gt_images.max())
print(pred_images.shape, pred_images.min(), pred_images.max())

loss = compute_loss(pred_images, gt_images, 'l1')
print("l1", loss)

loss = compute_loss(pred_images, gt_images, 'l2')
print("l2", loss)

loss = compute_loss(convert_normal_for_lpips(pred_images), convert_normal_for_lpips(gt_images), 'lpips_vgg')
print("lpips_vgg", loss)

loss = compute_loss(convert_normal_for_lpips(pred_images), convert_normal_for_lpips(gt_images), 'lpips_alex')
print("lpips_alex", loss)

loss = compute_loss(pred_images, gt_images, 'l1_w_lpips_alex')
print("l1_w_lpips_alex", loss)