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


def compute_loss(pred_images, gt_images, loss_type='l1'):
    """Compute loss between predicted and ground truth images"""
    if loss_type == 'l1':
        return F.l1_loss(pred_images, gt_images)
    elif loss_type == 'l2':
        return F.mse_loss(pred_images, gt_images)
    elif loss_type == 'smooth_l1':
        return F.smooth_l1_loss(pred_images, gt_images)
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

pred_images_exr_file = "F:/projects/renderformer/renderformertest/output/testcubesubdivise/0_view_0.exr"
pred_images = torch.from_numpy(imageio.v3.imread(pred_images_exr_file).astype(np.float32))

gt_images_exr_file = "F:/projects/renderformer/renderformertest/tmp/testcubesubdivise/0.exr"
gt_images = torch.from_numpy(imageio.v3.imread(gt_images_exr_file).astype(np.float32))[..., :3]

print(gt_images.shape)
print(pred_images.shape)

loss = compute_loss(pred_images, gt_images)
print(loss)
