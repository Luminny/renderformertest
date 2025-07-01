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
import json

from renderformer import GeoRasterRenderingPipeline
from renderformer.models.config import RenderFormerConfig
from renderformer.models.geo_raster import GeoRaster
from train_geo_raster import RenderFormerDataset


def test_train_dataset(pipeline, dataloader, device, config):
    """Train for one epoch"""
    pipeline.model.train()
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, data in enumerate(progress_bar):
        # Move data to device
        triangles = data['triangles'].to(device)
        mask = data['mask'].to(device)
        vn = data['vn'].to(device)
        c2w = data['c2w'].to(device)
        fov = data['fov'].to(device)
        gt_images = (data['gt_img'].to(device) 
                    if data['gt_img'] is not None else None)
        
        rendered_imgs = pipeline(
            triangles=triangles,
            mask=mask,
            vn=vn,
            c2w=c2w,
            fov=fov,
            resolution=config.resolution,
            torch_dtype=torch.float16,
        )
        
        print("Successfully rendered")
        break



def main():
    parser = argparse.ArgumentParser(description="Train RenderFormer model")
    
    # Data arguments
    parser.add_argument("--train_data_dir", type=str, required=True, 
                       help="Directory containing training H5 files")
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, 
                       help="Number of data loader workers")
    
    # Model arguments
    parser.add_argument("--resolution", type=int, default=256, 
                       help="Resolution for training")
    parser.add_argument("--model_config", type=str, default="F:/projects/renderformer/traindata_test/model/config.json", 
                       help="Model config file")
    
    args = parser.parse_args()
    
    # Setup device
    device = (torch.device('cuda') if torch.cuda.is_available() 
              else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Init model
    print(f"Creating new GeoRaster model from scratch...")
    model_config = RenderFormerConfig.from_json(args.model_config)
    
    # config = RenderFormerConfig(norm_first=True)
    pipeline = GeoRasterRenderingPipeline(GeoRaster(model_config))
    print("✓ New model created successfully")
    
    pipeline.to(device)
    
    # Create datasets and dataloaders
    train_dataset = RenderFormerDataset(args.train_data_dir, args.resolution)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_train_dataset(pipeline, train_dataloader, device, args)


if __name__ == '__main__':
    main() 