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

from renderformer import GeoRasterRenderingPipeline
from renderformer.models.config import RenderFormerConfig
from renderformer.models.geo_raster import GeoRaster


class RenderFormerDataset(Dataset):
    def __init__(self, data_dir, resolution=256, max_num_tris=2048):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.max_num_tris = max_num_tris
        self.h5_files = list(self.data_dir.glob("*.h5"))
        
        if len(self.h5_files) == 0:
            raise ValueError(f"No H5 files found in {data_dir}")
        
        print(f"Found {len(self.h5_files)} H5 files in {data_dir}")
    
    def __len__(self):
        return len(self.h5_files)
    
    def __getitem__(self, idx):
        h5_file = self.h5_files[idx]
        
        # triangles: [num_tris, 3, 3]
        # mask: [num_tris]
        # vn: [num_tris, 3, 3]
        # c2w: [num_views, 4, 4]
        # fov: [num_views, 1]
        # gt_img: [num_views, H, W, 3]
        # todo: num_views is not always 1, need to handle this
        
        with h5py.File(h5_file, 'r') as f:
            triangles = torch.from_numpy(
                np.array(f['triangles']).astype(np.float32)
            )
            num_tris = triangles.shape[0]
            vn = torch.from_numpy(np.array(f['vn']).astype(np.float32))
            c2w = torch.from_numpy(np.array(f['c2w']).astype(np.float32))
            fov = torch.from_numpy(
                np.array(f['fov']).astype(np.float32)
            ).unsqueeze(0)

        # Pad triangles to max_num_tris
        if num_tris < self.max_num_tris:
            # Create padding for triangles [max_num_tris - num_tris, 3, 3]
            triangles_padding = torch.zeros(
                self.max_num_tris - num_tris, 3, 3, dtype=triangles.dtype
            )
            triangles = torch.cat([triangles, triangles_padding], dim=0)
            
            # Pad vn to max_num_tris
            vn_padding = torch.zeros(
                self.max_num_tris - num_tris, 3, 3, dtype=vn.dtype
            )
            vn = torch.cat([vn, vn_padding], dim=0)
            
            # Create mask: True for real triangles, False for padding
            mask = torch.cat([
                torch.ones(num_tris, dtype=torch.bool),
                torch.zeros(self.max_num_tris - num_tris, dtype=torch.bool)
            ])
        elif num_tris > self.max_num_tris:
            # raise ValueError(f"num_tris > max_num_tris: {num_tris} > {self.max_num_tris} with file {h5_file}")
            # Truncate if too many triangles
            triangles = triangles[:self.max_num_tris]
            vn = vn[:self.max_num_tris]
            mask = torch.ones(self.max_num_tris, dtype=torch.bool)
        else:
            # Exact size, no padding needed
            mask = torch.ones(self.max_num_tris, dtype=torch.bool)

        gt_images_exr_file = str(h5_file).replace('.h5', '.exr')
        gt_images = torch.from_numpy(
            imageio.v3.imread(gt_images_exr_file).astype(np.float32)[..., :3]
        ).unsqueeze(0)

        data = {
            'triangles': triangles,  # Now [max_num_tris, 3, 3]
            'mask': mask,           # Now [max_num_tris]
            'c2w': c2w,
            'fov': fov,
            'vn': vn,              # Now [max_num_tris, 3, 3]
            'gt_img': gt_images,
            'file_path': str(h5_file)
        }
        return data


def compute_loss(pred_images, gt_images, loss_type='l1'):
    """Compute loss between predicted and ground truth images"""
    if loss_type == 'l1':
        # print(f"pred_images: {pred_images.shape}")
        # print(f"gt_images: {gt_images.shape}")  
        return F.l1_loss(pred_images, gt_images)
    elif loss_type == 'l2':
        return F.mse_loss(pred_images, gt_images)
    elif loss_type == 'smooth_l1':
        return F.smooth_l1_loss(pred_images, gt_images)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_epoch(model, dataloader, optimizer, scheduler, device, config):
    """Train for one epoch"""
    model.model.train()
    total_loss = 0.0
    num_batches = 0
    
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
        
        try:
            # Set precision dtype
            if config.precision == 'fp16':
                torch_dtype = torch.float16
            elif config.precision == 'bf16':
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
                
            rendered_imgs = model(
                triangles=triangles,
                mask=mask,
                vn=vn,
                c2w=c2w,
                fov=fov,
                resolution=config.resolution,
                torch_dtype=torch_dtype,
            )
            
            # Compute loss if ground truth is available
            if gt_images is not None:
                loss = compute_loss(rendered_imgs, gt_images, 
                                  config.loss_type)
            else:
                print(f"gt_images is None")
            
            optimizer.zero_grad()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), 
                                             config.grad_clip)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log to wandb
            if config.use_wandb:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': scheduler.get_last_lr()[0],
                    'batch': batch_idx
                })
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU OOM in batch {batch_idx}, skipping...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    scheduler.step()
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, dataloader, device, config):
    """Validate the model"""
    model.model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Validation"):
            # Move data to device
            triangles = data['triangles'].to(device)
            texture = data['texture'].to(device)
            mask = data['mask'].to(device)
            vn = data['vn'].to(device)
            c2w = data['c2w'].to(device)
            fov = data['fov'].to(device)
            gt_images = (data['gt_img'].to(device) 
                        if data['gt_img'] is not None else None)
            
            try:
                # Set precision dtype
                if config.precision == 'fp16':
                    torch_dtype = torch.float16
                elif config.precision == 'bf16':
                    torch_dtype = torch.bfloat16
                else:
                    torch_dtype = torch.float32
                    
                rendered_imgs = model(
                    triangles=triangles,
                    texture=texture,
                    mask=mask,
                    vn=vn,
                    c2w=c2w,
                    fov=fov,
                    resolution=config.resolution,
                    torch_dtype=torch_dtype,
                )
                
                if gt_images is not None:
                    loss = compute_loss(rendered_imgs, gt_images, config.loss_type)
                    total_loss += loss.item()
                    num_batches += 1
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"GPU OOM in validation batch, skipping...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path):
    """Save model checkpoint in Hugging Face format"""
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 保存模型权重
    model.model.save_pretrained(save_path)
    
    # 保存训练状态（优化器、调度器等）
    training_state = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    
    # # 保存训练状态到 JSON 文件
    # import json
    # with open(os.path.join(save_path, 'training_state.json'), 'w') as f:
    #     json.dump(training_state, f, indent=2)
    
    print(f"Model saved to {save_path} in Hugging Face format")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model checkpoint from Hugging Face format"""
    # 加载模型权重
    model = model.from_pretrained(checkpoint_path)
    
    # 加载训练状态
    training_state_path = os.path.join(checkpoint_path, 'training_state.json')
    if os.path.exists(training_state_path):
        import json
        with open(training_state_path, 'r') as f:
            training_state = json.load(f)
        
        # 恢复优化器和调度器状态
        optimizer.load_state_dict(training_state['optimizer_state_dict'])
        scheduler.load_state_dict(training_state['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from epoch {training_state['epoch']} "
              f"with loss {training_state['loss']}")
        
        return (model, optimizer, scheduler, training_state['epoch'],
                training_state['loss'])
    else:
        print(f"No training state found at {training_state_path}")
        return model, optimizer, scheduler, 0, float('inf')


def main():
    parser = argparse.ArgumentParser(description="Train RenderFormer model")
    
    # Data arguments
    parser.add_argument("--train_data_dir", type=str, required=True, 
                       help="Directory containing training H5 files")
    parser.add_argument("--val_data_dir", type=str, 
                       help="Directory containing validation H5 files")
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=8, 
                       help="Number of data loader workers")
    
    # Model arguments
    parser.add_argument("--model_id", type=str, 
                       default="microsoft/renderformer-v1.1-swin-large",
                       help="Model ID on Hugging Face or local path")
    parser.add_argument("--precision", type=str, choices=['bf16', 'fp16', 'fp32'], 
                       default='fp16', help="Precision for training")
    parser.add_argument("--resolution", type=int, default=256, 
                       help="Resolution for training")
    parser.add_argument("--model_config", type=str, default="F:/projects/renderformer/traindata_test/model/config.json", 
                       help="Model config file")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=1, 
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, 
                       help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, 
                       help="Gradient clipping value")
    parser.add_argument("--loss_type", type=str, choices=['l1', 'l2', 'smooth_l1'], 
                       default='l1', help="Loss function type")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints", 
                       help="Output directory for checkpoints")
    parser.add_argument("--save_freq", type=int, default=1, 
                       help="Save checkpoint every N epochs")
    parser.add_argument("--resume_from", type=str, 
                       help="Resume training from Hugging Face checkpoint directory")
    
    # Logging arguments
    parser.add_argument("--use_wandb", action="store_true", 
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="renderformer", 
                       help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, 
                       help="W&B run name")
    
    args = parser.parse_args()
    
    # Setup device
    device = (torch.device('cuda') if torch.cuda.is_available() 
              else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    
    # Init model
    print(f"Creating new GeoRaster model from scratch...")
    model_config = RenderFormerConfig.from_json(args.model_config)
    
    # config = RenderFormerConfig(norm_first=True)
    pipeline = GeoRasterRenderingPipeline(GeoRaster(model_config))
    print("✓ New model created successfully")
    
    # Apply optimizations
    if device == torch.device('cuda') and os.name == 'posix':  # avoid windows
        try:
            from renderformer_liger_kernel import apply_kernels
            apply_kernels(pipeline.model)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("Applied liger kernel optimizations")
        except ImportError:
            print("Liger kernel not available, skipping optimizations")
    elif device == torch.device('mps'):
        args.precision = 'fp32'
        print("bf16 and fp16 will cause too large error in MPS, "
              "force using fp32 instead.")
    
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
    
    val_dataloader = None
    if args.val_data_dir:
        val_dataset = RenderFormerDataset(args.val_data_dir, args.resolution)
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        pipeline.model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.01
    )

     # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    # if args.resume_from:
    #     pipeline, optimizer, scheduler, start_epoch, best_val_loss = load_checkpoint(
    #         pipeline, optimizer, scheduler, args.resume_from
    #     )
    #     pipeline.to(device)
    #     print(f"Resuming training from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(pipeline, train_dataloader, optimizer, scheduler, device, args)
        print(f"Training loss: {train_loss:.6f}")
        
        # Validate
        val_loss = None
        if val_dataloader is not None:
            val_loss = validate(pipeline, val_dataloader, device, args)
            print(f"Validation loss: {val_loss:.6f}")
            
            # Log to wandb
            if args.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    pipeline, optimizer, scheduler, epoch, val_loss,
                    os.path.join(args.output_dir, "best_model")
                )
        else:
            if args.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss
                })
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                pipeline, optimizer, scheduler, epoch, train_loss,
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}")
            )
    
    # Save final model
    save_checkpoint(
        pipeline, optimizer, scheduler, args.epochs-1, train_loss,
        os.path.join(args.output_dir, "final_model")
    )
    
    print("Training completed!")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main() 