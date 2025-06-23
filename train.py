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

from renderformer import RenderFormerRenderingPipeline


class RenderFormerDataset(Dataset):
    def __init__(self, data_dir, resolution=256):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.h5_files = list(self.data_dir.glob("*.h5"))
        
        if len(self.h5_files) == 0:
            raise ValueError(f"No H5 files found in {data_dir}")
        
        print(f"Found {len(self.h5_files)} H5 files in {data_dir}")
    
    def __len__(self):
        return len(self.h5_files)
    
    def __getitem__(self, idx):
        h5_file = self.h5_files[idx]
        
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

        data = {
            'triangles': triangles,
            'texture': texture,
            'mask': mask,
            'c2w': c2w,
            'fov': fov,
            'vn': vn,
            'gt_img': gt_images,
            'file_path': str(h5_file)
        }
        return data


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


def train_epoch(model, dataloader, optimizer, scheduler, device, config):
    """Train for one epoch"""
    model.model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, data in enumerate(progress_bar):
        # Move data to device
        triangles = data['triangles'].to(device)
        texture = data['texture'].to(device)
        mask = data['mask'].to(device)
        vn = data['vn'].to(device)
        c2w = data['c2w'].to(device)
        fov = data['fov'].to(device)
        gt_images = (data['gt_img'].to(device) 
                    if data['gt_img'] is not None else None)
        
        # Forward pass
        optimizer.zero_grad()
        
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
            
            # Compute loss if ground truth is available
            if gt_images is not None:
                loss = compute_loss(rendered_imgs, gt_images, 
                                  config.loss_type)
            else:
                # If no ground truth, use a simple regularization loss
                loss = torch.mean(torch.abs(rendered_imgs))
            
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
            gt_images = (data['gt_images'].to(device) 
                        if data['gt_images'] is not None else None)
            
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
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train RenderFormer model")
    
    # Data arguments
    parser.add_argument("--train_data_dir", type=str, required=True, 
                       help="Directory containing training H5 files")
    parser.add_argument("--val_data_dir", type=str, 
                       help="Directory containing validation H5 files")
    parser.add_argument("--batch_size", type=int, default=1, 
                       help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, 
                       help="Number of data loader workers")
    
    # Model arguments
    parser.add_argument("--model_id", type=str, 
                       default="microsoft/renderformer-v1.1-swin-large",
                       help="Model ID on Hugging Face or local path")
    parser.add_argument("--precision", type=str, choices=['bf16', 'fp16', 'fp32'], 
                       default='fp16', help="Precision for training")
    parser.add_argument("--resolution", type=int, default=512, 
                       help="Resolution for training")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=2, 
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
    parser.add_argument("--save_freq", type=int, default=10, 
                       help="Save checkpoint every N epochs")
    
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
    
    # Load model
    print(f"Loading model from {args.model_id}")
    pipeline = RenderFormerRenderingPipeline.from_pretrained(args.model_id)
    
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
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
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
                    os.path.join(args.output_dir, "best_model.pth")
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
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            )
    
    # Save final model
    save_checkpoint(
        pipeline, optimizer, scheduler, args.epochs-1, train_loss,
        os.path.join(args.output_dir, "final_model.pth")
    )
    
    print("Training completed!")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main() 