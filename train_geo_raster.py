"""
RenderFormer Training Script with TensorBoard and Wandb Support

This script provides comprehensive training for RenderFormer models with:
- TensorBoard visualization for loss curves, gradients, and sample images
- Wandb integration for experiment tracking
- Multi-GPU support with DataParallel
- Flexible loss functions (L1, L2, LPIPS)
- Checkpoint saving and resuming

Usage:
    # Basic training with TensorBoard
    python train_geo_raster.py --train_data_dir /path/to/data --use_tensorboard
    
    # View TensorBoard logs
    tensorboard --logdir ./tensorboard_logs
    
    # Training with both TensorBoard and Wandb
    python train_geo_raster.py --train_data_dir /path/to/data --use_tensorboard --use_wandb
"""

import os
import torch
import h5py
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb
from pathlib import Path
import imageio
from datetime import datetime
import lpips

from renderformer import GeoRasterRenderingPipeline
from renderformer.models.config import RenderFormerConfig
from renderformer.models.geo_raster import GeoRaster


class RenderFormerDataset(Dataset):
    def __init__(self, data_dir, resolution=256, max_num_tris=2048):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.max_num_tris = max_num_tris
        self.h5_files = list(self.data_dir.glob("*/*.h5"))
        
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


loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
# loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

def convert_for_lpips(img):
    # clip to [0, 1]
    img = torch.clamp(img, 0, 1)
    # normalize to [-1, 1]
    img = (img - 0.5) * 2
    # [B, V, H, W, 3] -> [B*V, 3, H, W]
    img = img.reshape(-1, *img.shape[-3:]).permute(0, 3, 1, 2)
    # print(img.shape)
    # print(img.device)
    return img

def compute_gradient_stats_by_module(model):
    """
    Compute gradient statistics by module for visualization and debugging.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dict containing gradient statistics for each module
    """
    # Handle DataParallel wrapper
    if hasattr(model, 'module'):
        model = model.module
    
    # Define module groups based on the model structure
    module_groups = {
        'embeddings': ['tri_token', 'reg_tokens'],
        'vn_encoding': ['vn_encoding_proj', 'vn_encoder_norm'],
        'transformer': ['transformer.layers'],
        'view_transformer_core': ['view_transformer.transformer'],
        'view_transformer_encoder': ['view_transformer.ray_map_patch_token', 'view_transformer.ray_map_encoder', 'view_transformer.ray_map_encoder_norm'],
        'view_transformer_output': ['view_transformer.out_dpt'],
        'rope_embeddings': ['rope_emb']
    }
    
    gradient_stats = {}
    
    for group_name, module_prefixes in module_groups.items():
        grad_tensors = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Check if this parameter belongs to the current module group
                for prefix in module_prefixes:
                    if name.startswith(prefix):
                        grad_tensors.append(param.grad.data.flatten())
                        break
        
        if grad_tensors:
            # Concatenate all gradient tensors for this module group
            all_grads = torch.cat(grad_tensors, dim=0)
            gradient_stats[group_name] = {
                'mean': all_grads.mean().item(),
                'max': all_grads.max().item(),
            }
    
    return gradient_stats


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
    elif loss_type == 'lpips_alex':
        return loss_fn_alex(convert_for_lpips(pred_images), convert_for_lpips(gt_images))
    # elif loss_type == 'lpips_vgg':
    #     return loss_fn_vgg(convert_for_lpips(pred_images), convert_for_lpips(gt_images))
    elif loss_type == 'l1_w_lpips_alex':
        return F.l1_loss(pred_images, gt_images) + 0.05 * loss_fn_alex(convert_for_lpips(pred_images), convert_for_lpips(gt_images)).mean()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_epoch(model, dataloader, optimizer, scheduler, device, config, log_file=None, tb_writer=None, epoch=0):
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
                # Handle multi-GPU training - convert tensor loss to scalar
                if hasattr(loss, 'mean'):
                    loss = loss.mean()
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
            
            # Log batch loss to file
            if log_file:
                with open(log_file, 'a') as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] Batch {batch_idx}, Loss: {loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.2e}\n")
            
            # Log to TensorBoard
            if tb_writer:
                global_step = epoch * len(dataloader) + batch_idx
                tb_writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
                tb_writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], global_step)
                
                # Log gradient norms by module every 100 steps
                if global_step % 50 == 1:
                    gradient_stats = compute_gradient_stats_by_module(model.model)
                    for module_name, stats in gradient_stats.items():
                        tb_writer.add_scalar(f'Gradients/{module_name}/Mean', stats['mean'], global_step)
                        tb_writer.add_scalar(f'Gradients/{module_name}/Max', stats['max'], global_step)
                
                # Log sample images every 500 steps
                if global_step % 50 == 0 and gt_images is not None:
                    # Take first image from batch for visualization
                    pred_img = torch.clamp(rendered_imgs[0, 0], 0, 1).cpu()  # [H, W, 3]
                    gt_img = torch.clamp(gt_images[0, 0], 0, 1).cpu()  # [H, W, 3]
                    
                    # Convert to format for TensorBoard (CHW)
                    pred_img = pred_img.permute(2, 0, 1)  # [3, H, W]
                    gt_img = gt_img.permute(2, 0, 1)  # [3, H, W]

                    # concatenate pred_img and gt_img
                    concat_img = torch.cat([pred_img, gt_img], dim=2)
                    
                    tb_writer.add_image('Images/Prediction_Ground_Truth', concat_img, global_step)
            
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
    
    # 保存模型权重，处理DataParallel包装
    model_to_save = model.model.module if hasattr(model.model, 'module') else model.model
    model_to_save.save_pretrained(save_path)
    
    # 保存训练状态（优化器、调度器等）
    training_state = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    
    # 保存训练状态到 PyTorch 文件（因为包含tensors，不能用JSON）
    torch.save(training_state, os.path.join(save_path, 'training_state.pt'))
    
    print(f"Model saved to {save_path} in Hugging Face format")


def load_training_state(optimizer, scheduler, checkpoint_path):
    """Load model checkpoint from Hugging Face format"""
    # 加载训练状态
    training_state_path = os.path.join(checkpoint_path, 'training_state.pt')
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path, map_location='cpu')
        
        # 恢复优化器和调度器状态
        optimizer.load_state_dict(training_state['optimizer_state_dict'])
        scheduler.load_state_dict(training_state['scheduler_state_dict'])
        
        print(f"Resuming from epoch {training_state['epoch']} "
              f"with loss {training_state['loss']}")
        
        return (optimizer, scheduler, training_state['epoch'],
                training_state['loss'])
    else:
        print(f"No training state found at {training_state_path}")
        return optimizer, scheduler, 0, float('inf')


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
    parser.add_argument("--pretrained", action="store_true", 
                       help="Use pretrained model")
    parser.add_argument("--model_id", type=str, 
                       default="microsoft/renderformer-v1-base",
                       help="Model ID on Hugging Face or local path")
    parser.add_argument("--precision", type=str, choices=['bf16', 'fp16', 'fp32'], 
                       default='fp16', help="Precision for training")
    parser.add_argument("--resolution", type=int, default=256, 
                       help="Resolution for training")
    parser.add_argument("--max_num_tris", type=int, default=2048, 
                       help="Maximum number of triangles for training")
    parser.add_argument("--model_config", type=str, default="F:/projects/renderformer/traindata_test/model/config.json", 
                       help="Model config file")
    parser.add_argument("--no_data_parallel", action="store_true", 
                       help="Disable DataParallel for multi-GPU training")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=1, 
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, 
                       help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, 
                       help="Gradient clipping value")
    parser.add_argument("--loss_type", type=str, choices=['l1', 'l2', 'smooth_l1', 'lpips_alex', 'lpips_vgg', 'l1_w_lpips_alex'], 
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
    parser.add_argument("--use_tensorboard", action="store_true", 
                       help="Use TensorBoard for logging")
    
    args = parser.parse_args()
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPU(s): {device}")
        if num_gpus > 1:
            print(f"Multi-GPU training will be enabled with DataParallel")
            # Adjust batch size for multi-GPU training
            if args.batch_size % num_gpus != 0:
                print(f"Warning: batch_size ({args.batch_size}) is not divisible by num_gpus ({num_gpus})")
                print(f"Consider using a batch size that's divisible by {num_gpus} for optimal performance")
    else:
        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        num_gpus = 1
        print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging file
    log_file = os.path.join(args.output_dir, "training_log.txt")
    with open(log_file, 'w') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Training Log - Started at {timestamp}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Arguments: {vars(args)}\n")
        f.write("=" * 50 + "\n")
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    
    # Initialize TensorBoard if requested
    tb_writer = None
    if args.use_tensorboard:
        tb_log_dir = os.path.join(args.output_dir, 
                                 f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"TensorBoard logs will be saved to: {tb_log_dir}")
        print(f"Run 'tensorboard --logdir {args.output_dir}' to view logs")
        
        # Log hyperparameters to TensorBoard
        hparams_dict = {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'resolution': args.resolution,
            'max_num_tris': args.max_num_tris,
            'precision': args.precision,
            'loss_type': args.loss_type,
            'grad_clip': args.grad_clip,
            'epochs': args.epochs
        }
        tb_writer.add_hparams(hparams_dict, {'hparam/train_loss': 0.0})
    
    # Init model
    if args.resume_from:
        print(f"Loading checkpoint from {args.resume_from}...")
        pipeline = GeoRasterRenderingPipeline.from_pretrained(args.resume_from)
    elif args.pretrained:
        print(f"Loading pretrained model from {args.model_id}...")
        pipeline = GeoRasterRenderingPipeline.from_pretrained(args.model_id)
    else:
        print(f"Creating new GeoRaster model from scratch...")
        model_config = RenderFormerConfig.from_json(args.model_config)
        pipeline = GeoRasterRenderingPipeline(GeoRaster(model_config))
        print("✓ New model created successfully")

    # Enable multi-GPU training with DataParallel
    if torch.cuda.is_available() and num_gpus > 1 and not args.no_data_parallel:
        pipeline.model = torch.nn.DataParallel(pipeline.model)
        print(f"Model wrapped with DataParallel for {num_gpus} GPUs")
    elif torch.cuda.is_available() and num_gpus > 1 and args.no_data_parallel:
        print(f"DataParallel disabled by --no_data_parallel flag, using single GPU")

    
    # Apply optimizations
    if device.type == 'cuda' and os.name == 'posix':  # avoid windows
        try:
            from renderformer_liger_kernel import apply_kernels
            apply_kernels(pipeline.model)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("Applied liger kernel optimizations")
        except ImportError:
            print("Liger kernel not available, skipping optimizations")
    elif device.type == 'mps':
        args.precision = 'fp32'
        print("bf16 and fp16 will cause too large error in MPS, "
              "force using fp32 instead.")

    pipeline.to(device)
    loss_fn_alex.to(device)
    
    # Create datasets and dataloaders
    train_dataset = RenderFormerDataset(args.train_data_dir, args.resolution, args.max_num_tris)
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
    if args.resume_from:
        optimizer, scheduler, start_epoch, best_val_loss = load_training_state(optimizer, scheduler, args.resume_from)
        print(f"Resuming training from epoch {start_epoch}")
    
    # Print gradient monitoring info
    if args.use_tensorboard:
        print("\n" + "="*60)
        print("GRADIENT MONITORING ENABLED")
        print("="*60)
        print("The following gradient statistics will be logged to TensorBoard:")
        print("- Module-level gradient norms, means, stds, max, min")
        print("- Per-layer statistics for transformer layers")
        print("- Both per-batch (every 100 steps) and per-epoch statistics")
        print("TensorBoard sections:")
        print("  - Gradients/* : Per-batch gradient statistics")
        print("  - Gradients_Epoch/* : Per-epoch gradient statistics")
        print("="*60)
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Log epoch start
        with open(log_file, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n[{timestamp}] Epoch {epoch+1}/{args.epochs}\n")
            f.write("-" * 30 + "\n")
        
        # Train
        train_loss = train_epoch(pipeline, train_dataloader, optimizer, scheduler, device, args, log_file, tb_writer, epoch)
        print(f"Training loss: {train_loss:.6f}")
        
        # Log epoch training loss to TensorBoard
        if tb_writer:
            tb_writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
            
            # Log gradient statistics at epoch end
            gradient_stats = compute_gradient_stats_by_module(pipeline.model)
            for module_name, stats in gradient_stats.items():
                tb_writer.add_scalar(f'Gradients_Epoch/{module_name}/Mean', stats['mean'], epoch)
                tb_writer.add_scalar(f'Gradients_Epoch/{module_name}/Max', stats['max'], epoch)
        
        # Log epoch training loss and gradient statistics
        with open(log_file, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] Epoch {epoch+1} Training Loss: {train_loss:.6f}\n")
            
            # Log gradient statistics to file
            if tb_writer:  # Only log if TensorBoard is enabled
                gradient_stats = compute_gradient_stats_by_module(pipeline.model)
                f.write(f"[{timestamp}] Gradient Statistics:\n")
                for module_name, stats in gradient_stats.items():
                    f.write(f"  {module_name}: mean={stats['mean']:.2e}, max={stats['max']:.2e}\n")
                f.write("\n")
        
        # Validate
        val_loss = None
        if val_dataloader is not None:
            val_loss = validate(pipeline, val_dataloader, device, args)
            print(f"Validation loss: {val_loss:.6f}")
            
            # Log epoch validation loss
            with open(log_file, 'a') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] Epoch {epoch+1} Validation Loss: {val_loss:.6f}\n")
            
            # Log to TensorBoard
            if tb_writer:
                tb_writer.add_scalar('Loss/Validation_Epoch', val_loss, epoch)
            
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
                with open(log_file, 'a') as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] New best model saved! Validation loss: {val_loss:.6f}\n")
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
        if (epoch + 1) % args.save_freq == 1:
            save_checkpoint(
                pipeline, optimizer, scheduler, epoch, train_loss,
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}")
            )
    
    # Save final model
    save_checkpoint(
        pipeline, optimizer, scheduler, args.epochs-1, train_loss,
        os.path.join(args.output_dir, "final_model")
    )
    
    # Log training completion
    with open(log_file, 'a') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"[{timestamp}] Training completed!\n")
        f.write(f"Final training loss: {train_loss:.6f}\n")
        if val_loss is not None:
            f.write(f"Final validation loss: {val_loss:.6f}\n")
        f.write("=" * 50 + "\n")
    
    print("Training completed!")
    
    # Close TensorBoard writer
    if tb_writer:
        # Update final hparams with actual results
        final_metrics = {'hparam/train_loss': train_loss}
        if val_loss is not None:
            final_metrics['hparam/val_loss'] = val_loss
        tb_writer.add_hparams(hparams_dict, final_metrics)
        
        tb_writer.close()
        print(f"TensorBoard logs saved to: {tb_log_dir}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main() 