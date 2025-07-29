"""
RenderFormer Training Script with Wandb Support

This script provides comprehensive training for RenderFormer models with:
- Wandb integration for experiment tracking and visualization
- Multi-GPU support with DistributedDataParallel
- Flexible loss functions (L1, L2, LPIPS)
- Checkpoint saving and resuming
"""

import os
import torch
import h5py
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from renderformer import GeoRasterRenderingPipeline, RenderFormerRenderingPipeline, TileBasedRenderingPipeline
from renderformer.models.config import RenderFormerConfig
from renderformer.models.geo_raster import GeoRaster
from renderformer.models.renderformer import RenderFormer

from train_loss import compute_loss, loss_fn_alex
from train_tools import compute_gradient_stats_by_module, save_checkpoint, load_epoch_from_training_state, load_optimizer_state
from train_datasets import RenderFormerDataset, tile_based_collate_fn


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Not using distributed training")
        return False, 0, 1, 0
    
    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://')
    
    # Set the device for this process
    torch.cuda.set_device(local_rank)
    
    print(f"Rank {rank}/{world_size}, Local rank {local_rank}")
    return True, rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def train_epoch(pipeline, dataloader, optimizer, scheduler, device, config, scaler=None, tb_writer=None, epoch=0, rank=0):
    """Train for one epoch"""
    pipeline.model.train()
    total_loss = 0.0
    num_batches = 0

    if isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(epoch)
    
    # Only show progress bar on rank 0
    if rank == 0:
        progress_bar = tqdm(dataloader, desc="Training")
    else:
        progress_bar = dataloader
    
    for batch_idx, data in enumerate(progress_bar):
        # Move data to device
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
                
            rendered_imgs = pipeline(
                data=data,
                resolution=config.resolution,
                torch_dtype=torch_dtype,
                device=device,
            )
            
            # Compute loss if ground truth is available
            if gt_images is not None:
                loss = compute_loss(rendered_imgs, gt_images.to(torch.float16), 
                                  config.loss_type)
                
                # Handle multi-GPU training - convert tensor loss to scalar
                if hasattr(loss, 'mean'):
                    loss = loss.mean()
            else:
                print(f"gt_images is None")
            
            optimizer.zero_grad()
            
            # Backward pass with GradScaler if available
            if scaler is not None:
                scaler.scale(loss).backward()
                
                # Gradient clipping with scaler
                if config.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(pipeline.model.parameters(), 
                                                 config.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Traditional backward pass
                loss.backward()
                
                # Gradient clipping
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(pipeline.model.parameters(), 
                                                 config.grad_clip)
                
                optimizer.step()
            
            # Step scheduler every batch for warmup + cosine decay
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log to TensorBoard
            # if tb_writer:
            #     global_step = epoch * len(dataloader) + batch_idx
            #     tb_writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
            #     tb_writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], global_step)
            #     
            #     # Log gradient norms by module every 100 steps
            #     if global_step % 1000 == 1:
            #         gradient_stats = compute_gradient_stats_by_module(pipeline.model)
            #         for module_name, stats in gradient_stats.items():
            #             tb_writer.add_scalar(f'Gradients/{module_name}/Mean', stats['mean'], global_step)
            #             tb_writer.add_scalar(f'Gradients/{module_name}/Max', stats['max'], global_step)
            #     
            #     # Log sample images every 500 steps
            #     if global_step % 25 == 0 and gt_images is not None:
            #         # Take first image from batch for visualization
            #         pred_img = torch.clamp(rendered_imgs[0, 0], 0, 1).cpu()  # [H, W, 3]
            #         gt_img = torch.clamp(gt_images[0, 0], 0, 1).cpu()  # [H, W, 3]
            #         
            #         # Convert to format for TensorBoard (CHW)
            #         pred_img = pred_img.permute(2, 0, 1)  # [3, H, W]
            #         gt_img = gt_img.permute(2, 0, 1)  # [3, H, W]

            #         # concatenate pred_img and gt_img
            #         concat_img = torch.cat([pred_img, gt_img], dim=2)
            #         
            #         tb_writer.add_image('Images/Prediction_Ground_Truth', concat_img, global_step)
            
            # Log to Wandb (only on rank 0)
            if config.use_wandb and rank == 0:
                global_step = epoch * len(dataloader) + batch_idx
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': scheduler.get_last_lr()[0],
                    'batch': batch_idx,
                    'global_step': global_step
                })
                
                # Log gradient norms by module every 1000 steps
                if global_step % 1000 == 1:
                    gradient_stats = compute_gradient_stats_by_module(pipeline.model)
                    for module_name, stats in gradient_stats.items():
                        wandb.log({
                            f'gradients/{module_name}/mean': stats['mean'],
                            f'gradients/{module_name}/max': stats['max'],
                            'global_step': global_step
                        })
                
                # Log sample images every 50 steps
                if global_step % 50 == 0 and gt_images is not None:
                    # Take first image from batch for visualization
                    pred_img = torch.clamp(rendered_imgs[0, 0], 0, 1).cpu()  # [H, W, 3]
                    gt_img = torch.clamp(gt_images[0, 0], 0, 1).cpu()  # [H, W, 3]
                    
                    # Convert to format for Wandb (CHW)
                    pred_img = pred_img.permute(2, 0, 1)  # [3, H, W]
                    gt_img = gt_img.permute(2, 0, 1)  # [3, H, W]

                    # concatenate pred_img and gt_img
                    concat_img = torch.cat([pred_img, gt_img], dim=2)
                    
                    wandb.log({
                        'images/prediction_ground_truth': wandb.Image(concat_img),
                        'global_step': global_step
                    })
            
            # Update progress bar (only on rank 0)
            if rank == 0:
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
            
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU OOM in batch {batch_idx}, skipping...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    # scheduler.step() is now called after each batch, not at epoch end
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, dataloader, device, config, rank=0):
    """Validate the model"""
    model.model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        if rank == 0:
            data_iter = tqdm(dataloader, desc="Validation")
        else:
            data_iter = dataloader
        
        for data in data_iter:
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


def main():
    parser = argparse.ArgumentParser(description="Train RenderFormer model")

    parser.add_argument("--pipeline_type", type=str, 
                       default="GeoRasterRenderingPipeline",
                       help="Pipeline type: GeoRasterRenderingPipeline, RenderFormerRenderingPipeline")
    
    
    # Data arguments
    parser.add_argument("--train_data_dir", type=str, required=True, 
                       help="Directory containing training H5 files")
    parser.add_argument("--val_data_dir", type=str, 
                       help="Directory containing validation H5 files")
    parser.add_argument("--batch_size", type=int, default=128, 
                       help="Batch size for training (paper recommends 128)")
    parser.add_argument("--num_workers", type=int, default=8, 
                       help="Number of data loader workers")
    
    # Model arguments
    parser.add_argument("--precision", type=str, choices=['bf16', 'fp16', 'fp32'], 
                       default='fp16', help="Precision for training")
    parser.add_argument("--resolution", type=int, default=256, 
                       help="Resolution for training")
    parser.add_argument("--max_num_tris", type=int, default=2048, 
                       help="Maximum number of triangles for training")
    parser.add_argument("--model_config", type=str, default="F:/projects/renderformer/traindata_test/model/config.json", 
                       help="Model config file")
    parser.add_argument("--resume_from", type=str, 
                       help="Resume training from Hugging Face checkpoint directory")
    parser.add_argument("--pretrain_from", type=str, 
                       help="Pretrain model from Hugging Face checkpoint directory")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=1, 
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, 
                       help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=0, 
                       help="Gradient clipping value")
    parser.add_argument("--loss_type", type=str, choices=['l1', 'l2', 'l1_w_l2', 'lpips_alex', 'lpips_vgg', 'l1_w_lpips_alex'], 
                       default='l1', help="Loss function type")
    parser.add_argument("--warmup_steps", type=int, default=8000, 
                       help="Number of warmup steps for learning rate")
    parser.add_argument("--cosine_decay_steps", type=int, default=None, 
                       help="Number of steps for cosine decay (if None, use total training steps)")
    parser.add_argument("--min_lr_ratio", type=float, default=0.01, 
                       help="Minimum learning rate as ratio of target LR (default: 0.01 = 1%)")
    parser.add_argument("--no_distributed", action="store_true", 
                       help="Disable distributed training even if multiple GPUs are available")
    
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
    parser.add_argument("--wandb_run_name", type=str, default="test_run", 
                       help="W&B run name")
    parser.add_argument("--wandb_dir", type=str, 
                       help="Directory to store wandb local files (default: ./wandb)")
    parser.add_argument("--use_tensorboard", action="store_true", 
                       help="Use TensorBoard for logging")
    parser.add_argument("--log_dir", type=str, default="./logs", 
                       help="Output directory for logs")
    
    args = parser.parse_args()

    # Setup distributed training
    is_distributed, rank, world_size, local_rank = setup_distributed()
    
    # Setup device
    if torch.cuda.is_available():
        if is_distributed:
            device = torch.device(f'cuda:{local_rank}')
            print(f"Rank {rank}: Using GPU {local_rank}")
        else:
            device = torch.device('cuda')
            num_gpus = torch.cuda.device_count()
            if rank == 0:
                print(f"Using {num_gpus} GPU(s) in single-process mode")
    else:
        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        if rank == 0:
            print(f"Using device: {device}")

    PIPELINE_CLASS = None
    MODEL_CLASS = None
    if args.pipeline_type == "GeoRasterRenderingPipeline":
        PIPELINE_CLASS = GeoRasterRenderingPipeline
        MODEL_CLASS = GeoRaster
    elif args.pipeline_type == "RenderFormerRenderingPipeline":
        PIPELINE_CLASS = RenderFormerRenderingPipeline
        MODEL_CLASS = RenderFormer
    elif args.pipeline_type == "TileBasedRenderingPipeline":
        PIPELINE_CLASS = TileBasedRenderingPipeline
        MODEL_CLASS = GeoRaster
    else:
        raise ValueError(f"Invalid pipeline type: {args.pipeline_type}")
    
    # Create output directory (only on rank 0)
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize TensorBoard if requested (only on rank 0)
    # tb_writer = None
    # tb_log_dir = None
    # if args.use_tensorboard and rank == 0:
    #     tb_log_dir = os.path.join(args.log_dir, 
    #                              f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    #     os.makedirs(tb_log_dir, exist_ok=True)
    #     tb_writer = SummaryWriter(log_dir=tb_log_dir)
    #     print(f"TensorBoard logs will be saved to: {tb_log_dir}")
    #     print(f"Run 'tensorboard --logdir {args.log_dir}' to view logs")
    
    # Initialize Wandb if requested (only on rank 0)
    if args.use_wandb and rank == 0:
        # Set wandb directory if specified
        if args.wandb_dir:
            os.environ['WANDB_DIR'] = args.wandb_dir
            print(f"Wandb local files will be saved to: {args.wandb_dir}")
        
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            dir=args.wandb_dir if args.wandb_dir else None
        )
        print(f"Wandb initialized with project: {args.wandb_project}, run: {args.wandb_run_name}")

    # Setup logging file (only on rank 0)
    log_file = os.path.join(args.output_dir, "training_log.txt")
    if rank == 0:
        with open(log_file, 'w') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"Training Log - Started at {timestamp}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Arguments: {vars(args)}\n")
            f.write("=" * 50 + "\n")
            # f.write(f"TensorBoard logs will be saved to: {tb_log_dir}\n")
            f.write("=" * 50 + "\n")
    
    # Init model
    if args.resume_from:
        if rank == 0:
            print(f"Loading checkpoint from {args.resume_from}...")
        pipeline = PIPELINE_CLASS.from_pretrained(args.resume_from)
    elif args.pretrain_from:
        if rank == 0:
            print(f"Loading pretrained model from {args.pretrain_from}...")
        pipeline = PIPELINE_CLASS.from_pretrained(args.pretrain_from)
    else:
        if rank == 0:
            print(f"Creating new {args.pipeline_type} model from scratch...")
        model_config = RenderFormerConfig.from_json(args.model_config)
        pipeline = PIPELINE_CLASS(MODEL_CLASS(model_config))
        if rank == 0:
            print("✓ New model created successfully")

    # Apply optimizations
    # if False and device.type == 'cuda' and os.name == 'posix':  # avoid windows
    if device.type == 'cuda' and os.name == 'posix':  # avoid windows
        try:
            from renderformer_liger_kernel import apply_kernels
            apply_kernels(pipeline.model)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if rank == 0:
                print("Applied liger kernel optimizations")
        except ImportError:
            if rank == 0:
                print("Liger kernel not available, skipping optimizations")
    elif device.type == 'mps':
        args.precision = 'fp32'
        if rank == 0:
            print("bf16 and fp16 will cause too large error in MPS, "
                  "force using fp32 instead.")

    pipeline.to(device)
    loss_fn_alex.to(device)

    # Enable distributed training with DDP
    if is_distributed and torch.cuda.is_available() and not args.no_distributed:
        pipeline.model = DDP(pipeline.model, device_ids=[local_rank], output_device=local_rank)
        if rank == 0:
            print(f"Model wrapped with DistributedDataParallel for distributed training")
    elif is_distributed and args.no_distributed:
        if rank == 0:
            print(f"Distributed training disabled by --no_distributed flag")

    # pipeline.to(device)
    # loss_fn_alex.to(device)
    
    # Create datasets and dataloaders
    train_dataset = RenderFormerDataset(args.train_data_dir, args.resolution, args.max_num_tris, args.pipeline_type)
    
    # Use DistributedSampler for distributed training
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    
    if args.pipeline_type == "TileBasedRenderingPipeline":
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=tile_based_collate_fn
        )
    else:
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    val_dataloader = None
    val_sampler = None
    if args.val_data_dir:
        val_dataset = RenderFormerDataset(args.val_data_dir, args.resolution, args.max_num_tris, args.pipeline_type)
        
        if is_distributed:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )
            
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        pipeline.model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    if args.resume_from:
        optimizer = load_optimizer_state(optimizer, args.resume_from, args.learning_rate)
    
    # Calculate total training steps
    total_steps = len(train_dataloader) * args.epochs
    
    # Setup learning rate scheduler with warmup + cosine decay
    if args.cosine_decay_steps is None:
        cosine_decay_steps = total_steps - args.warmup_steps
    else:
        cosine_decay_steps = args.cosine_decay_steps
    
    # Create warmup scheduler (linear increase from 0 to target LR)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,  # Start from 1% of target LR
        end_factor=1.0,     # End at 100% of target LR
        total_iters=args.warmup_steps
    )
    
    # Create cosine decay scheduler
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_decay_steps,
        eta_min=args.learning_rate * args.min_lr_ratio  # End at specified ratio of target LR
    )
    
    # Combine schedulers
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_steps]
    )

    # print scheduler state (only on rank 0)
    if rank == 0:
        print(f"Scheduler state: {scheduler.state_dict()}")
    
    # Initialize GradScaler for mixed precision training
    scaler = torch.amp.GradScaler()
    if rank == 0:
        print(f"GradScaler initialized for {args.precision} mixed precision training")
        
        print(f"Learning rate schedule:")
        print(f"  - Warmup steps: {args.warmup_steps}")
        print(f"  - Cosine decay steps: {cosine_decay_steps}")
        print(f"  - Total steps: {total_steps}")
        print(f"  - Target LR: {args.learning_rate}")
        print(f"  - Min LR: {args.learning_rate * args.min_lr_ratio} (ratio: {args.min_lr_ratio})")
        
        if args.use_wandb:
            print(f"Wandb logging enabled - project: {args.wandb_project}, run: {args.wandb_run_name}")
        # if args.use_tensorboard:
        #     print(f"TensorBoard logging enabled - logs saved to: {tb_log_dir}")
    

    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume_from:
        start_epoch = load_epoch_from_training_state(args.resume_from)
        if rank == 0:
            print(f"Resuming training from epoch {start_epoch}")
    
    # Print gradient monitoring info (only on rank 0)
    # if args.use_tensorboard and rank == 0:
    #     print("\n" + "="*60)
    #     print("GRADIENT MONITORING ENABLED")
    #     print("="*60)
    #     print("The following gradient statistics will be logged to TensorBoard:")
    #     print("- Module-level gradient means, max")
    #     print("- Both per-batch (every 100 steps) and per-epoch statistics")
    #     print("TensorBoard sections:")
    #     print("  - Gradients/* : Per-batch gradient statistics")
    #     print("  - Gradients_Epoch/* : Per-epoch gradient statistics")
    #     print("="*60)
    
    # Print gradient monitoring info for Wandb (only on rank 0)
    if args.use_wandb and rank == 0:
        print("\n" + "="*60)
        print("GRADIENT MONITORING ENABLED (WANDB)")
        print("="*60)
        print("The following gradient statistics will be logged to Wandb:")
        print("- Module-level gradient means, max")
        print("- Both per-batch (every 1000 steps) and per-epoch statistics")
        print("Wandb sections:")
        print("  - gradients/* : Per-batch gradient statistics")
        print("  - gradients_epoch/* : Per-epoch gradient statistics")
        print("  - images/prediction_ground_truth : Sample images")
        print("="*60)
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Set epoch for distributed sampler
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_loss = train_epoch(pipeline, train_dataloader, optimizer, scheduler, device, args, scaler, None, epoch, rank)
        if rank == 0:
            print(f"Training loss: {train_loss:.6f}")
        
        # Log epoch training loss to TensorBoard (only on rank 0)
        # if tb_writer and rank == 0:
        #     tb_writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
        #     
        #     # Log gradient statistics at epoch end
        #     gradient_stats = compute_gradient_stats_by_module(pipeline.model)
        #     for module_name, stats in gradient_stats.items():
        #         tb_writer.add_scalar(f'Gradients_Epoch/{module_name}/Mean', stats['mean'], epoch)
        #         tb_writer.add_scalar(f'Gradients_Epoch/{module_name}/Max', stats['max'], epoch)
        
        # Log epoch training loss to Wandb (only on rank 0)
        if args.use_wandb and rank == 0:
            wandb.log({
                'epoch': epoch,
                'train_loss_epoch': train_loss
            })
            
            # Log gradient statistics at epoch end
            gradient_stats = compute_gradient_stats_by_module(pipeline.model)
            for module_name, stats in gradient_stats.items():
                wandb.log({
                    f'gradients_epoch/{module_name}/mean': stats['mean'],
                    f'gradients_epoch/{module_name}/max': stats['max'],
                    'epoch': epoch
                })
        
        # Validate
        val_loss = None
        if val_dataloader is not None:
            val_loss = validate(pipeline, val_dataloader, device, args, rank)
            if rank == 0:
                print(f"Validation loss: {val_loss:.6f}")
            
            # Log to TensorBoard (only on rank 0)
            # if tb_writer and rank == 0:
            #     tb_writer.add_scalar('Loss/Validation_Epoch', val_loss, epoch)
            
            # Log to wandb (only on rank 0)
            if args.use_wandb and rank == 0:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                })
            
            # Save best model (only on rank 0)
            if val_loss < best_val_loss and rank == 0:
                best_val_loss = val_loss
                save_checkpoint(
                    pipeline, optimizer, scheduler, epoch, val_loss,
                    os.path.join(args.output_dir, "best_model")
                )
        else:
            if args.use_wandb and rank == 0:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss
                })
        
        # Save checkpoint periodically (only on rank 0)
        if (epoch + 1) % args.save_freq == 1 and rank == 0:
            save_checkpoint(
                pipeline, optimizer, scheduler, epoch, train_loss,
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}")
            )
    
    # Save final model (only on rank 0)
    if rank == 0:
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
        # if tb_writer:
        #     tb_writer.close()
        #     print(f"TensorBoard logs saved to: {tb_log_dir}")
        
        if args.use_wandb:
            wandb.finish()
    
    # Clean up distributed training
    if is_distributed:
        cleanup_distributed()


if __name__ == '__main__':
    main() 