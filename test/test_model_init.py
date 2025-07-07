import argparse
import os
import sys
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from renderformer import GeoRasterRenderingPipeline, RenderFormerRenderingPipeline
from renderformer.models.config import RenderFormerConfig
from renderformer.models.geo_raster import GeoRaster
from renderformer.models.renderformer import RenderFormer


def visualize_parameter_distribution(model, save_dir="./param_visualization"):
    """可视化网络参数分布"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 收集所有参数
    all_params = []
    layer_params = defaultdict(list)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_data = param.data.cpu().numpy().flatten()
            all_params.extend(param_data)
            
            # 按层类型分组
            layer_type = name.split('.')[0] if '.' in name else name
            layer_params[layer_type].extend(param_data)
    
    # 1. 整体参数分布
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.hist(all_params, bins=100, alpha=0.7, density=True)
    plt.title('Overall Parameter Distribution')
    plt.xlabel('Parameter Value')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    
    # 2. 参数分布箱线图
    plt.subplot(2, 2, 2)
    layer_names = list(layer_params.keys())[:10]  # 显示前10层
    layer_data = [layer_params[name] for name in layer_names]
    plt.boxplot(layer_data, labels=layer_names)
    plt.title('Parameter Distribution by Layer Type')
    plt.ylabel('Parameter Value')
    plt.xticks(rotation=45)
    
    # 3. 参数分布热力图 (seaborn)
    plt.subplot(2, 2, 3)
    sns.histplot(all_params, bins=50, kde=True)
    plt.title('Parameter Distribution with KDE')
    plt.xlabel('Parameter Value')
    
    # 4. 参数统计信息
    plt.subplot(2, 2, 4)
    stats = {
        'Mean': np.mean(all_params),
        'Std': np.std(all_params),
        'Min': np.min(all_params),
        'Max': np.max(all_params),
        'Median': np.median(all_params)
    }
    
    plt.bar(stats.keys(), stats.values())
    plt.title('Parameter Statistics')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'parameter_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats


def visualize_layer_parameters(model, save_dir="./param_visualization"):
    """可视化每层参数的详细信息"""
    os.makedirs(save_dir, exist_ok=True)
    
    layer_stats = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 收集每层统计信息
    layer_names = []
    layer_means = []
    layer_stds = []
    layer_param_counts = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_data = param.data.cpu().numpy()
            layer_names.append(name)
            layer_means.append(np.mean(param_data))
            layer_stds.append(np.std(param_data))
            layer_param_counts.append(param_data.size)
            
            layer_stats[name] = {
                'shape': param_data.shape,
                'mean': np.mean(param_data),
                'std': np.std(param_data),
                'min': np.min(param_data),
                'max': np.max(param_data),
                'param_count': param_data.size
            }
    
    # 1. 每层参数均值
    axes[0, 0].bar(range(len(layer_names)), layer_means)
    axes[0, 0].set_title('Layer Parameter Means')
    axes[0, 0].set_ylabel('Mean Value')
    axes[0, 0].set_xticks(range(0, len(layer_names), max(1, len(layer_names)//10)))
    
    # 2. 每层参数标准差
    axes[0, 1].bar(range(len(layer_names)), layer_stds)
    axes[0, 1].set_title('Layer Parameter Standard Deviations')
    axes[0, 1].set_ylabel('Std Value')
    axes[0, 1].set_xticks(range(0, len(layer_names), max(1, len(layer_names)//10)))
    
    # 3. 每层参数数量
    axes[1, 0].bar(range(len(layer_names)), layer_param_counts)
    axes[1, 0].set_title('Parameter Count per Layer')
    axes[1, 0].set_ylabel('Parameter Count')
    axes[1, 0].set_xticks(range(0, len(layer_names), max(1, len(layer_names)//10)))
    axes[1, 0].set_yscale('log')
    
    # 4. 参数范围（最大值-最小值）
    layer_ranges = [layer_stats[name]['max'] - layer_stats[name]['min'] for name in layer_names]
    axes[1, 1].bar(range(len(layer_names)), layer_ranges)
    axes[1, 1].set_title('Parameter Range per Layer')
    axes[1, 1].set_ylabel('Range (Max - Min)')
    axes[1, 1].set_xticks(range(0, len(layer_names), max(1, len(layer_names)//10)))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'layer_parameter_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return layer_stats


def visualize_weight_matrices(model, save_dir="./param_visualization", max_layers=5):
    """可视化权重矩阵热力图"""
    os.makedirs(save_dir, exist_ok=True)
    
    weight_layers = []
    for name, param in model.named_parameters():
        if param.requires_grad and len(param.shape) >= 2 and 'weight' in name:
            weight_layers.append((name, param))
    
    # 只显示前几层权重
    weight_layers = weight_layers[:max_layers]
    
    if weight_layers:
        fig, axes = plt.subplots(1, len(weight_layers), figsize=(5*len(weight_layers), 4))
        if len(weight_layers) == 1:
            axes = [axes]
        
        for i, (name, param) in enumerate(weight_layers):
            weight_data = param.data.cpu().numpy()
            
            # 如果维度太高，只显示前两个维度
            if len(weight_data.shape) > 2:
                weight_data = weight_data.reshape(weight_data.shape[0], -1)
            
            # 限制显示大小
            if weight_data.shape[0] > 100 or weight_data.shape[1] > 100:
                weight_data = weight_data[:100, :100]
            
            im = axes[i].imshow(weight_data, cmap='RdBu', aspect='auto')
            axes[i].set_title(f'{name}\n{weight_data.shape}')
            axes[i].set_xlabel('Output Dimension')
            axes[i].set_ylabel('Input Dimension')
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'weight_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()


def print_parameter_summary(model):
    """打印参数摘要信息"""
    print("\n" + "="*80)
    print("NETWORK PARAMETER SUMMARY")
    print("="*80)
    
    total_params = 0
    trainable_params = 0
    all_params = []
    
    print(f"{'Layer Name':<40} {'Shape':<20} {'Parameters':<12} {'Min':<12} {'Max':<12} {'Mean':<12} {'Trainable'}")
    print("-" * 120)
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        # 计算参数统计信息
        param_data = param.data.cpu().numpy()
        param_min = np.min(param_data)
        param_max = np.max(param_data)
        param_mean = np.mean(param_data)
        
        # 收集所有参数用于全局统计
        all_params.extend(param_data.flatten())
        
        if param.requires_grad:
            trainable_params += param_count
            trainable_str = "✓"
        else:
            trainable_str = "✗"
        
        shape_str = str(list(param.shape))
        print(f"{name:<40} {shape_str:<20} {param_count:<12,} {param_min:<12.6f} {param_max:<12.6f} {param_mean:<12.6f} {trainable_str}")
    
    print("-" * 120)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # 计算模型大小
    param_size = total_params * 4  # 假设 float32
    print(f"Estimated model size: {param_size / 1024 / 1024:.2f} MB")
    
    # 全局参数统计
    if all_params:
        all_params = np.array(all_params)
        print(f"\nGLOBAL PARAMETER STATISTICS:")
        print(f"  Min value: {np.min(all_params):.6f}")
        print(f"  Max value: {np.max(all_params):.6f}")
        print(f"  Mean value: {np.mean(all_params):.6f}")
        print(f"  Std value: {np.std(all_params):.6f}")
        print(f"  Median value: {np.median(all_params):.6f}")
        print(f"  25th percentile: {np.percentile(all_params, 25):.6f}")
        print(f"  75th percentile: {np.percentile(all_params, 75):.6f}")
        
        # 零值和近零值分析
        zero_count = np.sum(all_params == 0)
        near_zero_count = np.sum(np.abs(all_params) < 1e-6)
        print(f"  Zero parameters: {zero_count:,} ({zero_count/len(all_params)*100:.2f}%)")
        print(f"  Near-zero parameters (|x| < 1e-6): {near_zero_count:,} ({near_zero_count/len(all_params)*100:.2f}%)")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Test RenderFormer model initialization and visualize parameters")
    
    # Model arguments
    parser.add_argument("--pipeline_type", type=str, 
                       default="GeoRasterRenderingPipeline",
                       help="Pipeline type: GeoRasterRenderingPipeline, RenderFormerRenderingPipeline")
    parser.add_argument("--pretrained", action="store_true", 
                       help="Use pretrained model")
    parser.add_argument("--model_id", type=str,     
                       default="microsoft/renderformer-v1-base",
                       help="Model ID on Hugging Face or local path")
    parser.add_argument("--model_config", type=str, 
                       default="/home/luminyang/workspaces/renderformer/training/0701base/config.json", 
                       help="Model config file")
    
    # Visualization arguments
    parser.add_argument("--visualize", action="store_true", 
                       help="Enable parameter visualization")
    parser.add_argument("--vis_dir", type=str, default="./param_visualization",
                       help="Directory to save visualization results")
    
    args = parser.parse_args()
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device}")
    else:
        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        print(f"Using device: {device}")
    
    # Init model
    if args.pretrained:
        print(f"Loading pretrained model from {args.model_id}...")
        if args.pipeline_type == "GeoRasterRenderingPipeline":
            pipeline = GeoRasterRenderingPipeline.from_pretrained(args.model_id)
        elif args.pipeline_type == "RenderFormerRenderingPipeline":
            pipeline = RenderFormerRenderingPipeline.from_pretrained(args.model_id)
        else:
            raise ValueError(f"Invalid pipeline type: {args.pipeline_type}")
    else:
        print(f"Creating new GeoRaster model from scratch...")
        model_config = RenderFormerConfig.from_json(args.model_config)
        if args.pipeline_type == "GeoRasterRenderingPipeline":
            pipeline = GeoRasterRenderingPipeline(GeoRaster(model_config))
        elif args.pipeline_type == "RenderFormerRenderingPipeline":
            pipeline = RenderFormerRenderingPipeline(RenderFormer(model_config))
        else:
            raise ValueError(f"Invalid pipeline type: {args.pipeline_type}")
        print("✓ New model created successfully")
    
    # Move to device
    pipeline.to(device)
    
    print(f"Model: {pipeline.model}")
    print(f"Model type: {type(pipeline.model)}")
    
    # Print detailed parameter summary
    print_parameter_summary(pipeline.model)
    
    # Visualize parameters if requested
    if args.visualize:
        print(f"\nGenerating parameter visualizations...")
        print(f"Saving to: {args.vis_dir}")
        
        # Parameter distribution
        stats = visualize_parameter_distribution(pipeline.model, args.vis_dir)
        print(f"✓ Parameter distribution saved")
        
        # Layer-wise analysis
        layer_stats = visualize_layer_parameters(pipeline.model, args.vis_dir)
        print(f"✓ Layer-wise analysis saved")
        
        # Weight matrices
        visualize_weight_matrices(pipeline.model, args.vis_dir)
        print(f"✓ Weight matrices saved")
        
        print(f"\nVisualization completed! Check {args.vis_dir} for results.")
    
    print("✓ Model initialization test completed successfully")


if __name__ == '__main__':
    main()