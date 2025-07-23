import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Add the current directory to Python path to import from train_datasets
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_datasets import ray_triangle_intersection


def generate_random_triangles(num_triangles=3, bounds=(-2, 2)):
    """Generate random triangles within given bounds"""
    triangles = []
    aabbs = []
    
    for _ in range(num_triangles):
        # Generate 3 random vertices for triangle
        vertices = np.random.uniform(bounds[0], bounds[1], (3, 3))
        triangles.append(vertices)
        
        # Calculate AABB
        min_point = np.min(vertices, axis=0)
        max_point = np.max(vertices, axis=0)
        aabbs.append([min_point, max_point])
    
    return np.array(triangles), np.array(aabbs)


def generate_random_rays(num_rays=5, ray_length=3.0):
    """Generate random rays from origin"""
    # Random directions
    directions = np.random.uniform(-1, 1, (num_rays, 3))
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    
    # Ray endpoints
    origins = np.zeros((num_rays, 3))
    endpoints = origins + directions * ray_length
    
    return origins, endpoints, directions


def plot_triangle(ax, vertices, color='blue', alpha=0.3):
    """Plot a triangle as a 3D polygon"""
    # Create polygon collection
    poly = Poly3DCollection([vertices], alpha=alpha, facecolor=color, edgecolor='black')
    ax.add_collection3d(poly)


def plot_aabb(ax, min_point, max_point, color='red', alpha=0.1):
    """Plot AABB as wireframe"""
    x_min, y_min, z_min = min_point
    x_max, y_max, z_max = max_point
    
    # Define the 8 vertices of the AABB
    vertices = [
        [x_min, y_min, z_min], [x_max, y_min, z_min],
        [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max],
        [x_max, y_max, z_max], [x_min, y_max, z_max]
    ]
    
    # Define the 12 edges of the AABB
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]
    
    # Plot edges
    for edge in edges:
        start = vertices[edge[0]]
        end = vertices[edge[1]]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                color=color, alpha=alpha, linewidth=1)


def plot_ray(ax, origin, endpoint, color='green', linewidth=2):
    """Plot a ray from origin to endpoint"""
    ax.plot([origin[0], endpoint[0]], 
            [origin[1], endpoint[1]], 
            [origin[2], endpoint[2]], 
            color=color, linewidth=linewidth, marker='o', markersize=4)


def visualize_ray_triangle_intersection(seed=42):
    """Main visualization function"""
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate random triangles and their AABBs
    triangles, aabbs = generate_random_triangles(num_triangles=3)
    
    # Generate random rays
    ray_origins, ray_endpoints, ray_directions = generate_random_rays(num_rays=4)
    # ray_directions.reshape(1, 2, 2, 3)
    
    # Convert to torch tensors for intersection calculation
    triangles_aabb_torch = torch.tensor(aabbs, dtype=torch.float32).unsqueeze(0)  # [1, num_tris, 2, 3]
    rays_d_torch = torch.tensor(ray_directions.reshape(2, 2, 3), dtype=torch.float32).unsqueeze(0)  # [1, num_rays, 3]
    rays_d_torch = torch.nn.functional.normalize(rays_d_torch, dim=-1)
    
    # Calculate intersections
    intersection_mask = ray_triangle_intersection(triangles_aabb_torch, rays_d_torch)
    intersection_mask = intersection_mask.squeeze(0)  # [num_tris, num_rays]
    intersection_mask = intersection_mask.reshape(intersection_mask.shape[0], -1)
    print(f"intersection_mask: {intersection_mask}")
    print(f"intersection_mask: {intersection_mask.shape}")
    
    print("Intersection results:")
    print("Triangle \\ Ray:", end="")
    for i in range(ray_directions.shape[0]):
        print(f"  {i:2d}", end="")
    print()
    
    for i in range(triangles.shape[0]):
        print(f"Triangle {i:2d}:", end="")
        for j in range(ray_directions.shape[0]):
            result = "✓" if intersection_mask[i, j].item() else "✗"
            print(f"  {result}", end="")
        print()
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot triangles and their AABBs
    colors = ['blue', 'red', 'purple']
    for i, (triangle, aabb) in enumerate(zip(triangles, aabbs)):
        # Plot triangle
        plot_triangle(ax, triangle, color=colors[i], alpha=0.5)
        
        # Plot AABB
        plot_aabb(ax, aabb[0], aabb[1], color=colors[i], alpha=0.3)
        
        # Add triangle center for labeling
        center = np.mean(triangle, axis=0)
        ax.text(center[0], center[1], center[2], f'T{i}', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.7))
    
    # Plot rays with intersection results
    for i, (origin, endpoint, direction) in enumerate(zip(ray_origins, ray_endpoints, ray_directions)):
        # Check if this ray intersects with any triangle
        intersects_any = intersection_mask[:, i].any().item()
        ray_color = 'green' if intersects_any else 'gray'
        
        plot_ray(ax, origin, endpoint, color=ray_color, linewidth=3)
        
        # Add ray label
        mid_point = (origin + endpoint) / 2
        ax.text(mid_point[0], mid_point[1], mid_point[2], f'R{i}', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=ray_color, alpha=0.7))
    
    # Plot origin point
    ax.scatter([0], [0], [0], color='black', s=100, marker='*', label='Origin')
    
    # Set plot properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Ray-Triangle AABB Intersection Visualization\n'
                'Green rays: intersect with at least one triangle\n'
                'Gray rays: no intersection')
    
    # Set equal aspect ratio
    max_range = np.array([triangles[:, :, 0].max() - triangles[:, :, 0].min(),
                         triangles[:, :, 1].max() - triangles[:, :, 1].min(),
                         triangles[:, :, 2].max() - triangles[:, :, 2].min()]).max() / 2.0
    
    mid_x = (triangles[:, :, 0].max() + triangles[:, :, 0].min()) * 0.5
    mid_y = (triangles[:, :, 1].max() + triangles[:, :, 1].min()) * 0.5
    mid_z = (triangles[:, :, 2].max() + triangles[:, :, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.5, label='Triangle 0'),
        Patch(facecolor='red', alpha=0.5, label='Triangle 1'),
        Patch(facecolor='purple', alpha=0.5, label='Triangle 2'),
        Patch(facecolor='green', alpha=0.7, label='Intersecting Ray'),
        Patch(facecolor='gray', alpha=0.7, label='Non-intersecting Ray')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    return intersection_mask


if __name__ == "__main__":
    intersection_results = visualize_ray_triangle_intersection(42)
    print(f"\nIntersection mask shape: {intersection_results.shape}") 