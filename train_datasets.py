
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import imageio
import random
import logging

from renderformer.utils.transform import trans_to_cam_coord
from renderformer.utils.ray_generator import RayGenerator
from einops import rearrange


class RenderFormerDataset(Dataset):
    def __init__(self, data_dir, resolution=256, max_num_tris=2048, pipeline_type="GeoRasterRenderingPipeline"):
        self.pipeline_type = pipeline_type
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.max_num_tris = max_num_tris
        self.h5_files = list(self.data_dir.glob("*/*.h5"))
        self.exr_file_path = '_normal_depth.exr' if self.pipeline_type == "GeoRasterRenderingPipeline" else '_lighting.exr'
        self.need_texture = False if self.pipeline_type == "GeoRasterRenderingPipeline" else True
        
        if len(self.h5_files) == 0:
            raise ValueError(f"No H5 files found in {data_dir}")
        
        print(f"Found {len(self.h5_files)} H5 files in {data_dir}")
        
        # Setup logging for file errors
        self.logger = logging.getLogger(f"RenderFormerDataset_{id(self)}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.WARNING)
    
    def __len__(self):
        return len(self.h5_files)
    
    def _load_data_with_retry(self, idx, max_retries=5):
        """
        Try to load data from file with error handling and retry mechanism
        """
        for attempt in range(max_retries):
            try:
                return self._load_single_file(idx)
            except Exception as e:
                h5_file = self.h5_files[idx]
                error_msg = f"Error loading file {h5_file}: {type(e).__name__}: {str(e)}"
                self.logger.error(error_msg)
                print(f"[FILE ERROR] {error_msg}")
                
                if attempt < max_retries - 1:
                    # Try a random different file for next attempt
                    idx = random.randint(0, len(self.h5_files) - 1)
                    print(f"[RETRY] Attempting to load different file (attempt {attempt + 2}/{max_retries}): {self.h5_files[idx]}")
                else:
                    print(f"[CRITICAL] Failed to load any file after {max_retries} attempts")
                    raise e
        
        # This should never be reached, but just in case
        raise RuntimeError(f"Failed to load data after {max_retries} attempts")
    
    def _load_single_file(self, idx):
        """
        Load data from a single file (without retry logic)
        """
        h5_file = self.h5_files[idx]
        
        # triangles: [num_tris, 3, 3]
        # mask: [num_tris]
        # vn: [num_tris, 3, 3]
        # c2w: [num_views, 4, 4]
        # fov: [num_views, 1]
        # gt_img: [num_views, H, W, 3]
        # todo: num_views is not always 1, need to handle this
        
        try:
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
                if self.need_texture:
                    texture = torch.from_numpy(np.array(f['texture'])).float()
        except Exception as e:
            raise Exception(f"Failed to read H5 file {h5_file}: {type(e).__name__}: {str(e)}")

        # Pad triangles to max_num_tris
        if num_tris < self.max_num_tris:
            # Create padding for triangles [max_num_tris - num_tris, 3, 3]
            triangles_padding = torch.zeros(
                self.max_num_tris - num_tris, 3, 3, dtype=triangles.dtype
            )
            triangles = torch.cat([triangles, triangles_padding], dim=0)
            if self.need_texture:
                texture = torch.concatenate((texture, torch.zeros(
                    (self.max_num_tris - num_tris, *texture.shape[1:]))), dim=0)
            
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
            if self.need_texture:
                texture = texture[:self.max_num_tris]
            vn = vn[:self.max_num_tris]
            mask = torch.ones(self.max_num_tris, dtype=torch.bool)
        else:
            # Exact size, no padding needed
            mask = torch.ones(self.max_num_tris, dtype=torch.bool)

        # Load ground truth images
        gt_images_exr_file = str(h5_file).replace('.h5', self.exr_file_path)
        try:
            gt_images = torch.from_numpy(
                imageio.v3.imread(gt_images_exr_file).astype(np.float32)[..., :3]
            ).unsqueeze(0)
        except Exception as e:
            raise Exception(f"Failed to read EXR file {gt_images_exr_file}: {type(e).__name__}: {str(e)}")

        if self.pipeline_type == "GeoRasterRenderingPipeline":
            data = {
                'triangles': triangles,  # Now [max_num_tris, 3, 3]
                'mask': mask,           # Now [max_num_tris]
                'c2w': c2w,
                'fov': fov,
                'vn': vn,              # Now [max_num_tris, 3, 3]
                'gt_img': gt_images,
                'file_path': str(h5_file)
            }
        elif self.pipeline_type == "RenderFormerRenderingPipeline":
            data = {
                'triangles': triangles,  # Now [max_num_tris, 3, 3]
                'texture': texture,
                'mask': mask,           # Now [max_num_tris]
                'c2w': c2w,
                'fov': fov,
                'vn': vn,              # Now [max_num_tris, 3, 3]
                'gt_img': gt_images,
                'file_path': str(h5_file)
            }
        else:
            raise ValueError(f"Invalid pipeline type: {self.pipeline_type}")
        return data

    def __getitem__(self, idx):
        return self._load_data_with_retry(idx)


def ray_triangle_intersection(triangles_aabb, rays_d):
    """
    Check intersection between rays and triangle AABBs using slab method.      
    
    Args:
        triangles_aabb: [batch_size, num_tris, 2, 3] - AABB of triangles
                       triangles_aabb[..., 0, :] is min point, triangles_aabb[..., 1, :] is max point
        rays_d: [batch_size, H, W, 3] - ray directions (normalized, non-zero)
    
    Returns:
        mask: [batch_size, num_tris, H, W] - boolean mask indicating intersection
    """
    batch_size, num_tris, _, _ = triangles_aabb.shape
    _, H, W, _ = rays_d.shape
    
    # Reshape for broadcasting
    # triangles_aabb: [batch_size, num_tris, 2, 3] -> [batch_size, num_tris, 1, 1, 2, 3]
    triangles_aabb = triangles_aabb.unsqueeze(2).unsqueeze(3)
    
    # rays_d: [batch_size, H, W, 3] -> [batch_size, 1, H, W, 1, 3]
    rays_d = rays_d.unsqueeze(1).unsqueeze(4)
    
    # Extract min and max points of AABB
    # min_point: [batch_size, num_tris, 1, 1, 1, 3]
    # max_point: [batch_size, num_tris, 1, 1, 1, 3]
    min_point = triangles_aabb[..., 0, :].unsqueeze(-2)
    max_point = triangles_aabb[..., 1, :].unsqueeze(-2)

    t_low = min_point / rays_d # [batch_size, num_tris, H, W, 1, 3]
    t_high = max_point / rays_d # [batch_size, num_tris, H, W, 1, 3]

    # print(f"t_low: {t_low}")
    # print(f"t_high: {t_high}")

    t_close_axis = torch.minimum(t_low, t_high)
    t_far_axis = torch.maximum(t_low, t_high)

    # print(f"t_close_axis: {t_close_axis}")
    # print(f"t_far_axis: {t_far_axis}")

    t_close = torch.max(t_close_axis, dim=5)[0] # [batch_size, num_tris, H, W, 1, 1]    
    t_far = torch.min(t_far_axis, dim=5)[0] # [batch_size, num_tris, H, W, 1, 1]

    # print(f"t_close: {t_close}")
    # print(f"t_far: {t_far}")

    intersection_mask = (t_close <= t_far) & (t_far >= 0) # [batch_size, num_tris, H, W, 1, 1]
    
    # Remove extra dimensions
    intersection_mask = intersection_mask.squeeze(-1).squeeze(-1)  # [batch_size, num_tris, H, W]
    
    return intersection_mask

def triangle_mask_per_tile(triangles, c2w, fov, resolution, ray_generator: RayGenerator=None, tile_size=8):
    """
    triangles: [batch_size, num_tris, 3, 3]
    c2w: [batch_size, 4, 4]
    patch_size: int
    """
    # print(f"triangles: {triangles.shape}")
    # print(f"c2w: {c2w.shape}")
    # print(f"fov: {fov.shape}")

    triangles_cam, c2w_cam, _ = trans_to_cam_coord(c2w, triangles)
    # print(f"triangles_cam: {triangles_cam.shape}")
    # print(f"c2w_cam: {c2w_cam.shape}")

    # calculate the AABB of every triangle
    # [batch_size, num_tris, 3, 3] --> [batch_size, num_tris, 2, 3] 
    triangles_aabb = torch.cat([torch.min(triangles_cam, dim=2)[0] , torch.max(triangles_cam, dim=2)[0]], dim=2)

    _, rays_d = ray_generator(c2w_cam, fov / 180. * torch.pi, resolution) # [batch_size, H, W, 3]

    # intersect triangles with rays and get the mask
    # [batch_size, num_tris, 2, 3] --> [batch_size, num_tris, H, W]
    # print(f"triangles_aabb: {triangles_aabb.shape}")
    # print(f"rays_d: {rays_d.shape}")
    bs, tn, _ = triangles_aabb.shape
    mask_per_pixel = ray_triangle_intersection(triangles_aabb.reshape(bs, tn, 2, 3), rays_d) # [batch_size, num_tris, H, W]

    print(f"mask_per_pixel: {mask_per_pixel.shape}")
    triangle_num_per_pixel = mask_per_pixel.sum(dim=1) # [batch_size, H, W]
    print(f"triangle_num_per_pixel: {triangle_num_per_pixel.shape}")
    print(f"max triangle_num_per_pixel: {triangle_num_per_pixel.max()}")

    # 8*8 pixels as a tile
    triangle_mask_per_tile = rearrange(mask_per_pixel, 'b t (h1 p1) (w1 p2) -> b t (h1 w1) (p1 p2)', p1=tile_size, p2=tile_size).max(dim=-1)[0]
    print(f"triangle_num_per_tile: {triangle_mask_per_tile.shape}")
    triangle_num_per_tile = triangle_mask_per_tile.sum(dim=1)
    print(f"tile_size: {tile_size}, max triangle_num_per_tile: {triangle_num_per_tile.max()}")
    test_num = tile_size * 2
    print(f"per tile triangle num > {test_num}: {(triangle_num_per_tile > test_num).sum()}")


    return triangle_mask_per_tile


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_data_dir = r"F:\projects\renderformer\traindata\tri1024_v2"
    resolution = 256
    max_num_tris = 2048
    pipeline_type = "GeoRasterRenderingPipeline"
    batch_size = 4
    device = torch.device('cuda')
    train_dataset = RenderFormerDataset(train_data_dir, resolution, max_num_tris, pipeline_type)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        sampler=None,
        num_workers=0,
        pin_memory=True
    )

    for batch_idx, data in enumerate(train_dataloader):
        triangles = data['triangles'].to(device)
        mask = data['mask'].to(device)
        vn = data['vn'].to(device)
        c2w = data['c2w'].to(device)
        fov = data['fov'].to(device)

        mask = triangle_mask_per_tile(triangles, c2w.reshape(-1, 4, 4), fov.reshape(-1, 1), resolution, ray_generator=RayGenerator().to(device), tile_size=32)
        # print(f"mask: {mask}")
        break