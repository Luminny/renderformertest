
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import imageio
import random
import logging

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
