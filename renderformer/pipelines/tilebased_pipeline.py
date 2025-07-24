import torch

from renderformer.models.geo_raster import GeoRaster
from renderformer.utils.ray_generator import RayGenerator
from renderformer.utils.transform import trans_to_cam_coord
from einops import rearrange


class TileBasedRenderingPipeline:
    def __init__(self, model: GeoRaster):
        self.model = model
        self.config = model.config if hasattr(model, 'config') else model.module.config
        self.ray_generator = RayGenerator().to(self._get_model_device())

    @classmethod
    def from_pretrained(cls, model_id: str):
        model = GeoRaster.from_pretrained(model_id)
        model.eval()
        return cls(model)

    def _get_model_device(self):
        """Get device from model, handling DataParallel wrapper"""
        if hasattr(self.model, 'module'):
            # DataParallel wrapped model
            return next(self.model.module.parameters()).device
        else:
            # Regular model
            return next(self.model.parameters()).device

    @property
    def device(self):
        return self._get_model_device()

    def to(self, device: torch.device):
        self.model.to(device)
        self.ray_generator.to(device)
        return self

    def render_data(
        self, 
        data, 
        resolution: int = 512, 
        torch_dtype: torch.dtype = torch.float16, 
        device: torch.device = None
        ):
        
        triangles = data['triangles'].to(device) # [batch_num, tile_num, max_num_tris_per_tile, 3, 3]
        mask = data['mask'].to(device)           # [batch_num, tile_num, max_num_tris_per_tile]
        vn = data['vn'].to(device)               # [batch_num, tile_num, max_num_tris_per_tile, 3, 3]
        c2w = data['c2w'].to(device)             # [batch_num, view_num, 4, 4]
        fov = data['fov'].to(device)             # [batch_num, view_num, 1]

        bn, tile_num, max_num_tris_per_tile = triangles.shape[:3]
        triangles = triangles.reshape(-1, max_num_tris_per_tile, 3, 3)
        mask = mask.reshape(-1, max_num_tris_per_tile)
        vn = vn.reshape(-1, max_num_tris_per_tile, 3, 3)
        # c2w: [bs, nv, 4, 4] -> [bs*tile_num, nv, 4, 4] by repeating
        c2w_repeated = torch.repeat_interleave(c2w, tile_num, dim=0)

        assert self.config.turn_to_cam_coord, "Tile-based rendering pipeline only supports turning to camera coordinate"

        # Generate rays
        # rays_o, rays_d: [bs, nv, H, W, 3]
        c2w_for_view_tf = torch.eye(4, device=device, dtype=c2w.dtype).repeat(c2w.shape[0], c2w.shape[1], 1, 1)
        rays_o, rays_d = self.ray_generator(c2w_for_view_tf, fov / 180. * torch.pi, resolution)
        tile_rays_d = rearrange(rays_d, 'b nv (h t1) (w t2) c -> (b h w) nv t1 t2 c', b=bn, t1=self.config.tile_size, t2=self.config.tile_size)
        tile_rays_o = torch.repeat_interleave(rays_o, tile_num, dim=0)
        
        rendered_tiles = self.render(
            triangles=triangles,
            mask=mask,
            vn=vn,
            c2w=c2w_repeated,
            rays_o=tile_rays_o,
            rays_d=tile_rays_d,
            torch_dtype=torch_dtype,
        ) # [bn*tile_num, view_num, tile_resolution, tile_resolution, 3]
        print(f"rendered_tiles: {rendered_tiles.shape}")

        rendered_imgs = rearrange(
            rendered_tiles, '(b h1 w1) nv t1 t2 c -> b nv (h1 t1) (w1 t2) c', 
            b=bn, h1=resolution//self.config.tile_size, w1=resolution//self.config.tile_size, t1=self.config.tile_size, t2=self.config.tile_size)

        return rendered_imgs

    def render(
        self,
        triangles,
        mask,
        vn,
        c2w,
        rays_o,
        rays_d,
        torch_dtype: torch.dtype = torch.float16
    ):
        """
        Render images using the RenderFormer model
        
        Args:
            model: RenderFormer model
            config: RenderFormerConfig object
            triangles: Triangle data tensor [bs, num_tris, 3, 3] - vertices of triangles
            mask: Mask data tensor [bs, num_tris] - boolean mask indicating valid triangles
            vn: Vertex normal vectors tensor [bs, num_tris, 3, 3] - normal vectors of triangles
            c2w: Camera-to-world matrix tensor [bs, num_views, 4, 4]
            fov: Field of view tensor [bs, num_views, 1] - in degrees
            resolution: Render resolution (default: 512)
            torch_dtype: PyTorch dtype for inference, default is torch.float16

        Returns:
            torch.Tensor: Rendered HDR image tensor [bs, num_views, H, W, 3]
        """

        bs, nv = c2w.shape[0], c2w.shape[1]

        # Handle view transformation
        if self.config.turn_to_cam_coord:
            # Reshape for transformation
            # c2w: [bs, nv, 4, 4] -> [bs*nv, 4, 4]
            # triangles: [bs, num_tris, 3, 3] -> [bs*nv, num_tris, 3, 3] by repeating
            c2w_reshaped = c2w.reshape(-1, 4, 4)
            triangles_repeated = torch.repeat_interleave(triangles, nv, dim=0)
            
            tris_for_view_tf, c2w_for_view_tf, _ = trans_to_cam_coord(
                c2w_reshaped,
                triangles_repeated
            )
            # Reshape back
            # c2w_for_view_tf: [bs*nv, 4, 4] -> [bs, nv, 4, 4]
            # tris_for_view_tf: [bs*nv, num_tris, 3, 3] -> [bs, nv, num_tris, 3, 3]
            c2w_for_view_tf = c2w_for_view_tf.reshape(bs, nv, 4, 4)
            tris_for_view_tf = tris_for_view_tf.reshape(bs, nv, -1, 3, 3)
        else:
            # Expand triangles for each view
            # triangles: [bs, num_tris, 3, 3] -> [bs, nv, num_tris, 3, 3]
            tris_for_view_tf = triangles.unsqueeze(1).expand(-1, nv, -1, -1, -1)
            c2w_for_view_tf = c2w

        # Set precision
        assert torch_dtype in [torch.bfloat16, torch.float16, torch.float32], f"Invalid precision: {torch_dtype}\nChoose from: torch.bfloat16, torch.float16, torch.float32"
        tf32_view_tf = torch_dtype == torch.bfloat16 or torch_dtype == torch.float16

        # Perform rendering
        # Flatten triangles: [bs, num_tris, 3, 3] -> [bs, num_tris*9]
        # Flatten vn: [bs, num_tris, 3, 3] -> [bs, num_tris*9]
        # Flatten tri_vpos_view_tf: [bs, nv, num_tris, 3, 3] -> [bs, nv, num_tris*9]
        # with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch_dtype):
        with torch.autocast(device_type=self.device.type, dtype=torch_dtype):
            rendered_imgs = self.model(
                triangles.reshape(bs, -1, 9),
                mask,
                vn.reshape(bs, -1, 9),
                rays_o=rays_o,
                rays_d=rays_d,
                tri_vpos_view_tf=tris_for_view_tf.reshape(bs, nv, -1, 9),
                tf32_view_tf=tf32_view_tf,
            )

        # Process output
        # rendered_imgs: [bs, nv, C, H, W] -> [bs, nv, H, W, C]
        rendered_imgs = rendered_imgs.permute(0, 1, 3, 4, 2)

        return rendered_imgs

    def __call__(self, *args, **kwargs):
        return self.render_data(*args, **kwargs)
