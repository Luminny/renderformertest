import torch

from renderformer.models.geo_raster import GeoRaster
from renderformer.utils.ray_generator import RayGenerator
from renderformer.utils.transform import trans_to_cam_coord


class GeoRasterRenderingPipeline:
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

    def render(
        self,
        triangles,
        mask,
        vn,
        c2w,
        fov,
        resolution: int = 512,
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

        # Generate rays
        # rays_o, rays_d: [bs, nv, H, W, 3]
        rays_o, rays_d = self.ray_generator(c2w_for_view_tf, fov / 180. * torch.pi, resolution)

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
        return self.render(*args, **kwargs)
