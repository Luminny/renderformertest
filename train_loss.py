
import torch
import torch.nn.functional as F
import lpips

loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
# loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

def convert_for_lpips(img):
    # clip to [0, 1]
    img = torch.clamp(img, 0, 1)
    # normalize to [-1, 1]
    img = (img - 0.5) * 2
    # [B, V, H, W, 3] -> [B*V, 3, H, W]
    img = img.reshape(-1, *img.shape[-3:]).permute(0, 3, 1, 2)
    return img

def compute_loss(pred_images, gt_images, loss_type='l1'):
    """Compute loss between predicted and ground truth images"""
    if loss_type == 'l1':
        # print(f"pred_images: {pred_images.shape}")
        # print(f"gt_images: {gt_images.shape}")  
        return F.l1_loss(pred_images, gt_images)
    elif loss_type == 'l2':
        return F.mse_loss(pred_images, gt_images)
    elif loss_type == 'l1_w_l2':
        return F.l1_loss(pred_images, gt_images) + F.mse_loss(pred_images, gt_images)
    elif loss_type == 'smooth_l1':
        return F.smooth_l1_loss(pred_images, gt_images)
    elif loss_type == 'lpips_alex':
        return loss_fn_alex(convert_for_lpips(pred_images), convert_for_lpips(gt_images))
    # elif loss_type == 'lpips_vgg':
    #     return loss_fn_vgg(convert_for_lpips(pred_images), convert_for_lpips(gt_images))
    elif loss_type == 'l1_w_lpips_alex':
        return F.l1_loss(pred_images, gt_images) + 0.05 * loss_fn_alex(convert_for_lpips(pred_images), convert_for_lpips(gt_images)).mean()
    elif loss_type == 'l2_w_lpips_alex':
        return 10 * F.mse_loss(pred_images, gt_images) + 0.05 * loss_fn_alex(convert_for_lpips(pred_images), convert_for_lpips(gt_images)).mean()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
