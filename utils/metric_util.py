import torch

from kornia.losses import ssim as dssim
import lpips

def mse(image_pred, image_gt, valid_mask=None, reduction="mean"):
    value = (image_pred - image_gt) ** 2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == "mean":
        return torch.mean(value)
    return value


def psnr(image_pred, image_gt, valid_mask=None, reduction="mean"):
    return -10 * torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

def compute_lpips(image_pred, image_gt, kind='alex',
                  shuffle_channels=True, rescale=True):
    """ Computes LPIPS errors between groundtruth and predictions.
        Args: image_pred, image_gt: N x H x W x 3tensors in range [0,1]
        Args: kind denotes the specific  perceptual loss to use, can be alex or vgg.
        Args: shuffle_channels: shuffle the channels of the image before computing pipls
            Returns: LPIPS distance 
    """
    if rescale:
        image_pred = (image_pred - 0.5) * 2
        image_gt = (image_gt - 0.5) * 2

    print(torch.min(image_pred), torch.max(image_pred))
    if shuffle_channels:
        if image_pred is not None:
            image_pred = image_pred.permute(0, 3, 1, 2)
        if image_gt is not None:
            image_gt = image_gt.permute(0, 3, 1, 2)
    loss_fn = lpips.LPIPS(net=kind).cuda()
    return loss_fn.forward(image_pred, image_gt).mean().item()


def ssim(image_pred, image_gt, reduction="mean"):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    dssim_ = dssim.ssim_loss(image_pred, image_gt, window_size=5, max_val=1.0, reduction=reduction)  # dissimilarity in [0, 1]
    return 1 - 2 * dssim_  # in [-1, 1]
