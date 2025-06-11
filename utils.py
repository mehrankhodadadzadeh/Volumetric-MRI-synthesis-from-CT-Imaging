# utils.py

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from monai.metrics import SSIMMetric

# Instantiate SSIM metric once (for 3D volumes, normalized [0,1] range)
_ssim3d = SSIMMetric(
    data_range=1.0,
    spatial_dims=3,
    reduction="mean"
)

def mae_psnr_ssim(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    mask: np.ndarray = None
):
    """
    Compute MAE, PSNR, SSIM between prediction and ground truth,
    optionally under a binary mask.

    Inputs must be 3D numpy arrays (D, H, W) and normalized to [0,1].

    If `mask` is provided, metrics are computed only on foreground voxels.
    """

    # Copy inputs to avoid modifying them in-place
    p = prediction.copy()
    g = ground_truth.copy()

    if mask is not None:
        p[~mask] = 0
        g[~mask] = 0

    # --- MAE ---
    mae = np.mean(np.abs(p - g))

    # --- PSNR ---
    psnr = compare_psnr(g, p, data_range=1.0)

    # --- SSIM via MONAI ---
    # MONAI expects input as 5D: [B, C, D, H, W]
    p_tensor = torch.from_numpy(p[None, None]).float()
    g_tensor = torch.from_numpy(g[None, None]).float()
    with torch.no_grad():
        ssim_val = float(_ssim3d(p_tensor, g_tensor))

    return mae, psnr, ssim_val
