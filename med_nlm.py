import os
import glob
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ==========================
# CONFIG
# ==========================

HOME = Path(__file__).resolve().parent
INPUT_DIR = HOME / "DED/wiener_restored"        # noisy images
OUTPUT_DIR = HOME / "output_wiener"      # denoised output

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] Using device:", device)

# NLM parameters (tune these)
PATCH_SIZE = 3        # patch window (odd)
SEARCH_SIZE = 11      # search window (odd, e.g. 7, 11)
H_PARAM = 0.1         # filtering parameter (higher = more smoothing)


# ==========================
# GPU NLM IMPLEMENTATION
# (single-channel, PyTorch)
# ==========================

def nlm_denoise_single_channel_gpu(img_np,
                                   patch_size=PATCH_SIZE,
                                   search_size=SEARCH_SIZE,
                                   h=H_PARAM,
                                   device=device):
    """
    img_np: 2D numpy array in [0, 255] (uint8)
    returns: 2D uint8 numpy array
    """
    # convert to [0,1] float and move to GPU
    img = torch.from_numpy(img_np).float().to(device) / 255.0  # H, W
    img = img.unsqueeze(0).unsqueeze(0)  # 1,1,H,W

    H, W = img.shape[2], img.shape[3]

    patch_rad = patch_size // 2
    search_rad = search_size // 2

    # pad for patch extraction (reflect padding)
    img_pad = F.pad(img, (patch_rad, patch_rad, patch_rad, patch_rad), mode="reflect")

    # central patches for all pixels
    # shape: (1, patch_area, H*W)
    patches = F.unfold(img_pad, kernel_size=patch_size)  # 1, P, H*W
    patches = patches.squeeze(0).transpose(0, 1)         # (H*W, P)

    # output accumulators
    result = torch.zeros(1, H * W, device=device)
    norm = torch.zeros(1, H * W, device=device)

    # loop over search window offsets
    for dy in range(-search_rad, search_rad + 1):
        for dx in range(-search_rad, search_rad + 1):
            # shift image by (dy, dx)
            img_shift = torch.roll(img, shifts=(dy, dx), dims=(2, 3))

            # get patches of shifted image
            img_shift_pad = F.pad(img_shift, (patch_rad, patch_rad, patch_rad, patch_rad), mode="reflect")
            patches_shift = F.unfold(img_shift_pad, kernel_size=patch_size)  # 1, P, H*W
            patches_shift = patches_shift.squeeze(0).transpose(0, 1)         # (H*W, P)

            # squared distance between patches
            dist2 = ((patches - patches_shift) ** 2).mean(dim=1)  # (H*W,)

            # weight
            weights = torch.exp(-dist2 / (h * h))                  # (H*W,)
            weights = weights.unsqueeze(0)                         # 1, H*W

            # center pixel values from shifted image
            center_vals = img_shift.view(1, -1)                    # 1, H*W

            # accumulate
            result += weights * center_vals
            norm += weights

    # normalize
    out = (result / norm).view(1, 1, H, W)
    out = out.clamp(0.0, 1.0)

    # back to CPU numpy uint8
    out_np = (out.squeeze().detach().cpu().numpy() * 255.0).astype(np.uint8)
    return out_np


def nlm_denoise_image_gpu_after_median(pil_img, median_ksize=3):
    """
    Apply median filter first, then NLM-like denoising on GPU.
    Returns a PIL image.
    """
    # PIL -> numpy uint8, RGB
    img_np = np.array(pil_img)  # H, W, 3 (uint8)

    # ---- Step 1: median filter (per-channel via OpenCV) ----
    # cv2.medianBlur works on multi-channel images directly
    med_np = cv2.medianBlur(img_np, ksize=median_ksize)

    # ---- Step 2: NLM on median-filtered image ----
    if med_np.ndim == 2:  # grayscale
        den = nlm_denoise_single_channel_gpu(med_np)
        return Image.fromarray(den)

    elif med_np.ndim == 3:  # color: H, W, 3
        channels = []
        for c in range(3):
            den_c = nlm_denoise_single_channel_gpu(med_np[:, :, c])
            channels.append(den_c)
        den_rgb = np.stack(channels, axis=2)
        return Image.fromarray(den_rgb)

    else:
        raise ValueError("Unsupported image shape: {}".format(med_np.shape))


# ==========================
# PROCESS FOLDER
# ==========================

extensions = ("*.jpg", "*.jpeg", "*.png")
image_paths = []
for ext in extensions:
    image_paths.extend(glob.glob(str(INPUT_DIR / ext)))

image_paths = sorted(image_paths)

print(f"[INFO] Found {len(image_paths)} images in {INPUT_DIR}")

for img_path in tqdm(image_paths, desc="Median + NLM (GPU)"):
    img_path = Path(img_path)
    out_path = OUTPUT_DIR / img_path.name

    # load noisy image
    img = Image.open(img_path).convert("RGB")

    # median filter --> NLM GPU
    den_img = nlm_denoise_image_gpu_after_median(img, median_ksize=3)

    # save result
    den_img.save(out_path)

print(f"\n[INFO] Done. Median+NLM-denoised images saved in: {OUTPUT_DIR}")

