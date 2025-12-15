import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt

# -----------------------------
# CONFIG: Set your directories
# -----------------------------
GT_DIR = "best_images/images_with_person"
METHOD1_DIR = "salt_and_pepper_CLAHE"
METHOD2_DIR = "salt_and_pepper_gamma"

# Image extension (change if PNG/JPG differ)
EXT = ".png"   # or ".jpg"

# -----------------------------
# Metric functions
# -----------------------------
def mse(img1, img2):
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

def psnr(img1, img2):
    mse_val = mse(img1, img2)
    if mse_val == 0:
        return float("inf")
    return 20 * log10(255.0 / sqrt(mse_val))

def compute_ssim(img1, img2):
    # ssim from skimage expects grayscale OR multichannel=True
    return ssim(img1, img2, data_range=255, channel_axis=2)

# -----------------------------
# Helper: Read images safely
# -----------------------------
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -----------------------------
# PROCESSING FUNCTION
# -----------------------------
def evaluate_method(denoise_dir):
    mse_list = []
    psnr_list = []
    ssim_list = []

    for filename in os.listdir(GT_DIR):
        if not filename.endswith(EXT):
            continue
        
        gt_path = os.path.join(GT_DIR, filename)
        denoise_path = os.path.join(denoise_dir, filename)

        if not os.path.exists(denoise_path):
            print(f"WARNING: {denoise_path} does not exist, skipping.")
            continue

        gt = load_image(gt_path)
        dn = load_image(denoise_path)

        # Ensure same size
        if gt.shape != dn.shape:
            print(f"Size mismatch in {filename}, resizing denoised image.")
            dn = cv2.resize(dn, (gt.shape[1], gt.shape[0]))

        # Compute metrics
        mse_val = mse(gt, dn)
        psnr_val = psnr(gt, dn)
        ssim_val = compute_ssim(gt, dn)

        mse_list.append(mse_val)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    return mse_list, psnr_list, ssim_list


# -----------------------------
# RUN EVALUATION
# -----------------------------
print("\nEvaluating Method 1...")
m1_mse, m1_psnr, m1_ssim = evaluate_method(METHOD1_DIR)

print("\nEvaluating Method 2...")
m2_mse, m2_psnr, m2_ssim = evaluate_method(METHOD2_DIR)


# -----------------------------
# PRINT SUMMARY
# -----------------------------
def summarize(name, mse_list, psnr_list, ssim_list):
    print(f"\n===== {name} =====")
    print(f"Images evaluated: {len(mse_list)}")
    print(f"Average MSE:  {np.mean(mse_list):.4f}")
    print(f"Average PSNR: {np.mean(psnr_list):.4f} dB")
    print(f"Average SSIM: {np.mean(ssim_list):.4f}")

summarize("Method 1", m1_mse, m1_psnr, m1_ssim)
summarize("Method 2", m2_mse, m2_psnr, m2_ssim)
