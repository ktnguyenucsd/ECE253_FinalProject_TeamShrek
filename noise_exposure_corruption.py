import os
import cv2
import numpy as np

# =================================
# CONFIG
# =================================

HOME = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(HOME, "images_with_person")
OUTPUT_DIR = os.path.join(HOME, "images_with_person_corrupted")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# for reproducibility (you can change/remove)
np.random.seed(42)

# ranges for random corruption
GAMMA_MIN, GAMMA_MAX = 0.6, 1.6      # exposure variation
NOISE_MIN, NOISE_MAX = 0.0, 0.08     # Gaussian noise std in [0,1] space


def apply_gamma(img_float, gamma):
    """
    img_float: image in [0,1], float32
    gamma: scalar > 0
    """
    # avoid issues at 0
    img_corr = np.power(img_float, gamma)
    return img_corr


def apply_gaussian_noise(img_float, sigma):
    """
    img_float: image in [0,1], float32
    sigma: std dev of Gaussian noise
    """
    if sigma <= 0:
        return img_float
    noise = np.random.normal(0.0, sigma, img_float.shape).astype(np.float32)
    img_noisy = img_float + noise
    img_noisy = np.clip(img_noisy, 0.0, 1.0)
    return img_noisy


def main():
    image_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print(f"[INFO] Found {len(image_files)} images in {INPUT_DIR}")
    print(f"[INFO] Saving corrupted versions to {OUTPUT_DIR}")

    for idx, fname in enumerate(image_files, start=1):
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)

        img_bgr = cv2.imread(in_path)
        if img_bgr is None:
            print(f"[WARNING] Could not read {fname}, skipping.")
            continue

        # convert to float [0,1]
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        # sample random gamma and noise sigma
        gamma = np.random.uniform(GAMMA_MIN, GAMMA_MAX)
        sigma = np.random.uniform(NOISE_MIN, NOISE_MAX)

        # apply exposure (gamma) + noise
        img_corr = apply_gamma(img, gamma)
        img_corr = apply_gaussian_noise(img_corr, sigma)

        # back to uint8 BGR for saving
        img_corr = (img_corr * 255.0).round().astype(np.uint8)
        img_corr_bgr = cv2.cvtColor(img_corr, cv2.COLOR_RGB2BGR)

        cv2.imwrite(out_path, img_corr_bgr)

        print(f"[{idx}/{len(image_files)}] {fname}  -> gamma={gamma:.2f}, sigma={sigma:.3f}")

    print("\n[INFO] Done. Corrupted images written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
