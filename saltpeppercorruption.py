import os
import cv2
import numpy as np

# =================================
# CONFIG
# =================================

HOME = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(HOME, "images_with_person")
OUTPUT_DIR = os.path.join(HOME, "images_with_person_saltpepper")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Random ranges for corruptions
GAMMA_RANGE = (0.5, 2.0)            # strong exposure shift
GAUSS_SIGMA_RANGE = (0.03, 0.12)    # fairly strong Gaussian noise

SALT_RANGE = (0.01, 0.05)           # 1–5% of pixels become white
PEPPER_RANGE = (0.01, 0.05)         # 1–5% of pixels become black


# ---------------------------------
# Salt and Pepper Noise
# ---------------------------------
def add_salt_pepper(img, p_salt=0.02, p_pepper=0.02):
    """
    img: RGB image in [0,1]
    """
    noisy = img.copy()
    H, W, C = img.shape

    # Salt mask
    num_salt = int(p_salt * H * W)
    coords = (np.random.randint(0, H, num_salt), np.random.randint(0, W, num_salt))
    noisy[coords] = 1.0     # white pixels

    # Pepper mask
    num_pepper = int(p_pepper * H * W)
    coords = (np.random.randint(0, H, num_pepper), np.random.randint(0, W, num_pepper))
    noisy[coords] = 0.0     # black pixels

    return noisy


# ---------------------------------
# Gamma exposure
# ---------------------------------
def apply_gamma(img, gamma):
    img_gamma = np.power(img, gamma)
    return np.clip(img_gamma, 0., 1.)


# ---------------------------------
# Gaussian noise
# ---------------------------------
def apply_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img + noise
    return np.clip(out, 0., 1.)


# =================================
# MAIN
# =================================

def main():
    image_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print(f"[INFO] Found {len(image_files)} images.")
    print(f"[INFO] Saving salt-pepper corrupted images → {OUTPUT_DIR}\n")

    for i, fname in enumerate(image_files, 1):
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)

        img_bgr = cv2.imread(in_path)
        if img_bgr is None:
            print(f"[WARNING] Could not read {fname}, skipping.")
            continue

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Random parameters
        gamma = np.random.uniform(*GAMMA_RANGE)
        sigma = np.random.uniform(*GAUSS_SIGMA_RANGE)
        p_salt = np.random.uniform(*SALT_RANGE)
        p_pepper = np.random.uniform(*PEPPER_RANGE)

        # Apply corruption
        img_cor = apply_gamma(img, gamma)
        img_cor = apply_gaussian_noise(img_cor, sigma)
        img_cor = add_salt_pepper(img_cor, p_salt, p_pepper)

        # Back to uint8 BGR
        img_out = (img_cor * 255).astype(np.uint8)
        img_out_bgr = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, img_out_bgr)

        print(f"[{i}/{len(image_files)}] {fname}  → gamma={gamma:.2f}, "
              f"sigma={sigma:.3f}, salt={p_salt:.3f}, pepper={p_pepper:.3f}")

    print("\n[INFO] Done — corrupted images saved.")


if __name__ == "__main__":
    main()
