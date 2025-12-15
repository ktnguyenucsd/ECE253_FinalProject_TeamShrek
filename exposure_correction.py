import os
import cv2
import numpy as np

# ------------- CONFIG -------------
INPUT_DIR        = "outputshrek"          # folder with your noisy images
OUT_CLAHE_DIR    = "salt_and_pepper_CLAHE_shrek"    # output folder for CLAHE
OUT_GAMMA_DIR    = "salt_and_pepper_gamma_shrek"    # output folder for adaptive gamma

# which file types to process
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# CLAHE parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID  = (8, 8)

# adaptive gamma target mean luminance
TARGET_MEAN_Y = 0.5
GAMMA_MIN, GAMMA_MAX = 0.4, 2.5
# -----------------------------------


def is_image_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in VALID_EXTS


def apply_clahe_color(img_bgr):
    """
    Apply CLAHE on the luminance channel (Y) in YCrCb.
    This avoids weird color shifts.
    """
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0]

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                            tileGridSize=CLAHE_TILE_GRID)
    Y_clahe = clahe.apply(Y)

    ycrcb[:, :, 0] = Y_clahe
    img_clahe_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return img_clahe_bgr


def apply_adaptive_gamma(img_bgr):
    """
    Your adaptive gamma method applied on Y (luminance) in YCrCb.
    """
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float32) / 255.0

    mean_Y = Y.mean()
    eps = 1e-6

    # compute gamma
    if abs(mean_Y - TARGET_MEAN_Y) < 0.02:
        gamma = 1.0
    else:
        gamma = np.log(TARGET_MEAN_Y + eps) / np.log(mean_Y + eps)

    gamma = np.clip(gamma, GAMMA_MIN, GAMMA_MAX)

    # build LUT
    lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)],
                   dtype="uint8")

    y_new = cv2.LUT((Y * 255.0).astype("uint8"), lut)
    ycrcb[:, :, 0] = y_new

    img_gamma_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return img_gamma_bgr, mean_Y, gamma


def main():
    os.makedirs(OUT_CLAHE_DIR, exist_ok=True)
    os.makedirs(OUT_GAMMA_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(INPUT_DIR) if is_image_file(f))

    print(f"Found {len(files)} images in '{INPUT_DIR}'")

    for i, fname in enumerate(files, 1):
        in_path = os.path.join(INPUT_DIR, fname)
        img = cv2.imread(in_path)

        if img is None:
            print(f"[WARN] Could not read {in_path}, skipping.")
            continue

        # --- CLAHE ---
        img_clahe = apply_clahe_color(img)
        out_clahe_path = os.path.join(OUT_CLAHE_DIR, fname)
        cv2.imwrite(out_clahe_path, img_clahe)

        # --- Adaptive gamma ---
        img_gamma, mean_Y, gamma = apply_adaptive_gamma(img)
        out_gamma_path = os.path.join(OUT_GAMMA_DIR, fname)
        cv2.imwrite(out_gamma_path, img_gamma)

        if i % 10 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] {fname} | mean_Y={mean_Y:.3f}, gamma={gamma:.3f}")

    print("Done! Saved:")
    print(f" - CLAHE images to: {OUT_CLAHE_DIR}")
    print(f" - Gamma-corrected images to: {OUT_GAMMA_DIR}")


if __name__ == "__main__":
    main()
