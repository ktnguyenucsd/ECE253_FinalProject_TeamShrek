import os
import cv2
import numpy as np

# ============================
# CONFIG
# ============================
INPUT_DIR = "C:/Users/khoa/person_seg_project/NLM + Exposure Correction/salt_and_pepper_CLAHE"   # <-- change this
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_image_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in VALID_EXTS


def luminance_metrics(img_bgr):
    """
    Compute luminance-based metrics on Y channel (YCrCb).

    Returns:
        mean_Y       : mean luminance in [0,1]
        abs_diff     : |mean_Y - 0.5|
        std_Y        : standard deviation of luminance in [0,1]
        entropy_Y    : entropy of luminance histogram (bits)
    """
    # Convert to YCrCb and extract Y
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float32) / 255.0  # normalize to [0,1]

    # Mean and std
    mean_Y = float(Y.mean())
    std_Y = float(Y.std())
    abs_diff = abs(mean_Y - 0.5)

    # Histogram entropy (on 256 bins in [0,1] -> 0..255)
    hist, _ = np.histogram((Y * 255).astype(np.uint8),
                           bins=256, range=(0, 256))
    p = hist.astype(np.float64)
    p /= p.sum() + 1e-12  # avoid division by zero

    # Only non-zero probabilities contribute to entropy
    p_nonzero = p[p > 0]
    entropy_Y = float(-(p_nonzero * np.log2(p_nonzero)).sum())

    return mean_Y, abs_diff, std_Y, entropy_Y


def main():
    files = sorted(f for f in os.listdir(INPUT_DIR) if is_image_file(f))
    print(f"[INFO] Found {len(files)} images in '{INPUT_DIR}'\n")

    all_means = []
    all_absdiff = []
    all_stds = []
    all_ents = []

    for i, fname in enumerate(files, 1):
        path = os.path.join(INPUT_DIR, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Could not read {path}, skipping.")
            continue

        mean_Y, abs_diff, std_Y, ent_Y = luminance_metrics(img)

        all_means.append(mean_Y)
        all_absdiff.append(abs_diff)
        all_stds.append(std_Y)
        all_ents.append(ent_Y)

        print(f"[{i}/{len(files)}] {fname}")
        print(f"   mean_Y      = {mean_Y:.4f}")
        print(f"   |mean_Y-0.5|= {abs_diff:.4f}")
        print(f"   std_Y       = {std_Y:.4f}")
        print(f"   entropy_Y   = {ent_Y:.4f}\n")

    if all_means:
        print("=== AVERAGES OVER FOLDER ===")
        print(f"mean_Y (avg)       = {np.mean(all_means):.4f}")
        print(f"|mean_Y-0.5| (avg) = {np.mean(all_absdiff):.4f}")
        print(f"std_Y (avg)        = {np.mean(all_stds):.4f}")
        print(f"entropy_Y (avg)    = {np.mean(all_ents):.4f}")


if __name__ == "__main__":
    main()
