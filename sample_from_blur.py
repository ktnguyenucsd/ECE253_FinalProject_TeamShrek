import os
import shutil

SOURCE_DIR = "blur_images"      # contains subfolders
TARGET_DIR = "new_images"       # output folder
EXTS = (".jpg", ".jpeg", ".png")
STEP = 120                       # take 1 out of every 10 images

def main():
    os.makedirs(TARGET_DIR, exist_ok=True)

    # recursively collect all image paths
    all_images = []
    for root, dirs, files in os.walk(SOURCE_DIR):
        for f in files:
            if f.lower().endswith(EXTS):
                all_images.append(os.path.join(root, f))

    # sort for consistent order
    all_images.sort()

    print(f"Found {len(all_images)} images inside '{SOURCE_DIR}' (recursive)")

    # sample 1 of every STEP images
    sampled = all_images[::STEP]
    print(f"Sampling {len(sampled)} images (1 per {STEP})\n")

    # copy sampled images to new_images/
    for idx, src in enumerate(sampled, start=1):
        # preserve extension
        ext = os.path.splitext(src)[1].lower()
        dst_name = f"{idx}{ext}"  # rename sequentially if you want
        dst_path = os.path.join(TARGET_DIR, dst_name)

        print(f"Copying {src} --> {dst_path}")
        shutil.copy2(src, dst_path)

    print("\nDone! Sampled images saved to:", TARGET_DIR)

if __name__ == "__main__":
    main()

