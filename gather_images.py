import os
import shutil

# Base folder where all your original folders live
# Example: raw_images/folder1, raw_images/folder2, ...
BASE_DIR = "raw_images"

# Target folder where all images will be copied and renamed
TARGET_DIR = "input_images"

# Which file extensions to treat as images
EXTS = (".jpg", ".jpeg", ".png")

def main():
    # make sure target exists
    os.makedirs(TARGET_DIR, exist_ok=True)

    # collect all image paths from all subfolders
    image_paths = []
    for root, dirs, files in os.walk(BASE_DIR):
        for fname in files:
            if fname.lower().endswith(EXTS):
                full_path = os.path.join(root, fname)
                image_paths.append(full_path)

    print(f"Found {len(image_paths)} images under {BASE_DIR}")

    # sort for reproducible order
    image_paths.sort()

    # copy & rename sequentially
    for idx, src_path in enumerate(image_paths, start=1):
        # preserve extension
        ext = os.path.splitext(src_path)[1].lower()
        new_name = f"{idx}{ext}"
        dst_path = os.path.join(TARGET_DIR, new_name)

        print(f"Copying {src_path} -> {dst_path}")
        shutil.copy2(src_path, dst_path)

    print("\nDone! All images collected into:", TARGET_DIR)

if __name__ == "__main__":
    main()
