import os
import json
import urllib.request
import shutil

import numpy as np
import cv2
import torch

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from pycocotools import mask as mask_utils


# ==============================
# CONFIG
# ==============================

HOME = os.path.dirname(os.path.abspath(__file__))

IMAGES_DIR = os.path.join(HOME, "input_images")
MASKS_DIR = os.path.join(HOME, "output_masks")
FILTERED_IMAGES_DIR = os.path.join(HOME, "images_with_person")

OUTPUT_JSON_PATH = os.path.join(HOME, "my_new_annotations.json")

WEIGHTS_DIR = os.path.join(HOME, "weights")
CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "sam_vit_h_4b8939.pth")

YOLO_MODEL_NAME = "yolov8m.pt"
PERSON_CATEGORY_ID = 1
YOLO_CONF_THRESH = 0.4


# ==============================
# HELPER: download SAM checkpoint
# ==============================

def ensure_sam_checkpoint():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    if not os.path.isfile(CHECKPOINT_PATH):
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        print(f"[INFO] SAM checkpoint not found, downloading:\n  {url}")
        urllib.request.urlretrieve(url, CHECKPOINT_PATH)
    else:
        print(f"[INFO] Found SAM checkpoint at: {CHECKPOINT_PATH}")


# ==============================
# PROCESS ONE IMAGE
# ==============================

def process_image_to_masks_and_anns(
    img_path,
    image_id,
    start_ann_id,
    sam_predictor,
    yolo_model,
    category_id=1,
    conf_thresh=0.4
):
    """
    Returns:
        union_mask: (H, W) bool or None
        person_masks: list[(H,W) bool] or None
        coco_ann_list: list[annotation dict] with ids starting at start_ann_id
        H, W: image shape
    """
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load: {img_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = image_rgb.shape[:2]

    # ---- YOLO person detection ----
    yolo_result = yolo_model(image_rgb, verbose=False)[0]

    boxes = []
    for box, cls, conf in zip(
        yolo_result.boxes.xyxy.cpu().numpy(),
        yolo_result.boxes.cls.cpu().numpy(),
        yolo_result.boxes.conf.cpu().numpy()
    ):
        if int(cls) == 0 and conf > conf_thresh:
            boxes.append(box.tolist())

    # no persons → nothing to do
    if len(boxes) == 0:
        return None, None, [], H, W

    # ---- SAM segmentation ----
    sam_predictor.set_image(image_rgb)
    person_masks = []
    coco_ann_list = []
    ann_id = start_ann_id

    for box in boxes:
        input_box = np.array(box, dtype=np.float32)

        masks, scores, _ = sam_predictor.predict(
            box=input_box[None, :],
            multimask_output=False
        )

        mask = masks[0].astype(bool)
        person_masks.append(mask)

        # COCO RLE
        binary = mask.astype(np.uint8)
        rle = mask_utils.encode(np.asfortranarray(binary))
        rle["counts"] = rle["counts"].decode("ascii")

        area = float(mask_utils.area(rle))
        bbox = mask_utils.toBbox(rle).tolist()

        ann = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "iscrowd": 0,
            "segmentation": rle,
            "area": area,
            "bbox": bbox
        }
        coco_ann_list.append(ann)
        ann_id += 1

    union_mask = np.any(np.stack(person_masks, axis=0), axis=0)

    return union_mask, person_masks, coco_ann_list, H, W


# ==============================
# MAIN
# ==============================

def main():

    if not os.path.isdir(IMAGES_DIR):
        raise FileNotFoundError(f"Missing folder: {IMAGES_DIR}")

    os.makedirs(MASKS_DIR, exist_ok=True)
    os.makedirs(FILTERED_IMAGES_DIR, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print(f"[INFO] Found {len(image_files)} images in {IMAGES_DIR}")

    ensure_sam_checkpoint()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # load models
    print("[INFO] Loading SAM model...")
    sam = sam_model_registry["vit_h"](checkpoint=CHECKPOINT_PATH).to(device)
    sam_predictor = SamPredictor(sam)

    print("[INFO] Loading YOLO model:", YOLO_MODEL_NAME)
    yolo_model = YOLO(YOLO_MODEL_NAME)

    images_list = []
    annotations_list = []
    categories_list = [{
        "id": PERSON_CATEGORY_ID,
        "name": "person",
        "supercategory": "person"
    }]

    image_id = 1   # will be used ONLY for images with at least one person
    ann_id = 1     # will be used ONLY for actual annotations

    for fname in image_files:
        img_path = os.path.join(IMAGES_DIR, fname)
        print(f"\n[PROCESSING] {fname}")

        union_mask, person_masks, coco_anns, H, W = process_image_to_masks_and_anns(
            img_path=img_path,
            image_id=image_id,       # proposed ID if person exists
            start_ann_id=ann_id,
            sam_predictor=sam_predictor,
            yolo_model=yolo_model,
            category_id=PERSON_CATEGORY_ID,
            conf_thresh=YOLO_CONF_THRESH
        )

        # ---- PERSON FOUND ----
        if union_mask is not None:

            # Save mask
            mask_png = (union_mask.astype(np.uint8) * 255)
            mask_name = os.path.splitext(fname)[0] + "_mask.png"
            mask_path = os.path.join(MASKS_DIR, mask_name)
            cv2.imwrite(mask_path, mask_png)
            print(f"  Saved mask: {mask_name}")

            # Copy original image into filtered folder
            dst_image_path = os.path.join(FILTERED_IMAGES_DIR, fname)
            shutil.copy2(img_path, dst_image_path)
            print(f"  Copied image to: {dst_image_path}")

            # Add annotations (with correct, continuous IDs)
            annotations_list.extend(coco_anns)
            ann_id += len(coco_anns)   # <<< increment by how many we just created

            # Add IMAGE to JSON (ONLY IF PERSON FOUND)
            images_list.append({
                "id": image_id,
                "license": 0,
                "file_name": fname,
                "width": W,
                "height": H,
                "date_captured": "2025-12-10"
            })

            image_id += 1  # <<< next image with a person gets the next ID

        # ---- NO PERSON FOUND ----
        else:
            print("  No persons detected → skipping image entirely (no JSON, no mask, no copy).")

    # ---- SAVE JSON ----
    coco_out = {
        "info": {
            "year": "2025",
            "version": "1.0",
            "description": "Auto-generated person masks with YOLO+SAM",
            "contributor": "",
            "url": "",
            "date_created": "2025-12-10"
        },
        "images": images_list,
        "annotations": annotations_list,
        "categories": categories_list
    }

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(coco_out, f)

    print("\n================================")
    print("DONE!")
    print(f"Images included in JSON: {len(images_list)}  (IDs 1..{len(images_list)})")
    print(f"Total annotations: {len(annotations_list)}  (IDs 1..{len(annotations_list)})")
    print(f"Masks saved in: {MASKS_DIR}")
    print(f"Filtered images saved in: {FILTERED_IMAGES_DIR}")
    print(f"JSON saved at: {OUTPUT_JSON_PATH}")
    print("================================")


if __name__ == "__main__":
    main()
