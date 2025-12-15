import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# =============================
# DEVICE
# =============================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] Using device:", device)

# =============================
# TRANSFORMS (same as training)
# =============================

def get_transforms(train=False):
    trans = []
    if train:
        trans.append(transforms.RandomHorizontalFlip(0.5))
    trans.append(transforms.ToTensor())
    return transforms.Compose(trans)

val_tran = get_transforms(train=False)

# =============================
# HELPER: convert tensor → image
# =============================

def im_convert(tensor):
    image = tensor.clone().detach().cpu().numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return image

# =============================
# LOAD TRAINED MODEL
# =============================
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

num_classes = 2  # background + person

print("[INFO] Loading model...")
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# load saved model weights (if you saved any)
if os.path.exists("trained_frcnn.pth"):
    print("[INFO] Loading trained weights...")
    model.load_state_dict(torch.load("trained_frcnn.pth", map_location=device))

model.to(device)
model.eval()

# =============================
# INPUT + OUTPUT FOLDERS
# =============================

HOME = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(HOME, "images_with_person")   # folder with images
OUTPUT_DIR = os.path.join(HOME, "output")              # save results here
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[INFO] Input folder:", INPUT_DIR)
print("[INFO] Output folder:", OUTPUT_DIR)

# =============================
# PROCESS ALL IMAGES
# =============================

image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))])
print(f"[INFO] Found {len(image_files)} images to process.")

for idx, fname in enumerate(image_files, start=1):
    print(f"[{idx}/{len(image_files)}] Processing {fname}...")

    img_path = os.path.join(INPUT_DIR, fname)
    img = cv2.imread(img_path)

    if img is None:
        print("  [WARNING] Failed to read image:", fname)
        continue

    # convert for model
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor_img = val_tran(pil_img).to(device)

    # inference
    with torch.no_grad():
        outputs = model([tensor_img])

    scores = outputs[0]["scores"].detach().cpu().numpy()
    boxes  = outputs[0]["boxes"].detach().cpu().numpy()

    # apply confidence threshold
    mask = scores > 0.75
    scores = scores[mask]
    boxes  = boxes[mask]

    # draw detections
    for i in range(len(scores)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 3)

        # optional: draw score
        cv2.putText(img, f"{scores[i]:.2f}", (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # save output image
    out_path = os.path.join(OUTPUT_DIR, fname)
    cv2.imwrite(out_path, img)

print("\n[INFO] Done! All output images saved in:", OUTPUT_DIR)
