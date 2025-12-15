import os
import json
import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ==========================
# CONFIG
# ==========================

HOME = os.path.dirname(os.path.abspath(__file__))

IMAGES_DIR = os.path.join(HOME, "images_with_person")      # eval images
GT_ANN_PATH = os.path.join(HOME, "my_new_annotations.json")# ground-truth COCO JSON
PRED_ANN_PATH = os.path.join(HOME, "predictions_frcnn.json")  # where we'll save predictions

NUM_CLASSES = 2   # background + person
SCORE_THRESH = 0.05  # keep low here, COCOeval handles precision-recall

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] Using device:", device)


# ==========================
# MODEL + TRANSFORMS
# ==========================

def get_val_transforms():
    return transforms.Compose([
        transforms.ToTensor()
    ])

print("[INFO] Loading model...")
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

# load your trained weights
weights_path = os.path.join(HOME, "trained_frcnn.pth")
if os.path.exists(weights_path):
    print(f"[INFO] Loading trained weights from {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
else:
    print("[WARNING] trained_frcnn.pth not found, using COCO-pretrained weights only")

model.to(device)
model.eval()

val_transform = get_val_transforms()


# ==========================
# BUILD PREDICTIONS JSON
# ==========================

print("[INFO] Loading ground-truth COCO annotations...")
coco_gt = COCO(GT_ANN_PATH)

img_ids = sorted(coco_gt.getImgIds())
print(f"[INFO] Number of images in JSON: {len(img_ids)}")

all_detections = []

with torch.no_grad():
    for idx, img_id in enumerate(img_ids, start=1):
        img_info = coco_gt.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        img_path = os.path.join(IMAGES_DIR, file_name)

        print(f"[{idx}/{len(img_ids)}] Processing {file_name} (image_id={img_id})")

        # load image
        img = Image.open(img_path).convert("RGB")
        img_tensor = val_transform(img).to(device)

        # run model
        outputs = model([img_tensor])
        outputs = outputs[0]

        boxes = outputs["boxes"].detach().cpu().numpy()
        scores = outputs["scores"].detach().cpu().numpy()
        labels = outputs["labels"].detach().cpu().numpy()

        # keep only person class (label == 1) and above score threshold
        keep = (scores >= SCORE_THRESH) & (labels == 1)
        boxes = boxes[keep]
        scores = scores[keep]

        # convert boxes from [x1,y1,x2,y2] to [x,y,w,h] as COCO expects
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1

            det = {
                "image_id": int(img_id),
                "category_id": 1,           # person
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(score)
            }
            all_detections.append(det)

# save predictions
print(f"[INFO] Saving detections to {PRED_ANN_PATH} ...")
with open(PRED_ANN_PATH, "w") as f:
    json.dump(all_detections, f)

print("[INFO] Detections JSON saved.")


# ==========================
# COCO EVALUATION (mAP)
# ==========================

print("[INFO] Running COCO evaluation...")

coco_dt = coco_gt.loadRes(PRED_ANN_PATH)

coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

print("\n[INFO] Done. mAP (AP@[0.50:0.95]) is the first number above.")

# ---- Save summary to file ----
summary_path = os.path.join(HOME, "map_results.txt")
with open(summary_path, "w") as f:
    for i, metric in enumerate(coco_eval.stats):
        f.write(f"{i}: {metric}\n")

print("[INFO] mAP results saved to:", summary_path)