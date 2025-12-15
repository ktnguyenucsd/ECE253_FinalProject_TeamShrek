import os
import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from torchvision import transforms, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ==========================
# DEVICE
# ==========================
cuda_enable = True
if torch.cuda.is_available() and cuda_enable:
    device = torch.device("cuda:0")
    print("[INFO] Using GPU")
else:
    device = torch.device("cpu")
    print("[INFO] Using CPU")


# ==========================
# DATASET
# ==========================

class ImageDataset(Dataset):
    def __init__(self, root, annotation, transforms=None, selected_ids=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)

        all_ids = list(sorted(self.coco.imgs.keys()))

        # Filter if only some image IDs are desired
        if selected_ids is not None:
            selected_ids = set(selected_ids)
            self.ids = [i for i in all_ids if i in selected_ids]
        else:
            self.ids = all_ids

        print(f"[INFO] Dataset initialized with {len(self.ids)} images.")

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_file = self.coco.loadImgs(img_id)[0]["file_name"]
        img_path = os.path.join(self.root, img_file)
        img = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, areas = [], []
        for ann in anns:
            x_min = ann['bbox'][0]
            y_min = ann['bbox'][1]
            x_max = x_min + ann['bbox'][2]
            y_max = y_min + ann['bbox'][3]
            boxes.append([x_min, y_min, x_max, y_max])
            areas.append(ann['area'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(anns),), dtype=torch.int64)
        img_id_tensor = torch.tensor([img_id])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((len(anns),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": img_id_tensor,
            "area": areas,
            "iscrowd": iscrowd
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)



def get_transforms(train=True):
    trans = []
    if train:
        trans.append(transforms.RandomHorizontalFlip(0.5))
        # you can add more augmentations here
        # trans.append(transforms.ColorJitter(...))
    trans.append(transforms.ToTensor())
    return transforms.Compose(trans)


# ==========================
# PATHS (EDIT IF NEEDED)
# ==========================

HOME = os.path.dirname(os.path.abspath(__file__))

train_data_path = os.path.join(HOME, "images_with_person")
coco_path = os.path.join(HOME, "my_new_annotations.json")

print("[INFO] Image root:", train_data_path)
print("[INFO] COCO JSON:", coco_path)

selected_ids = list(range(2000, 3377))   # inclusive 2000 → 3376

train_dataset = ImageDataset(
    root=train_data_path,
    annotation=coco_path,
    transforms=get_transforms(train=True),
    selected_ids=selected_ids,
)

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
)


# ==========================
# MODEL: Faster R-CNN
# ==========================

# load COCO-pretrained Faster R-CNN
model = models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")  # or pretrained=True (older torchvision)

# we only have 2 classes: background (0) and person (1)
num_classes = 2

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model = model.to(device)

# ==========================
# OPTIMIZER + LR SCHEDULER
# ==========================

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005,
)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1,
)


# ==========================
# TRAINING LOOP
# ==========================

epochs = 20  # adjust as needed

for epoch in range(epochs):
    model.train()
    print(f"\n========== Epoch {epoch+1}/{epochs} ==========")
    for i, (imgs, targets) in enumerate(train_loader, start=1):
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i % 10 == 0 or i == 1:
            print(f"  Iteration {i}, Loss: {losses.item():.4f}")

    lr_scheduler.step()
    
torch.save(model.state_dict(), "trained_frcnn.pth")
print("\n[INFO] Saved trained model to trained_frcnn.pth")
print("\nTraining finished!")


# ==========================
# HELPER FOR VISUALIZATION
# ==========================

def im_convert(tensor):
    image = tensor.clone().detach().cpu().numpy()
    image = image.transpose(1, 2, 0)
    # naive denorm if you ever normalize
    image = image.clip(0, 1)
    return image
