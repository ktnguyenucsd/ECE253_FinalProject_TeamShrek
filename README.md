Folder for weights and pretrained models
https://drive.google.com/drive/folders/1rYN2laVyB12wYf4o7zrdsts4I962hqN1?usp=sharing

## Model Weights Setup

Download the following files and place them in the project directory:

project_root/
└── weights/
├── trained_frcnn.pth
└── yolov8m.pt


##  Scripts Pipeline Overview

Below is an overview of the project pipeline and the purpose of each script.

### `saltpeppercorruption.py` *(Optional)*
- Applies synthetic visual corruptions (Salt and pepper/gaussian noise) to the input images.


### `generate_annotations.py` *(Optional)*
- Generates segmentation masks and bounding-box annotations for images containing pedestrians.
- Used to prepare training data for the detector.



###  `train_frcnn.py` *(Optional)*
- Trains a Faster R-CNN model using the generated annotations and masks.
- Outputs a trained model checkpoint (`trained_frcnn.pth`).

---


### `infer_and_save.py` *(Required)*
- Runs inference using the trained Faster R-CNN model on the cleaned or restored dataset.
- Saves prediction outputs (bounding boxes, confidence scores) to disk.
- These predictions are used as input for evaluation.


### `eval_map.py` *(Required)*
- Computes **mean Average Precision (mAP)** across multiple Intersection-over-Union (IoU) thresholds.
- Evaluates performance over **10 IoU thresholds**, following the COCO-style mAP protocol.
- Outputs quantitative detection performance metrics.

  
### `denoising_scores.py`
- Computes quantitative image-quality metrics (SSIM, MSE, PSNR) for denoised images.
- Used to evaluate the effectiveness of denoising methods prior to detection.

### `exposure_correction.py`
- Applies exposure correction techniques (gamma correction, CLAHE) to input images.

### `exposure_scores.py`
- Computes exposure-related metrics such as mean luminance, contrast, entropy, and exposure error.

### `med_nlm.py`
- Designed to remove salt-and-pepper and Gaussian noise while preserving image structure using Med. filter + Non Local Means.
