TRAINING SCHEMA
===============


HARDWARE AND SOFTWARE ENVIRONMENT

| Component     | Details                                        |
|---------------|------------------------------------------------|
| GPU           | NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM) |
| CUDA          | 11.8                                           |
| Python        | 3.12.3                                         |
| PyTorch       | 2.7.1+cu118                                    |
| Ultralytics   | 8.4.31                                         |
| OS            | Windows 11                                     |

All training was done on a single GPU. 6 GB VRAM was the main constraint — batch size was kept
at 8 to avoid out-of-memory errors on large datasets like SH17.


DETECTION MODEL

The same model architecture was used for all 6 training runs across the 3 datasets.

| Property    | Value                        |
|-------------|------------------------------|
| Model       | YOLO26L (yolo26l.pt)         |
| Parameters  | 24,750,366 (24.75M)          |
| GFLOPs      | 86.1                         |
| Layers      | 190 (fused for inference)    |
| Task        | Object Detection             |
| Pretrained  | Yes (COCO)                   |

YOLO26L is a large-scale YOLO variant pretrained on COCO. The pretrained weights provide strong
general object detection capability which is then fine-tuned on each PPE dataset.


POSE ESTIMATION MODEL (used in compliance pipeline, not trained)

| Property    | Value                  |
|-------------|------------------------|
| Model       | YOLOv8x-Pose           |
| Weights     | yolov8x-pose.pt        |
| Keypoints   | 17 (COCO format)       |
| Task        | Pose Estimation        |
| Pretrained  | Yes (COCO)             |
| Fine-tuned  | No — used as-is        |

YOLOv8x-Pose is the largest YOLOv8 pose model. It was not fine-tuned — the pretrained COCO weights
are used directly to detect persons and extract body keypoints during compliance checking.


TRAINING HYPERPARAMETERS (same for all 6 runs)

| Parameter      | Value         |
|----------------|---------------|
| Epochs         | 150 (max)     |
| Early Stopping | patience = 20 |
| Batch Size     | 8             |
| Image Size     | 640 × 640 px  |
| Optimizer      | AdamW (auto)  |
| Learning Rate  | lr0 = 0.001   |
| LR Final       | lrf = 0.01    |
| Weight Decay   | 0.0005        |
| Warmup Epochs  | 3             |
| AMP            | Enabled       |
| IoU Threshold  | 0.7           |
| Device         | GPU (CUDA:0)  |
| Workers        | 8             |


AUGMENTATION

| Augmentation    | Value          |
|-----------------|----------------|
| Mosaic          | 1.0 (enabled)  |
| Horizontal Flip | 0.5            |
| HSV Hue         | 0.015          |
| HSV Saturation  | 0.7            |
| HSV Value       | 0.4            |
| Scale           | 0.5            |
| Translate       | 0.1            |
| Auto Augment    | RandAugment    |
| Random Erasing  | 0.4            |
| Close Mosaic    | Last 10 epochs |

Mosaic augmentation tiles 4 images into one, significantly improving detection of small objects.
It is disabled for the final 10 epochs to allow the model to stabilize on clean inputs.


TRAINING RUNS

6 models were trained in total: 2 split configurations × 3 datasets.

| Dataset | Split    | Saved Weights Path                                  | Epochs Run | Best Epoch | Status   |
|---------|----------|-----------------------------------------------------|------------|------------|----------|
| CHV     | Original | runs/detect/chv_original/weights/best.pt            | 73         | 53         | Complete |
| CHV     | 8020     | runs/detect/chv_8020/weights/best.pt                | 71         | 64         | Complete |
| CPPE-5  | Original | runs/detect/cppe_5_original/weights/best.pt         | 131        | 56         | Complete |
| CPPE-5  | 8020     | runs/detect/cppe_5_8020/weights/best.pt             | 62         | 52         | Complete |
| SH17    | Original | runs/detect/sh17_original/weights/best.pt           | 115        | 95         | Complete |
| SH17    | 8020     | runs/detect/sh17_8020/weights/best.pt               | 144        | 124        | Complete |

Early stopping triggered on all runs before reaching the 150-epoch maximum.


TRAINING DURATION

| Dataset | Split    | Epochs Run | Approx. Time per Epoch | Total Duration |
|---------|----------|------------|------------------------|----------------|
| CHV     | Original | 73         | ~30 sec                | ~37 min        |
| CHV     | 8020     | 71         | ~2.5 min               | ~3 hr          |
| CPPE-5  | Original | 131        | ~2 min                 | ~4.5 hr        |
| CPPE-5  | 8020     | 62         | ~3.5 min               | ~3.5 hr        |
| SH17    | Original | 115        | ~7.5 min               | ~14.5 hr       |
| SH17    | 8020     | 144        | ~7.5 min               | ~18 hr         |

CHV original split trains significantly faster (~30 sec/epoch) because the original split images are
smaller on average than the 8020 resplit. SH17 is the most expensive at ~7.5 min/epoch due to its
8,099 images and 17 classes.

Total GPU time across all 6 runs: approximately 44 hours.


LANDMARK EXTRACTION

Pose landmarks were extracted from all dataset images before training, using two separate methods.
These are stored as JSON files and used later in the compliance pipeline for keypoint-based anchoring.

| Property              | MediaPipe Holistic       | YOLOv8 Pose              |
|-----------------------|--------------------------|--------------------------|
| Model                 | MediaPipe Holistic       | YOLOv8 Pose (pretrained) |
| Keypoints per person  | 33 (MediaPipe format)    | 17 (COCO format)         |
| Keypoints stored      | 17 (COCO body subset)    | 17 (COCO body format)    |
| Keypoint fields       | x, y, z, visibility      | x, y, z=0, visibility    |
| Depth (z)             | Yes (normalized)         | Not available (z=0)      |
| Output format         | JSON per image           | JSON per image           |
| Input                 | Full image               | Full image               |

Only YOLOv8 Pose keypoints are used in the final compliance pipeline, because they use the same
COCO 17-keypoint format as the compliance anchor definitions. MediaPipe's 33-keypoint format
requires index remapping and was found to cause connection mismatch errors when used directly.
