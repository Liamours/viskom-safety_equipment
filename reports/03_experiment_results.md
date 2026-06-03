DETECTION MODEL RESULTS
=======================

All models are evaluated on the held-out test split using the best saved checkpoint (best.pt).
Metrics: mAP50, mAP50-95, Precision, Recall — computed by the Ultralytics validation engine.

Model: YOLO26L (24.75M parameters, 86.1 GFLOPs)
Hardware: NVIDIA RTX 4050 Laptop GPU (6 GB VRAM)
Input size: 640 × 640 px


OVERALL PERFORMANCE

| Dataset | Split    | Test Images | mAP50  | mAP50-95 | Precision | Recall |
|---------|----------|-------------|--------|----------|-----------|--------|
| CHV     | Original | 133         | 92.33% | 54.77%   | 92.57%    | 85.87% |
| CHV     | 8020     | 266         | 89.06% | 52.84%   | 89.88%    | 82.20% |
| CPPE-5  | Original | 29          | 76.49% | 52.70%   | 81.17%    | 69.82% |
| CPPE-5  | 8020     | 206         | 72.64% | 45.21%   | 79.27%    | 70.12% |
| SH17    | Original | 1,620       | 64.31% | 43.21%   | 74.59%    | 59.04% |
| SH17    | 8020     | 1,620       | 65.17% | 43.41%   | 76.22%    | 60.62% |

CHV achieves the highest scores — its 6-class setup is simpler and the images are relatively clean.
CPPE-5 is harder due to heavy occlusion in medical settings. SH17 is the hardest: 17 classes,
severe class imbalance (118:1), and many rare/small items bring the average down significantly.

The gap between mAP50 and mAP50-95 is large across all datasets (e.g. CHV: 92% vs 55%).
This means the model detects items at the right location but is not tight on bounding box fit,
which is common for flexible or irregularly shaped PPE like vests and coveralls.


PER-CLASS RESULTS — CHV (Original split, 133 test images)

| Class         | AP@0.50 | AP@0.50:0.95 |
|---------------|---------|--------------|
| red_helmet    | 96.32%  | 54.24%       |
| yellow_helmet | 94.86%  | 58.50%       |
| white_helmet  | 94.52%  | 56.39%       |
| person        | 91.77%  | 54.10%       |
| vest          | 90.68%  | 54.38%       |
| blue_helmet   | 85.80%  | 51.04%       |

All helmet colors score 85–96% at mAP50. Vest scores 90.68% — solid, though vest recall
drops in the compliance pipeline when images differ from the training distribution
(e.g. stock photos or unusual color/style).


PER-CLASS RESULTS — CHV (8020 split, 266 test images)

| Class         | AP@0.50 | AP@0.50:0.95 |
|---------------|---------|--------------|
| yellow_helmet | 91.33%  | 55.43%       |
| person        | 90.36%  | 55.46%       |
| red_helmet    | 90.23%  | 53.14%       |
| white_helmet  | 88.55%  | 53.25%       |
| blue_helmet   | 88.07%  | 48.44%       |
| vest          | 85.82%  | 51.33%       |

Slightly lower across the board compared to the original split, which is expected — the 8020
split has more test images (266 vs 133) including harder examples.


PER-CLASS RESULTS — CPPE-5 (Original split, 29 test images)

| Class       | AP@0.50 | AP@0.50:0.95 |
|-------------|---------|--------------|
| Face_Shield | 93.08%  | 69.64%       |
| Coverall    | 81.28%  | 60.92%       |
| Mask        | 76.46%  | 52.60%       |
| Goggles     | 71.32%  | 37.91%       |
| Gloves      | 60.32%  | 42.42%       |

Face shields are the easiest to detect (93% — large, distinctive shape). Gloves are the hardest
(60%) because they are small, often partially occluded, and appear in many orientations.
Note: 29 test images is a very small evaluation set; results may have high variance.


PER-CLASS RESULTS — CPPE-5 (8020 split, 206 test images)

| Class       | AP@0.50 | AP@0.50:0.95 |
|-------------|---------|--------------|
| Coverall    | 85.36%  | 59.46%       |
| Goggles     | 73.76%  | 44.24%       |
| Mask        | 72.72%  | 43.66%       |
| Face_Shield | 70.42%  | 40.46%       |
| Gloves      | 60.93%  | 38.25%       |

With 206 test images (vs 29 in original split), coverall improves to 85% but face_shield drops
from 93% to 70% — the extra test images include harder examples.


PER-CLASS RESULTS — SH17 (Original split, 1,620 test images)

| Class             | AP@0.50 | AP@0.50:0.95 |
|-------------------|---------|--------------|
| face              | 93.48%  | 71.28%       |
| head              | 91.95%  | 72.17%       |
| person            | 91.19%  | 76.10%       |
| hands             | 87.23%  | 60.85%       |
| ear               | 83.39%  | 53.06%       |
| face-mask-medical | 72.88%  | 45.30%       |
| glasses           | 71.40%  | 39.42%       |
| helmet            | 70.61%  | 50.71%       |
| shoes             | 66.17%  | 39.53%       |
| gloves            | 59.69%  | 37.48%       |
| safety-vest       | 55.63%  | 36.18%       |
| medical-suit      | 53.01%  | 29.83%       |
| earmuffs          | 49.00%  | 32.57%       |
| safety-suit       | 45.10%  | 29.64%       |
| face-guard        | 42.93%  | 28.13%       |
| tools             | 38.66%  | 22.32%       |
| foot              | 20.99%  | 9.92%        |

Body-part classes (face, head, person, hands, ear) score highest because they are large and
abundant. Critical PPE like helmet (70.6%) and safety-vest (55.6%) score lower.
Foot detection is the weakest (20.99%) — feet are small, occluded, and rarely labeled.


PER-CLASS RESULTS — SH17 (8020 split, 1,620 test images)

| Class             | AP@0.50 | AP@0.50:0.95 |
|-------------------|---------|--------------|
| face              | 92.63%  | 69.92%       |
| head              | 88.31%  | 68.90%       |
| person            | 88.11%  | 72.01%       |
| ear               | 87.40%  | 55.37%       |
| hands             | 86.60%  | 59.96%       |
| medical-suit      | 75.48%  | 49.55%       |
| glasses           | 71.86%  | 41.35%       |
| helmet            | 70.16%  | 49.84%       |
| face-mask-medical | 69.71%  | 43.88%       |
| gloves            | 64.15%  | 40.82%       |
| shoes             | 60.91%  | 35.60%       |
| face-guard        | 57.40%  | 34.59%       |
| safety-vest       | 52.80%  | 33.88%       |
| earmuffs          | 50.22%  | 29.94%       |
| tools             | 38.40%  | 22.29%       |
| foot              | 29.81%  | 14.01%       |
| safety-suit       | 23.87%  | 16.06%       |

Results are largely comparable to the original split. Medical-suit improves notably (53% → 75%).
Safety-suit drops (45% → 24%), likely because the 8020 random split concentrated harder examples
into the test set for that class.


INFERENCE SPEED (RTX 4050 Laptop GPU, on test set)

| Dataset | Split    | Preprocess | Inference | Postprocess |
|---------|----------|------------|-----------|-------------|
| CHV     | Original | 3.2 ms     | 25.4 ms   | 0.2 ms      |
| CHV     | 8020     | 0.9 ms     | 19.3 ms   | 0.2 ms      |
| CPPE-5  | Original | 1.9 ms     | 21.4 ms   | 0.3 ms      |
| CPPE-5  | 8020     | 5.5 ms     | 21.8 ms   | 1.1 ms      |
| SH17    | Original | 0.7 ms     | 24.8 ms   | 0.2 ms      |
| SH17    | 8020     | 0.6 ms     | 26.8 ms   | 0.1 ms      |

Inference time is 19–27 ms per image across all runs (~37–53 FPS), well within real-time range.
Preprocessing variance is due to different average image sizes across test sets.


TRAINING LOGS AND PLOTS

| Dataset | Split    | Training Log                                      | Plots Folder                               |
|---------|----------|---------------------------------------------------|--------------------------------------------|
| CHV     | Original | runs/detect/chv_original/results.csv             | runs/detect/chv_original/                 |
| CHV     | 8020     | runs/detect/chv_8020/results.csv                 | runs/detect/chv_8020/                     |
| CPPE-5  | Original | runs/detect/cppe_5_original/results.csv          | runs/detect/cppe_5_original/              |
| CPPE-5  | 8020     | runs/detect/cppe_5_8020/results.csv              | runs/detect/cppe_5_8020/                  |
| SH17    | Original | runs/detect/sh17_original/results.csv            | runs/detect/sh17_original/                |
| SH17    | 8020     | runs/detect/sh17_8020/results.csv                | runs/detect/sh17_8020/                    |

Available plots per run:
  results.png                     — training loss and metric curves across epochs
  confusion_matrix.png            — per-class confusion matrix
  confusion_matrix_normalized.png — row-normalized confusion matrix
  BoxPR_curve.png                 — Precision-Recall curve per class
  BoxF1_curve.png                 — F1 vs confidence threshold curve
  val_batch0_labels.jpg           — sample ground truth annotations from validation
  val_batch0_pred.jpg             — model predictions on the same sample
