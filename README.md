# PACT: Pose-Anchored Compliance Tracker

**Per-worker PPE compliance reporting via pose-anchored body region assignment.**

> This work is currently under review for submission to **ICERA 2026**.

PACT combines YOLO26L PPE detection, YOLOv8x-pose person localization, and a pose-anchored IoU assignment module to attribute each detected PPE item to a specific worker. A rule-based reporter computes fractional per-person compliance scores and severity-tiered corrective actions.

---

## Contributors

| Name | Role |
|---|---|
| Nabila Putri Azhari | Paper Writer & Literature Review |
| Fathan Arya Maulana | Paper Writer & Documentator |
| M. Rifqi Dzaky Azhad | Coder |

---

## Datasets

| Dataset | Domain | Images | Instances | Classes |
|---------|--------|--------|-----------|---------|
| [CHV](https://github.com/zj-jayzhang/CHV) | Construction | 1,330 | 9,209 | 6 |
| [CPPE-5](https://github.com/Rishit-dagli/CPPE-5) | Medical | 1,029 | 4,698 | 5 |
| [SH17](https://github.com/ahmadmughees/SH17) | Industrial | 8,099 | 75,994 | 17 |

- **CHV**: `person`, `vest`, `{red,yellow,white,blue}_helmet`. Split: 1,064/133/133. Helmet colors merged to single class at compliance stage.
- **CPPE-5**: `Coverall`, `Face_Shield`, `Gloves`, `Goggles`, `Mask`. COCO JSON → YOLO TXT. No person-class annotations.
- **SH17**: 17 classes (body parts + PPE). Split: 6,479 train / 1,620 val-as-test. Class imbalance up to 118:1.

---

## Preprocessing

**CHV & CPPE-5**: letterboxed to 640×640 on-the-fly by Ultralytics; CMYK/RGBA normalized at load time.

**SH17 — offline resize**: images pre-resized (longest side = 640 px, aspect preserved, LANCZOS) before training to fit 6 GB VRAM at batch size 8. Labels copied unchanged — YOLO coords are scale-invariant. See `code/analysis/resize_sh17.py`.

**Landmark extraction**: YOLOv8x-pose (17 COCO keypoints) used in pipeline; MediaPipe (33 kps) extracted but discarded due to index mismatch.

| Method | CHV | CPPE-5 | SH17 |
|---|---|---|---|
| MediaPipe detection rate | 95.7% | 84.7% | 88.1% |
| YOLOv8x-pose detection rate | 98.8% | 99.6% | 96.9% |

---

## Models

**YOLO26L** — trained (fine-tuned from COCO per dataset)

| Parameters | GFLOPs | Layers | Task |
|---|---|---|---|
| 24.75M | 86.1 | 190 (fused) | Object detection |

**YOLOv8x-pose** — inference only, not fine-tuned

| Keypoints | Weights | Task |
|---|---|---|
| 17 (COCO) | yolov8x-pose.pt | Person detection + pose estimation |

---

## Training

GPU: NVIDIA RTX 4050 Laptop (6 GB) · CUDA 11.8 · PyTorch 2.7.1 · Ultralytics 8.4.31

| Parameter | Value | Augmentation | Value |
|---|---|---|---|
| Max epochs | 150 | Mosaic | 1.0 |
| Early stopping | patience 20 | Horiz. flip | p=0.5 |
| Batch size | 8 | HSV jitter | h=0.015, s=0.7, v=0.4 |
| Optimizer | AdamW | Auto augment | RandAugment |
| lr0 / lrf | 0.001 / 0.01 | Random erasing | 0.4 |
| Weight decay | 0.0005 | | |

| Dataset | Split | Epochs | Best | Duration |
|---|---|---|---|---|
| CHV | Original | 73 | 53 | ~37 min |
| CHV | 80/20 | 71 | 64 | ~3 hr |
| CPPE-5 | Original | 131 | 56 | ~4.5 hr |
| CPPE-5 | 80/20 | 62 | 52 | ~3.5 hr |
| SH17 | Original | 115 | 95 | ~14.5 hr |
| SH17 | 80/20 | 144 | 124 | ~18 hr |

Total GPU time: ~44 hours.

---

## Inference Pipeline (PACT)

**Stage 1 — Parallel Detection**: YOLO26L produces PPE boxes; YOLOv8x-pose produces person boxes + 17 keypoints. Person-class detections from YOLO26L are discarded. Pose conf < 0.50 filtered.

**Stage 2 — Pose-Anchored Assignment**: Anatomical anchor regions built from keypoints (head → helmet, shoulder/hip → vest, wrist → gloves; expanded δ=0.10, vis threshold τᵥ=0.30). Each PPE box assigned to person with highest anchor IoU (τₐ=0.10).

**Stage 3 — Compliance Reporting**: Per-person score = worn ∩ required / required. Missing items classified by severity:

| Tier | PPE | Label |
|---|---|---|
| Critical / High | helmet, vest, coverall, mask | `[URGENT]` |
| Medium / Low | gloves, face shield, goggles, shoes | `[ADVISORY]` |

---

## Evaluation Metrics

**PPE Detection (mAP50)**

$$\text{mAP}_{50} = \frac{1}{C} \sum_{c=1}^{C} \text{AP}_{50}^{(c)}$$

**Person Detection**

$$\text{F1} = \frac{2 \cdot \text{P} \cdot \text{R}}{\text{P} + \text{R}}, \quad \text{P} = \frac{\text{TP}}{\text{TP}+\text{FP}}, \quad \text{R} = \frac{\text{TP}}{\text{TP}+\text{FN}}$$

**Assignment Accuracy** (multi-person frames only, ≥2 GT persons)

$$\text{Acc}_{\text{assign}} = \frac{A}{T}$$

**Per-Person Compliance Score**

$$\text{score}_{i} = \frac{|\text{worn}_i \cap \text{required}|}{|\text{required}|}, \quad \text{compliance}_{\text{scene}} = \frac{1}{N}\sum_{i=1}^{N} \text{score}_{i}$$

---

## Results

**PPE Detection**

| Dataset | Split | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|
| CHV | Original | 0.923 | 0.548 | 0.926 | 0.859 |
| CHV | 80/20 | 0.891 | 0.528 | 0.899 | 0.822 |
| CPPE-5 | Original | 0.765 | 0.527 | 0.812 | 0.698 |
| SH17 | Original | 0.643 | 0.432 | 0.746 | 0.590 |

**Person Detection** (YOLOv8x-pose, pretrained only)

| Dataset | Precision | Recall | F1 |
|---|---|---|---|
| CHV | 0.907 | 0.800 | 0.850 |
| SH17 | 0.944 | 0.836 | 0.887 |
| CPPE-5 | — | — | — (no GT person boxes) |

**Assignment Accuracy**

| Dataset | Split | Correct / Total | Accuracy |
|---|---|---|---|
| CHV | Original | 408 / 435 | 0.938 |
| CHV | 80/20 | 630 / 659 | 0.956 |
| SH17 | Original | 521 / 584 | 0.892 |

**Runtime** (RTX 4050, 1,782 images avg)

| Stage | Latency |
|---|---|
| YOLO26L (PPE) | 29.44 ms |
| YOLOv8x-pose | 60.73 ms |
| Anchor + compliance | 0.42 ms |
| **End-to-end** | **90.72 ms / 11.02 FPS** |

---

## Figures

**Pipeline**

![Pipeline](conference/draft/figures/pipeline.png)

**PPE Detection Flow**

![PPE flow](conference/illustration/flow_ppe_detection.png)

**Landmark Detection Flow**

![Landmark flow](conference/illustration/flow_landmark_detection.png)

**Qualitative Results**

| Dataset | Raw | Pose | Detection | Report |
|---|---|---|---|---|
| CHV | ![](conference/draft/figures/raw1.png) | ![](conference/draft/figures/landmark1.png) | ![](conference/draft/figures/pact1.png) | ![](conference/draft/figures/report1.png) |
| CPPE-5 | ![](conference/draft/figures/raw2.png) | ![](conference/draft/figures/landmark2.png) | ![](conference/draft/figures/pact2.png) | ![](conference/draft/figures/report2.png) |
| SH17 | ![](conference/draft/figures/raw3.png) | ![](conference/draft/figures/landmark3.png) | ![](conference/draft/figures/pact3.png) | ![](conference/draft/figures/report3.png) |

**Failure Cases** (missed vests: occlusion/side-view; false positives CPPE-5: background color similarity)

![Failure cases](conference/draft/figures/wrongcase.png)

---

## Project Structure

```
viskom-safety_equipment/
├── code/
│   ├── analysis/       # dataset analysis, splits, SH17 resize
│   ├── compliance/     # PACT pipeline, rules, visualizer
│   ├── detection/      # YOLO configs, trainer, evaluator
│   ├── experiment/
│   │   ├── phase_1/   # train + evaluate detection models
│   │   └── phase_2/   # PACT eval, reports, ablation
│   ├── landmarks/      # keypoint extraction (MediaPipe + YOLOv8)
│   ├── reporting/      # HTML report generator
│   └── utils/
├── results/
│   ├── phase_1/        # detection eval JSONs
│   └── phase_2/        # PACT eval JSONs + component images
├── reports/            # markdown experiment summaries
└── conference/
    ├── draft/          # LaTeX paper + figures
    └── illustration/   # TikZ flow diagrams + PNGs
```

---

## Usage

```bash
# Phase 1 — train
python code/experiment/phase_1/train_chv.py
python code/experiment/phase_1/train_cppe5.py
python code/experiment/phase_1/train_sh17.py
python code/experiment/phase_1/evaluate_all.py

# Phase 2 — PACT compliance
python code/experiment/phase_2/evaluate_pact.py
python code/experiment/phase_2/sample_pact_eval.py
python code/experiment/phase_2/generate_report.py

# Ablation
python code/experiment/ablation_assignment.py

# Render flow diagrams
cd conference/illustration && uv run render.py
```

**Dependencies**: `ultralytics>=8.4.0`, `torch>=2.0.0`, `mediapipe`, `opencv-python`, `Pillow`, `numpy`, `tqdm`, `jinja2`

---

## References

- Wang et al., *Sensors* 2021 — CHV dataset
- Dagli & Shaikh, *SN Computer Science* 2023 — CPPE-5 dataset
- Ahmad & Rahimi, *JSSR* 2024 — SH17 dataset
- Sapkota et al., arXiv:2509.25164, 2026 — YOLO26
- Vukicevic et al., *AI Review* 2024 — PPE compliance survey
