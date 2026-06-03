PROJECT STATUS
==============


LANDMARK EXTRACTION

| Task                              | Status | Notes                          |
|-----------------------------------|--------|--------------------------------|
| MediaPipe landmarks — CHV         | Done   | 1,273 / 1,330 detected (95.7%) |
| YOLOv8 Pose landmarks — CHV       | Done   | 1,314 / 1,330 detected (98.8%) |
| MediaPipe landmarks — CPPE-5      | Done   | 872 / 1,029 detected (84.7%)   |
| YOLOv8 Pose landmarks — CPPE-5    | Done   | 1,025 / 1,029 detected (99.6%) |
| MediaPipe landmarks — SH17        | Done   | 7,133 / 8,099 detected (88.1%) |
| YOLOv8 Pose landmarks — SH17      | Done   | 7,852 / 8,099 detected (96.9%) |


DETECTION MODEL TRAINING

| Task                            | Status | Notes                                          |
|---------------------------------|--------|------------------------------------------------|
| Dataset prep — CHV              | Done   | Original + 8020 splits ready                   |
| Dataset prep — CPPE-5           | Done   | Original + 8020 splits ready                   |
| Dataset prep — SH17             | Done   | Images resized to 640px longest side           |
| Train CHV (original split)      | Done   | 73 epochs, best at epoch 53, ~37 min           |
| Train CHV (8020 split)          | Done   | 71 epochs, best at epoch 64, ~3 hr             |
| Train CPPE-5 (original split)   | Done   | 131 epochs, best at epoch 56, ~4.5 hr          |
| Train CPPE-5 (8020 split)       | Done   | 62 epochs, best at epoch 52, ~3.5 hr           |
| Train SH17 (original split)     | Done   | 115 epochs, best at epoch 95, ~14.5 hr         |
| Train SH17 (8020 split)         | Done   | 144 epochs, best at epoch 124, ~18 hr          |
| Evaluate all 6 models           | Done   | mAP50 range: 64–92% depending on dataset       |


COMPLIANCE PIPELINE (PACT)

PACT (Pose-Anchored Compliance Tracker) is the core system that combines pose estimation
and PPE detection to determine whether each detected person is wearing required equipment.

| Task                                    | Status | Notes                                               |
|-----------------------------------------|--------|-----------------------------------------------------|
| Pipeline design and implementation      | Done   | code/compliance/pipeline.py                         |
| COCO 17-keypoint anchor mapping         | Done   | code/landmarks/base.py (COCO_PPE_ANCHOR_GROUPS)     |
| Pose skeleton drawing (COCO format)     | Done   | code/landmarks/core.py + COCO_CONNECTIONS           |
| Helmet anchor upward extension fix      | Done   | Anchor top extended to person bbox top              |
| Helmet fallback anchor (no keypoints)   | Done   | Top 35% of person bbox used when head kps invisible |
| False person filter (low conf pose)     | Done   | Pose detections below 0.50 confidence are skipped   |
| Per-person partial compliance scoring   | Done   | Score = worn_required / total_required              |
| Overall compliance rate (partial)       | Done   | Mean of per-person scores (not binary)              |
| Pipeline evaluation — CHV               | Done   | Assignment accuracy 93.8–95.6%                      |
| Pipeline evaluation — CPPE-5            | Done   | Single-person frames, assignment N/A                |
| Pipeline evaluation — SH17              | Done   | Assignment accuracy 89.2%                           |


REPORT GENERATION SYSTEM

| Task                                    | Status | Notes                                               |
|-----------------------------------------|--------|-----------------------------------------------------|
| Rule-based text generator               | Done   | code/reporting/generator.py                         |
| Severity tier classification            | Done   | CRITICAL / HIGH / MEDIUM / LOW per PPE type         |
| Dataset-specific compliance rules       | Done   | code/compliance/rules.py (chv, cppe5, sh17)         |
| Per-person narrative generation         | Done   | One-sentence summary with compliance score          |
| Overall scene assessment                | Done   | Mentions compliance rate and context                |
| Action item grouping by PPE type        | Done   | Grouped across workers, not repeated per-person     |
| HTML report with embedded images        | Done   | code/reporting/renderer.py                          |
| Plain-text report export                | Done   | results/components/{dataset}_06_report.txt          |
| JSON output                             | Done   | results/components/{dataset}_05_json.json           |
| 3-dataset demo run (CHV, CPPE-5, SH17) | Done   | Sample images processed, all 18 component files     |


OUTPUT FILE LOCATIONS

| Type                  | Location                                             |
|-----------------------|------------------------------------------------------|
| Best weights          | runs/detect/{run_name}/weights/best.pt               |
| Training log          | runs/detect/{run_name}/results.csv                   |
| Training plots        | runs/detect/{run_name}/*.png                         |
| Raw image panel       | results/components/{dataset}_01_raw.png              |
| Landmarks panel       | results/components/{dataset}_02_landmarks.png        |
| PACT detection panel  | results/components/{dataset}_03_pact.png             |
| Protocol text         | results/components/{dataset}_04_protocol.txt         |
| JSON output           | results/components/{dataset}_05_json.json            |
| Report text           | results/components/{dataset}_06_report.txt           |
| Full figure (3×6)     | results/figure_pipeline.png / .pdf                   |


KNOWN LIMITATIONS

| Issue                                       | Cause                                          | Status      |
|---------------------------------------------|------------------------------------------------|-------------|
| Vest not detected on stock/clipart photos   | Training data is real site photos only         | Known limit |
| Low recall on rare SH17 classes             | Class imbalance (up to 118:1)                  | Known limit |
| Helmet anchor may miss on extreme poses     | Face keypoints occluded; fallback now in place | Mitigated   |
| Assignment error on heavily overlapping     | IoU-based assignment struggles at close range  | Known limit |
| CPPE-5 person detection (no person class)   | Dataset has no person annotation               | By design   |
