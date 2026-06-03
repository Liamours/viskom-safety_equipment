DATASET DETAILS
===============


DATASET DESCRIPTIONS

| Dataset | Full Name                          | PPE Categories Covered                              | Source / Reference   |
|---------|------------------------------------|-----------------------------------------------------|----------------------|
| CHV     | Construction Helmet and Vest       | Person, safety vest, helmets (blue/red/white/yellow)| Wang et al., 2021    |
| CPPE-5  | PPE-5 Class Dataset                | Coverall, face shield, gloves, goggles, mask        | Dagli & Shaikh, 2021 |
| SH17    | Safety Helmet 17-Class Dataset     | 17 body/PPE parts incl. helmet, vest, gloves, mask  | Ahmad & Rahimi, 2024 |

CHV focuses on construction environments. CPPE-5 covers medical and healthcare PPE. SH17 is the
most comprehensive, covering industrial workplaces with 17 distinct body part and equipment classes.


OVERVIEW

| Dataset | Images | Instances | Classes | Annotation Format | Landmark JSONs |
|---------|--------|-----------|---------|-------------------|----------------|
| CHV     | 1,330  | 9,209     | 6       | YOLO TXT          | 2,660          |
| CPPE-5  | 1,029  | 4,698     | 5       | COCO JSON         | 2,058          |
| SH17    | 8,099  | 75,994    | 17      | YOLO TXT          | 16,198         |

Each image has two landmark JSON files: one from MediaPipe Holistic and one from YOLOv8 Pose.


IMAGE PROPERTIES

| Dataset | Color Mode     | Height (min/max/mean) | Width (min/max/mean) | Channels |
|---------|----------------|-----------------------|----------------------|----------|
| CHV     | RGB, CMYK      | 160 / 4500 / 777 px   | 107 / 6750 / 1067 px | 3–4      |
| CPPE-5  | RGB, RGBA, P   | 110 / 6240 / 795 px   | 54 / 7329 / 1118 px  | 1–4      |
| SH17    | RGB            | 186 / 640 / 520 px    | 336 / 640 / 547 px   | 3        |

Note: SH17 images were pre-resized (longest side = 640 px, aspect ratio preserved) before training.
CHV and CPPE-5 are resized on-the-fly by the training pipeline (letterboxed to 640×640).


TRAIN/VAL/TEST SPLITS

Two split configurations were prepared per dataset:
- Original: uses the dataset author's original train/val/test partitioning
- 8020: custom 80/20 random split for comparison

| Dataset | Split    | Train  | Val   | Test  |
|---------|----------|--------|-------|-------|
| CHV     | Original | 1,064  | 133   | 133   |
| CHV     | 8020     | 905    | 159   | 266   |
| CPPE-5  | Original | 849    | —     | 28    |
| CPPE-5  | 8020     | 700    | 123   | 206   |
| SH17    | Original | 6,479  | 1,620 | —     |
| SH17    | 8020     | 5,507  | —     | 1,619 |

Note: CPPE-5 original split has no validation set. SH17 original split has no separate test set
(validation set is used for evaluation).


CLASS DISTRIBUTION

CHV

| Class         | Instances |
|---------------|-----------|
| person        | 3,887     |
| vest          | 1,784     |
| yellow_helmet | 1,299     |
| white_helmet  | 1,195     |
| red_helmet    | 536       |
| blue_helmet   | 508       |

CHV separates helmet color into 4 sub-classes. The pipeline merges all helmet colors into a single
"helmet" category for compliance checking.

CPPE-5 (train split)

| Class       | Instances |
|-------------|-----------|
| Gloves      | 1,282     |
| Mask        | 1,252     |
| Coverall    | 1,152     |
| Face_Shield | 430       |
| Goggles     | 375       |

Note: CPPE-5 has no "person" class — person detection is handled entirely by the YOLOv8-pose model.

SH17

| Class             | Instances |
|-------------------|-----------|
| hands             | 15,850    |
| person            | 13,802    |
| head              | 11,985    |
| face              | 8,950     |
| ear               | 7,730     |
| tools             | 4,647     |
| shoes             | 4,560     |
| gloves            | 2,790     |
| glasses           | 1,945     |
| helmet            | 927       |
| foot              | 759       |
| face-mask-medical | 670       |
| safety-vest       | 530       |
| earmuffs          | 318       |
| safety-suit       | 240       |
| medical-suit      | 157       |
| face-guard        | 134       |

SH17 includes both body parts (hands, face, ear) and PPE items (helmet, gloves, safety-vest).
Body-part classes are used internally for pose anchoring but not counted as PPE for compliance.


CLASS IMBALANCE

| Dataset | Most Common Class        | Least Common Class  | Imbalance Ratio |
|---------|--------------------------|---------------------|-----------------|
| CHV     | person (3,887)           | blue_helmet (508)   | 7.6 : 1         |
| CPPE-5  | Gloves (1,282)           | Goggles (375)       | 3.4 : 1         |
| SH17    | hands (15,850)           | face-guard (134)    | 118.3 : 1       |

SH17 has extreme imbalance — rare items like face-guard (134 instances) are heavily outnumbered
by hands (15,850). This directly causes low recall on rare classes in detection results.


LANDMARK EXTRACTION

Pose landmarks were extracted for every image using two methods: MediaPipe Holistic and YOLOv8 Pose.
Each detected person yields 17 COCO-format body keypoints stored as (x, y, z, visibility) per point.
Results are saved as JSON, separated into detected and undetected subfolders per dataset.

| Dataset | Total Images | MediaPipe Detected | MediaPipe Rate | YOLOv8 Detected | YOLOv8 Rate |
|---------|--------------|--------------------|----------------|-----------------|-------------|
| CHV     | 1,330        | 1,273              | 95.7%          | 1,314           | 98.8%       |
| CPPE-5  | 1,029        | 872                | 84.7%          | 1,025           | 99.6%       |
| SH17    | 8,099        | 7,133              | 88.1%          | 7,852           | 96.9%       |

YOLOv8 Pose consistently outperforms MediaPipe on detection rate, particularly for CPPE-5 (+15%).
MediaPipe struggles with heavily-occluded or non-frontal poses common in medical PPE images.

Output folders per dataset (under landmarks/{DATASET}/):
  landmark-mediapipe/   — one JSON per image (detected or not)
  landmark-yolov8/      — one JSON per image (detected or not)
  detected-mediapipe/   — annotated images where at least one person was found
  detected-yolov8/      — annotated images where at least one person was found
  undetected-mediapipe/ — images with no person found by MediaPipe
  undetected-yolov8/    — images with no person found by YOLOv8
