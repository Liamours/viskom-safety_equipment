PACT COMPLIANCE PIPELINE — EVALUATION RESULTS
=============================================

PACT (Pose-Anchored Compliance Tracker) is evaluated at three levels.
Evaluation uses the held-out test split. IoU threshold = 0.50 for detection matching.

Level 1a  Person detection: YOLOv8x-pose predictions vs ground truth person boxes
Level 1b  PPE detection: YOLO26L predictions vs ground truth PPE boxes, per class
Level 2   PPE-to-person assignment: was each detected PPE item assigned to the correct person?


HOW PACT WORKS

1. YOLOv8x-pose runs on the input image to detect persons and extract 17 COCO body keypoints
   per person (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles).

2. YOLO26L (the trained detection model) runs on the same image to find PPE items.

3. Each detected PPE item is assigned to the person whose bounding box has the highest IoU
   with the PPE bounding box. If no person overlaps, the item is assigned to the nearest person.

4. For each person, anatomical anchor regions are computed from their keypoints:
   - helmet_region: derived from face keypoints (nose, eyes, ears), extended upward to person bbox top
   - vest_region: derived from shoulder and torso keypoints
   - gloves_region: derived from wrist keypoints
   (and so on for each PPE type)

5. A PPE item is counted as "worn" only if its bounding box overlaps the correct anchor region
   (IoU ≥ 0.05). This prevents false assignment (e.g., a helmet on the ground credited to a worker).

6. Compliance is determined per-person: score = worn_required_items / total_required_items.
   Overall scene compliance = mean of per-person scores.


OVERVIEW — ALL DATASETS

| Dataset | Split    | Test Images | L1a F1 (person) | L1b Macro F1 (PPE) | L2 Assignment Acc  |
|---------|----------|-------------|-----------------|--------------------|--------------------|
| CHV     | Original | 133         | 0.846           | 0.791              | 93.8%  (408/435)   |
| CHV     | 8020     | 266         | 0.864           | 0.775              | 95.6%  (630/659)   |
| CPPE-5  | Original | 29          | N/A *           | 0.654              | N/A **             |
| SH17    | Original | 200         | 0.833           | 0.521              | 89.2%  (521/584)   |

* CPPE-5 has no person-class annotation. YOLOv8-pose detected persons but there are 0 GT person
  boxes to compare against — this metric cannot be computed.
** CPPE-5 test subset used here is single-person only — assignment accuracy check was skipped.


LEVEL 1A — PERSON DETECTION (YOLOv8x-pose vs ground truth)

| Dataset | Split    | Precision | Recall | F1    | TP  | FP  | FN  |
|---------|----------|-----------|--------|-------|-----|-----|-----|
| CHV     | Original | 0.874     | 0.820  | 0.846 | 369 | 53  | 81  |
| CHV     | 8020     | 0.895     | 0.836  | 0.864 | 671 | 79  | 132 |
| SH17    | Original | 0.874     | 0.796  | 0.833 | 304 | 44  | 78  |

YOLOv8x-pose achieves F1 0.83–0.86 without any dataset-specific fine-tuning — strong given it
is a general pretrained model. Recall is the limiting factor: some persons are missed when they
are small, partially occluded, or at the edge of the frame.


LEVEL 1B — PPE DETECTION PER CLASS (YOLO26L vs ground truth)

CHV — Original split, 133 test images

| Class   | Precision | Recall | F1    |
|---------|-----------|--------|-------|
| helmet  | 0.902     | 0.861  | 0.881 |
| vest    | 0.931     | 0.562  | 0.701 |
| MACRO   | 0.917     | 0.712  | 0.791 |

Helmet detection is strong (F1 0.881). Vest recall is only 0.562 — the model often misses vests,
likely because vests are partially occluded by arms or appear in unusual colors.


CHV — 8020 split, 266 test images

| Class   | Precision | Recall | F1    |
|---------|-----------|--------|-------|
| helmet  | 0.937     | 0.792  | 0.858 |
| vest    | 0.883     | 0.569  | 0.692 |
| MACRO   | 0.910     | 0.680  | 0.775 |

Consistent with original split. Vest recall remains a persistent weakness (0.56–0.57).


CPPE-5 — Original split, 29 test images

| Class        | Precision | Recall | F1    |
|--------------|-----------|--------|-------|
| face_shield  | 0.700     | 0.824  | 0.757 |
| coverall     | 0.789     | 0.667  | 0.723 |
| goggles      | 0.783     | 0.562  | 0.654 |
| mask         | 0.788     | 0.500  | 0.612 |
| gloves       | 0.735     | 0.410  | 0.526 |
| MACRO        | 0.759     | 0.593  | 0.654 |

Face shields score highest (F1 0.757) — they are large and visually distinctive. Gloves are the
hardest (F1 0.526) — small, deformable, and frequently occluded by other objects or the body.


SH17 — Original split, 200-image sample

| Class             | Precision | Recall | F1    |
|-------------------|-----------|--------|-------|
| face              | 0.989     | 0.817  | 0.895 |
| head              | 0.932     | 0.802  | 0.862 |
| helmet            | 0.909     | 0.714  | 0.800 |
| ear               | 0.906     | 0.643  | 0.752 |
| face-guard        | 1.000     | 0.500  | 0.667 |
| face-mask-medical | 0.565     | 0.812  | 0.667 |
| glasses           | 0.778     | 0.571  | 0.659 |
| shoes             | 0.755     | 0.514  | 0.612 |
| hands             | 0.876     | 0.448  | 0.593 |
| medical-suit      | 1.000     | 0.333  | 0.500 |
| tools             | 0.538     | 0.188  | 0.279 |
| gloves            | 0.226     | 0.453  | 0.302 |
| safety-suit       | 0.200     | 0.250  | 0.222 |
| earmuffs          | 1.000     | 0.111  | 0.200 |
| safety-vest       | 0.600     | 0.231  | 0.333 |
| foot              | 0.000     | 0.000  | 0.000 |
| MACRO             | 0.705     | 0.462  | 0.521 |

High-visibility classes (face, head, helmet) score well. Rare items (earmuffs: 0.111 recall,
foot: 0.000) essentially fail due to too few training examples. Gloves have very low precision
(0.226) — many false positives because hands are frequently misclassified as gloves.


LEVEL 2 — PPE-TO-PERSON ASSIGNMENT ACCURACY

Evaluated on multi-person frames only (≥ 2 ground truth persons per image).
A GT PPE item is eligible if a matching detection (IoU ≥ 0.50) was found,
then checked whether PACT assigned it to the correct person.

| Dataset | Split    | Correct | Total | Accuracy |
|---------|----------|---------|-------|----------|
| CHV     | Original | 408     | 435   | 93.8%    |
| CHV     | 8020     | 630     | 659   | 95.6%    |
| SH17    | Original | 521     | 584   | 89.2%    |

CHV achieves 93–95% assignment accuracy. SH17 is lower (89.2%) because scenes are more crowded
and persons overlap more, making the IoU-based assignment ambiguous.

Errors occur mainly when two workers stand very close and a PPE item (e.g. a helmet) partially
overlaps both person bounding boxes — the item goes to the wrong person.


KEY FINDINGS

Person detection (Level 1a):
- YOLOv8x-pose (pretrained, not fine-tuned) achieves F1 0.83–0.86 on CHV and SH17.
- Recall is the bottleneck: small or occluded persons are missed.
- CPPE-5 cannot be evaluated here because the dataset has no person-class ground truth.

PPE detection (Level 1b):
- CHV: helmets are detected reliably (F1 ~0.88). Vest recall is consistently low (~0.56),
  mainly because vests are partially occluded or in unusual lighting.
- CPPE-5: face shields and coveralls are easiest; gloves are hardest (F1 0.526).
- SH17: huge variance across 17 classes. Common classes score well (helmet F1 0.80),
  rare classes nearly fail (foot F1 0.00, earmuffs recall 0.11).

PPE assignment (Level 2):
- Assignment accuracy is high in most cases (89–95%).
- Failures are concentrated in crowded scenes where person bounding boxes overlap significantly.
- SH17 is harder than CHV for this reason (89% vs 93–95%).
