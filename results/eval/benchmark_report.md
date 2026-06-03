Runtime Benchmark Report
========================

Script: code/experiment/benchmark_all.py
Output: results/eval/benchmark_all_summary.json
GPU: NVIDIA RTX 4050 Laptop (6 GB)
Warmup: 3 images discarded per dataset
Timing: each stage measured once per image (no double-inference)

---

Per-Dataset Results

Dataset   Images   PPE det   Pose det   Anchor+compliance   Report   Total     FPS
CPPE-5    29       29.80 ms  43.59 ms   0.41 ms             0.13 ms  73.93 ms  13.53
CHV       133      33.28 ms  46.64 ms   0.79 ms             0.15 ms  80.86 ms  12.37
SH17      1620     29.12 ms  62.19 ms   0.39 ms             0.13 ms  91.83 ms  10.89
Overall   1782     29.44 ms  60.73 ms   0.42 ms             0.13 ms  90.72 ms  11.02

---

Previous Benchmark Error

Old benchmark.py called _detect_ppe and _detect_persons standalone, then called pipeline.run()
which internally runs both models again. This caused three model inference runs per image.

Old reported numbers (CPPE-5 only, n=29):
  Anchor+compliance: 65.79 ms  <- inflated by second ppe+pose inference pass
  Total: 140.87 ms
  FPS: 7.1

Corrected numbers (all datasets, n=1782):
  Anchor+compliance: 0.42 ms   <- pure Python bbox/IoU logic
  Total: 90.72 ms
  FPS: 11.02

---

Findings

Bottleneck is pose detection (YOLOv8x-pose), not anchor construction.
  Pose = 60.73 ms (67% of total)
  PPE det = 29.44 ms (32% of total)
  Anchor+compliance = 0.42 ms (<1% of total)
  Report = 0.13 ms (<1% of total)

SH17 pose inference slower (62.19 ms) than CPPE-5 (43.59 ms) — SH17 images larger resolution.
Anchor logic is resolution-independent; stays flat at 0.4-0.8 ms across all datasets.

Reviewer claim "anchor stage slower than both NN stages combined" is factually incorrect.
0.42 ms vs 90.17 ms combined NN inference.

---

Correction for Paper

Table VI should reflect corrected stage timings measured across all three datasets.
FPS reported as 11.02 overall (vs 7.1 previously).
