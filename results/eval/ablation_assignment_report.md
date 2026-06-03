# Assignment Accuracy Ablation Report

**Script**: `code/experiment/ablation_assignment.py`
**Output**: `results/eval/ablation_assignment.csv`
**Metric**: Assignment accuracy = correct / total (multi-person frames only, ≥2 GT persons and ≥2 predicted persons)
**Fixed**: τ_v = 0.30
**Sweep**: τ_a ∈ {0.05, 0.10, 0.15, 0.20} × δ ∈ {0.05, 0.10, 0.15, 0.20, 0.25} = 20 configs per dataset

---

## CHV (94 multi-person images)

| τ_a \ δ | 0.05   | 0.10   | 0.15   | 0.20   | 0.25   |
|---------|--------|--------|--------|--------|--------|
| 0.05    | 0.9845 | 0.9820 | 0.9822 | 0.9822 | 0.9773 |
| 0.10    | 0.9844 | **0.9844** | 0.9820 | 0.9821 | 0.9795 |
| 0.15    | 0.9842 | 0.9842 | 0.9844 | 0.9819 | 0.9793 |
| 0.20    | 0.9865 | 0.9839 | 0.9840 | 0.9843 | 0.9791 |

- **Default (τ_a=0.10, δ=0.10)**: 379/385 = **0.9844**
- Range across all 20 configs: 0.9773 – 0.9865 (spread = 0.0092)
- No config drops below 0.977

## SH17 (582 multi-person images)

| τ_a \ δ | 0.05   | 0.10   | 0.15   | 0.20   | 0.25   |
|---------|--------|--------|--------|--------|--------|
| 0.05    | 0.9739 | 0.9722 | 0.9653 | 0.9657 | 0.9607 |
| 0.10    | 0.9617 | **0.9723** | 0.9734 | 0.9632 | 0.9614 |
| 0.15    | 0.9521 | 0.9598 | 0.9668 | 0.9692 | 0.9617 |
| 0.20    | 0.9462 | 0.9557 | 0.9634 | 0.9636 | 0.9659 |

- **Default (τ_a=0.10, δ=0.10)**: 246/253 = **0.9723**
- Range across all 20 configs: 0.9462 – 0.9739 (spread = 0.0277)
- No config drops below 0.946

---

## Observations

**τ_a effect (SH17)**: Higher τ_a rejects more PPE-person pairs (fewer `total`). At τ_a=0.20, δ=0.05: only 130 pairs evaluated vs 307 at τ_a=0.05, δ=0.05. Accuracy stays ≥0.946 but coverage shrinks.

**δ effect (SH17)**: Larger δ expands anchor boxes → more pairs pass → `total` grows. At τ_a=0.05: total grows from 307 (δ=0.05) to 458 (δ=0.25) while accuracy drops from 0.9739 to 0.9607.

**CHV**: Nearly flat across all configs (0.9773–0.9865, spread <0.01). Simple two-person construction scenes; PPE-to-person separation is unambiguous regardless of threshold choice.

**Default params are near-optimal**: τ_a=0.10, δ=0.10 ranks among the best configs on both datasets without tuning.

---

## δ Sensitivity on F1 (CPPE-5, ablation.py)

Sweep: δ ∈ {0.10, 0.15, 0.20, 0.25} × τ_a ∈ {0.01, 0.05, 0.10, 0.20} × τ_v ∈ {0.20, 0.30, 0.40, 0.50}. Metric: compliance F1.

F1 varies only with τ_v. δ and τ_a produce **zero change** at every τ_v level:

| τ_v  | δ = 0.10 | δ = 0.15 | δ = 0.20 | δ = 0.25 |
|------|----------|----------|----------|----------|
| 0.20 | 0.6165   | 0.6165   | 0.6165   | 0.6165   |
| 0.30 | 0.6217   | 0.6217   | 0.6217   | 0.6217   |
| 0.40 | **0.6320**   | **0.6320**   | **0.6320**   | **0.6320**   |
| 0.50 | 0.6296   | 0.6296   | 0.6296   | 0.6296   |

*(Each cell is constant across all 4 τ_a values as well — 64 configs collapse to 4 unique F1 values.)*

- δ effect on F1: **0.0000** (no variation)
- τ_a effect on F1: **0.0000** (no variation)
- τ_v is sole driver: F1 range 0.6165 – 0.6320 (spread = 0.0155)
- Best: τ_v = 0.40 → F1 = 0.6320

---

## Summary

Assignment accuracy exceeds **0.97 on both datasets** at default settings and remains above **0.94 across the entire 4×5 grid**. The system is robust to τ_a and δ variation; no threshold combination causes collapse.
