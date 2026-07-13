# ARCHIVE — Old Vision Experiments (CIFAR-10 classification & SPair-71k correspondence)

**Status: DISCARDED (user decision 2026-07-13).** This file archives the 2026-07 vision
experiment series (internally numbered E1–E5 *at the time*; that numbering is now retired — the
current series restarts at E1 = ImageNet-C, `experiments/E1_imagenet_c.md`). Kept only as
historical record; not relevant to current work. The scripts/handlers listed below were deleted
from the working tree in the same commit that added this file — recover any of them with
`git log --diff-filter=D` / `git checkout <sha>^ -- <path>`.

## What was studied

RandOpt (Neural Thickets, arXiv 2603.12228) applied to DINOv2 on: CIFAR-10 / CUB-200 /
FGVC-Aircraft classification and SPair-71k semantic correspondence (PCK@0.1, no head).

## Removed files

| File | Was |
|---|---|
| `randopt_vision.py` | DINOv2+linear-head classification runner (CIFAR-10 etc.) |
| `randopt_correspondence.py` | SPair-71k correspondence runner (PCK from patch embeddings) |
| `scripts/randopt_corr_thicket.py` + `run_corr_thicket.sh` | thicket-existence scan (scope×σ, A/B held-out splits, one-expert stat, ρ(A,B)) |
| `scripts/grad_reachability.py` + `run_grad_reachability.sh` | gradient-reachability control (plateau vs needle), soft-argmax/InfoNCE PCK surrogates, ‖Δw‖→equivalent-σ |
| `scripts/ssl_thicket_profile.py` + `run_ssl_profile.sh` | multi-task thicket profile (PCK + kNN/mAP panel) |
| `scripts/recenter_thicket.py` + `run_recenter_thicket.sh` | thicket-emergence curve along a gradient trajectory |
| `data_handlers/{cifar10,cub200,fgvc_aircraft,spair71k}.py` | dataset handlers |
| `utils/reward_score/vision.py` | classification reward |
| `VISION.md` | CIFAR-10 vision-extension design doc |

## Key findings (final state, 2026-07-08)

- **CIFAR-10:** DINOv2+probe base ~98.7% at ceiling → no headroom, gain ~0.
- **SPair-71k thicket sweep (~36k perturbations, scope×σ, A/B honesty):** best unbiased
  one-expert gain +0.34pp, inside the ~0.8pp noise floor → **no thicket** around the raw SSL
  init for correspondence. scope→cliff mapped: `all` collapses between σ=1e-3 and 3e-3
  (DINOv2-base); `last1` robust to σ=0.1.
- **Gradient-reachability control: NEEDLE, not plateau.** Same scope/radius, gradient ascent
  gained +6.4…+10.5pp held-out (best 58.37→68.88% PCK at ‖Δw‖ inside the σ=1e-3 ball) — better
  models exist at the searched radii but occupy negligible measure under isotropic sampling.
- **Multi-task profile:** no thicket on any axis (spatial or discriminative); the "thickets live
  along the pretext axes" hypothesis refuted.
- **Recenter/emergence curve:** task-adapted centers (50/150/300 gradient steps, PCK
  58.4→68.7%) are still needles — thicket gain stays <0.4pp, ρ≈0.
- **Scale ladder (large 0.304B / giant 1.136B):** flat at every tested scale; giant's collapse
  cliff sits at ~1e-2 (vs base ~3e-3) — more robust, but no scaling-law signal; scale confound
  closed at N=300 (paper-N pooling campaign `e5g-1e3-d{1..5}` was in flight when archived).
- **Unified conclusion:** the thicket phenomenon was absent from DINOv2's SSL landscape on these
  tasks — good models exist but have negligible measure under isotropic sampling; untested lever
  was structured (non-isotropic) perturbation.

## Where the numbers live

- W&B project `randopt`, runs `thicket-*`, `grad-reach-*`, `ssl-profile-*`, `recenter-thicket`,
  `e5-*`, `e5g-*`.
- `results/report.html` (SSL Correspondence section), `results/corr_thicket_*/results.json`,
  `results/grad_reach_last1_results.json`, `results/ssl-profile-*/`, `results/recenter-thicket/`,
  `results/e5-*/`.
- Presentation artifact (advisor-ready): https://claude.ai/code/artifact/d3b77bb8-4333-4b04-8cec-e4b198e55e5a

## Methodological notes worth remembering (survive the archive)

- **A/B honesty split** (rank on split A, report the winner's B score): our addition, not the
  paper's — it defeats the winner's curse on *best-single-perturbation* claims. Removed from the
  current ImageNet-C runner; reintroduce only after the split-design review with Nimrod
  (TASKS.md).
- **ρ(A,B) diagnostic:** ρ≈0 → perturbation effects are pure noise; high ρ with negative mean →
  reproducible-but-harmful; no regime ever showed high ρ *and* positive gain.
- **Ops lessons** are kept in CLAUDE.md (they are about the cluster, not these experiments).
