# E1 — Image Classification on an SSL Vision Model: kNN / Linear Probe / Finetune / RandOpt (DINOv2-giant on ImageNet-C)

**Status: APPROVED PLAN — in implementation.** First experiment of the RandOpt SSL Vision series.
Follow the Working Agreement in `CLAUDE.md` (code-approval gate, teaching pace, anchor framing,
**method of work**, pre-experiment checklist). Nothing is submitted to the cluster without
explicit user approval per submission. Task tracking: `TASKS.md`.

Paper reference: Neural Thickets — arXiv 2603.12228, https://arxiv.org/abs/2603.12228.

## Dataset — ImageNet-C

ImageNet-C (Hendrycks & Dietterich, ICLR 2019): the ImageNet-1k **validation** set
(50,000 images = 1,000 classes × exactly 50 images/class) with **15 corruption types × 5
severities** applied. Each (corruption, severity) pair is a full 50k copy, shipped as 224×224
JPEGs (DINOv2's native input size). Source: canonical Zenodo tarballs (record 2235448), one tar
per corruption family; the handler downloads a family tar and extracts only the requested
corruption/severity subtree (`data_handlers/imagenet_c.py`).

Canonical ImageNet-C usage is **evaluation-only** (train on clean ImageNet, test on corruptions),
so no standard train/val/test split of it exists. Our setting is **supervised adaptation on
corrupted data** (closer to domain / test-time adaptation), so we define our own split.

**Splits (user-approved): 25 train / 10 val / 15 test per class** — class-stratified, seeded,
disjoint, contiguous ranges of a per-class shuffle. Val is *used only by larger experiments*
(hyperparameter selection); short POCs run one fixed config. Test is touched once per experiment.
15k test → binomial 1 SE ≈ 0.4pp at 50% accuracy.

## Model

`facebook/dinov2-giant`: 1.136B params, backbone only (no classification head; classification is
done via kNN on embeddings or a trained linear head, per rung). On the paper's scaling ladder
(fails ~0.1B, small gains ~0.5B, works from ~1.5B) giant sits **at the ~1.1B rung** — say so in
any claim. The largest DINOv2; the decisive larger rung would be DINOv3-7B (HF-gated).

## The four rungs (same splits, same metric: top-1 on test)

| Rung | Method | Trains what | Script |
|---|---|---|---|
| 1 | **kNN** on backbone embeddings (hyperparameter k) — doubles as base/headroom readout | nothing | `scripts/knn_imagenet_c.py` |
| 2 | **Linear classifier** (frozen backbone) | head only (gradient) | `scripts/probe_imagenet_c.py` |
| 3 | **Finetune** (some layers first; entire model in larger tier) | backbone(+head) (gradient) | `scripts/finetune_imagenet_c.py` |
| 4 | **RandOpt** (perturb → select on train → majority-vote ensemble on test) | nothing (no gradients) | `scripts/randopt_imagenet_c.py` |

Rung details and tiers: see the approved plan summary in `TASKS.md` and the method-of-work rules
in `CLAUDE.md` (POC = one fixed config, no val selection; sweeps proposed per rule ⑤ before each
larger tier — no hyperparameter ranges are pre-committed here).

Notes:
- **Shared feature cache** (`vision/features.py`): kNN/probe never modify the backbone → each
  split's embeddings are computed once (~40 GPU-min/corruption) and cached
  (`results/features/...`); afterwards k-sweeps are seconds and probe training minutes. FT and
  RandOpt cannot use the cache in their core loops (they move backbone weights).
- **kNN readout convention**: DINO-style weighted vote (k=20, τ=0.07 as protocol starting point);
  math identical to `VisionEngine.eval_global` / `knn_predict` for parity with the RandOpt rung.
- **Selection honesty (RandOpt)**: selection uses only train-split data; test is evaluated once
  by the ensemble — so **only ensemble numbers are quotable**. Best-single-perturbation scoring
  maxima are inflated by the winner's curse and must not be quoted. The train/test split design
  for RandOpt is queued for a brainstorm with Nimrod before any larger tier (`TASKS.md`).
- **Head reuse**: Task 2's trained head `.pt` loads into `VisionEngine(linear_init_path=…)` and
  is the in-house option for a task-adapted RandOpt center later (see TASKS.md queued items).

## POC corruption

`gaussian_noise` severity 3. Headroom check lives in the Task-1 POC: if the kNN base leaves <5pp
headroom, swap the corruption before proceeding. Larger tier adds `motion_blur`, `fog`,
`jpeg_compression` at severity 3 (one adaptation per corruption; no mixing).

## Ops

- W&B project `randopt`, run names `e1-<task>-<corruption>-s<severity>`; results in
  `results/e1-*/results.json`.
- Cluster flow per `CLAUDE.md`: `git push` → dedicated `pvc-sync` job (verify HEAD in its logs) →
  compute jobs with `SKIP_SYNC=1`, always `--backoff-limit 0`. Entry point:
  `scripts/run_e1_baselines.sh` (`TASK=knn|probe|ft|randopt`).
- First real run needs the Zenodo download (noise.tar ≈ 21GB): run it as a dedicated CPU job so
  no GPU idles through the fetch.
- Pre-experiment checklist (`CLAUDE.md`) answered in writing before every submission; free-GPU
  check + cost estimate + explicit user approval per submit.
