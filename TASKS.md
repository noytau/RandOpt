# TASKS.md — Claude session task backlog

Standing backlog for Claude sessions on this repo. The Working Agreement in `CLAUDE.md` applies
to every task here (code-approval gate, teaching, anchor framing, method of work, pre-experiment
checklist, explicit approval before any cluster submission).

## E1 — Image classification on ImageNet-C, DINOv2-giant (`experiments/E1_imagenet_c.md`)

Four benchmark rungs, same splits (25/10/15 per class) and metric (test top-1). Each follows the
method of work: code → approval → short POC → verify-then-evaluate → approval → larger experiment.

- [ ] **Task 1 — kNN on backbone embeddings** (`scripts/knn_imagenet_c.py` + `vision/features.py`
      cache). POC: fixed k=20/τ=0.07 on gaussian_noise s3 (doubles as base/headroom check).
      Larger: k sweep on val, all 4 corruptions.
- [ ] **Task 2 — linear classifier** (`scripts/probe_imagenet_c.py`, frozen backbone, cached
      features). POC: one fixed config. Larger: sweep on val, all corruptions. Saves head `.pt`
      (loadable via `VisionEngine(linear_init_path=…)`).
- [ ] **Task 3 — finetune** (`scripts/finetune_imagenet_c.py`). Before configs: check free
      GPUs/VRAM + measure giant fwd+bwd throughput, then propose. POC: partial FT (head +
      last-N blocks), one config. Larger: staged unfreezing → **entire-model FT** + sweep on val.
- [ ] **Task 4 — RandOpt** (`scripts/randopt_imagenet_c.py`, drafted + CPU dry-run tested).
      POC: 1–2 cells, N≈20, small K (measures s/perturbation). Larger: grid proposed after
      reviewing the paper's N / top-K / σ / scope choices with the user.

## Queued

- [ ] **Review RandOpt's train/test split design — brainstorm with Nimrod, NOT to be
      implemented automatically.** Open questions: how selection data (scoring gallery/queries
      inside train) should be sized/structured; whether best-single-perturbation stats need a
      held-out guard (the old A/B protocol was removed from the runner on 2026-07-13 — only
      ensemble numbers are quotable until this review happens); what the paper's own split
      protocol implies for ours.

- [ ] **Entire-model FT of giant** — the larger tier of Task 3; propose configs from the POC's
      throughput evidence.
- [ ] **Pretrained-head center variant**: check `facebook/dinov2-giant-imagenet1k-1-layer`
      (Meta's checkpoint = same backbone + a linear head *Meta* trained on ImageNet-1k; loads via
      `Dinov2ForImageClassification`; no training by us). Uses: (1) baseline — its top-1 on the
      E1 splits vs the kNN/probe numbers; (2) RandOpt center that is already task-adapted.
      Needs a `vision/engine.py` ctor option (classification model + `classifier` scope
      bookkeeping) — approval gate applies. Task 2's trained probe head is the cheaper in-house
      alternative center.
