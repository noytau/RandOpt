# TASKS.md — Claude session task backlog

Standing backlog for Claude sessions on this repo. The Working Agreement in `CLAUDE.md` applies
to every task here (code-approval gate for new functions, teach as you go, anchor framing,
pre-experiment checklist, explicit approval before any cluster submission).

## Open

- [ ] **E6 follow-up: check `facebook/dinov2-giant-imagenet1k-1-layer` as a baseline** (added
      2026-07-12). This is Meta's published checkpoint = the same dinov2-giant backbone + a
      linear head *Meta* trained on ImageNet-1k (loads via `Dinov2ForImageClassification`; no
      training by us). Two uses:
      1. **Baseline:** its top-1 on the E6 ImageNet-C splits, next to run-1's backbone-kNN
         numbers — quantifies what a properly trained head buys under corruption.
      2. **Task-adapted center:** rerun the E6 RandOpt protocol perturbing around
         backbone+pretrained-head — this is the E1–E5-licensed test of "thickets appear around
         task-adapted inits" that run 1 (backbone-only) does not cover.
      Needs: `vision/engine.py` ctor option to load the classification model (touches
      perturbation-scope bookkeeping, incl. making the `classifier` scope point at Meta's head) —
      new-function approval gate applies. Context: `experiments/E6_imagenet_c.md`.
