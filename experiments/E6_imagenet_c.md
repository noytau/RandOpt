# E6 — RandOpt vs FT vs Linear Probe on ImageNet-C (DINOv2-giant)

**Status: SPEC — not approved to run.** This file is the task brief for a Claude session. Follow the
Working Agreement in `CLAUDE.md` (code-approval gate for every new function, teach as you go,
anchor framing) and the Pre-experiment checklist. Nothing gets submitted to the cluster without
explicit user approval per submission.

## Motivation — why this experiment

E1–E5 concluded that around DINOv2's **raw SSL init** there is no thicket (flat plateau / needle),
and the licensed next step was: *give RandOpt a task-adapted center to perturb around*. E6 does
exactly that:

- The center is **`facebook/dinov2-giant`, backbone only** (user decision 2026-07-12) — loaded
  exactly as in E5 via `Dinov2Model.from_pretrained`. "Center" = the fixed weight vector RandOpt
  samples Gaussian perturbations around. **No classification head exists in run 1**, so
  classification is head-free via **kNN top-1**: embed the adaptation split as a labeled gallery,
  classify a query image by its nearest gallery neighbors (same scheme as E3's cub_knn/air_knn
  metrics in `scripts/ssl_thicket_profile.py` — reuse that code). No ImageNet training by anyone.
- Deferred to `TASKS.md`: check `facebook/dinov2-giant-imagenet1k-1-layer` (Meta's checkpoint =
  same backbone + a linear head *Meta* trained on ImageNet-1k, loads via
  `Dinov2ForImageClassification`) as an additional baseline and as the *task-adapted center*
  variant.
- The task is **classification under corruption (ImageNet-C)** — unlike CIFAR-10 (base ~98.7%,
  no headroom), corruption drops top-1 substantially, so there is real room to improve.
- Research question (anchor form): *starting from a task-adapted center, how close does
  training-free RandOpt come to fine-tuning and to retraining the linear probe, at the same
  adaptation budget?*
- ⚠ **Framing note (updated 2026-07-12, backbone-only decision):** with a backbone-only center,
  run 1 does **not** test the "task-adapted center" hypothesis — the center is the same raw SSL
  init as E1–E5, evaluated on a new task (corrupted-image kNN classification) that has real
  headroom. Run 1's honest reading: *does the E1–E5 plateau/needle result also hold on a
  classification metric with headroom?* The task-adapted-center test returns when the
  `imagenet1k-1-layer` baseline (`TASKS.md`) is brought in. Confirm the intended claim with the
  user before writing conclusions.

## Pre-experiment checklist (from CLAUDE.md)

### 1. Scale check
DINOv2-giant = **1.136B params** (1,136,482,305 in scope=all, measured in E5). On the paper's
ladder (§6: fails 0.1B, small gains 0.5B, works ~1.5B) giant sits **just below the working
threshold** — the same rung E5 tested. It is the largest DINOv2; if E6 is borderline, the decisive
rung is DINOv3-7B (HF-gated). Scale is therefore *mostly* addressed but say "at the ~1.1B rung"
in any claim.

### 2. Base & improvement check
- **Base** = the unperturbed `dinov2-giant` backbone's **kNN top-1 accuracy on the ImageNet-C
  test split** (gallery = adaptation-split features; `base/test_accuracy`). Every gain is
  measured against this number.
- Paper analogue: the paper measures the pretrained model's task performance as the base; its
  "pretrain NLL" (negative log-likelihood of the pretrained LM) is an *init-quality* measure, not
  a baseline. Our non-generative stand-in for init quality is the base top-1 itself (optionally
  kNN accuracy as a representation probe).
- **Headroom**: DINOv2-g clean ImageNet *kNN* top-1 ≈ 83.5% (DINOv2 paper); under corruption
  (severity 3) expect substantially less, and a 2k-image gallery (≈2 images/class for 1000
  classes) will pull it lower still → **measure the base first**; if a chosen corruption leaves
  <5pp headroom, swap it out; if the tiny gallery makes kNN degenerate, enlarge the gallery
  before touching anything else.
- **Anchor table for E6** (all rungs: same adaptation set, same test split, same metric = top-1):

  | Rung | Method | Trains what | Adaptation budget |
  |---|---|---|---|
  | Baseline 1 | Fine-tune backbone(+head), low LR | backbone weights (gradient) | M images, corrupted |
  | Baseline 2 | Retrain linear head, frozen backbone | head only (gradient) | same M images |
  | Contender | RandOpt around the raw backbone (kNN eval) | nothing (no gradients) | same M images for selection + gallery |

  ⚠ Metric-comparability gap: the contender is scored by kNN, Baseline 2 by its trained head's
  logits. Either evaluate *all* rungs by kNN on the adapted backbone (head-free everywhere) or
  accept the mixed metric and say so — user decision, see Open questions.

### 3. Perturbation-scope check
- Scopes (run 1, backbone-only center): `all` (paper-faithful, ~1.14B scalars) as primary;
  `last_n_blocks` as control. The `classifier` scope only becomes meaningful once the
  pretrained-head variant (`TASKS.md`) is in play. The engine's `set_perturb_scope` already
  supports these (verify the scope filter matches giant's 40-block naming, cf.
  `scripts/count_params.py`).
- σ grid: E5 mapped giant's landscape — peak interest σ ∈ {3e-4, 1e-3, 3e-3}, collapse cliff
  ~1e-2. Use {3e-4, 1e-3, 3e-3} for `all`; the head-only scope tolerates much larger σ (re-derive
  with a 10-perturbation smoke scan before committing a grid).
- N and K: paper uses N=5000, K=top 1–5% — start with N=300/cell for the scan (E5 cadence), and
  only scale the winning cell toward paper-N (pooled across seed-disjoint jobs, per-perturbation
  dumps already supported by the thicket script pattern).
- Selection honesty: the **A/B protocol** (rank on split A, report split B) is **optional in E6**
  — **OFF for the first run** (user decision, for simplicity). The ensemble result stays honest
  without it, because selection uses the adaptation set and reporting uses the untouched test
  split. What is lost without A/B is the *unbiased one-expert stat*: any "best single
  perturbation" number must be labeled optimistic (winner's curse) or simply not quoted. Turn A/B
  back on before making any best-single-perturbation claim. See "Why the A/B split exists" below.

### 4. Ops check
- W&B project `randopt`, run names `e6-*`.
- Cluster flow per CLAUDE.md: `git push` → dedicated `pvc-sync` job (verify HEAD in its logs) →
  compute jobs with `SKIP_SYNC=1`, always `--backoff-limit 0`, different `SEED` per job.
- Giant fits on one A5000/A6000 with BATCH=32 (E5-verified). E5 measured ~40s/perturbation for
  800 images; classification forward is comparable → estimate ~1200 perturbations/scope ≈
  **13–16 GPU-hours per scope** on freely-available nodes. Verify free GPUs and get explicit
  approval before each submit.

## Design

### Data
ImageNet-C (Hendrycks & Dietterich 2019): 15 corruptions × 5 severities applied to the ImageNet
val set (50k images each). **Design decision (default, user may override):**

- Pick **4 corruptions at severity 3** spanning the categories: gaussian_noise (noise), motion_blur
  (blur), fog (weather), jpeg_compression (digital). Run the full comparison per corruption
  (adaptation must not mix corruptions — that's a different, harder question).
- Per corruption, split its 50k images (class-stratified, fixed seed):
  **adaptation M=2,000** (FT + probe training AND RandOpt scoring), **test=5,000** (touched once,
  at the end, by all three rungs). 5,000 test images → binomial 1 SE ≈ 0.7pp at 50% acc;
  pre-register the noise floor accordingly. Two extra splits **A=2,000 / B=2,000** are carved out
  only when the A/B protocol is enabled (OFF in the first run — see §3 and the A/B section below).
- Source: canonical Zenodo tarballs (large) or HF mirror `haideraltahan/wds_imagenetc`
  (webdataset) — **verify the mirror's integrity/coverage before relying on it**.

### Why the A/B split exists (and when you can skip it)
Suppose 1,000 perturbations that each truly do *nothing*, scored on 2,000 images. Each score
still fluctuates ~±1pp by luck (which images happen to land right). The **maximum** of 1,000
lucky coin-flips sits ~+2–3pp above base — it looks like a discovered expert but is pure
selection of noise (the *winner's curse*). Re-score that same winner on a **fresh disjoint split
B** and the fake gain vanishes; a *real* expert keeps its gain on B. This bit us in E1: the best
perturbation was +2pp on its scoring set and **+0pp held-out**. So: rank on A, report the
winner's B score → unbiased "one-expert" gain. The correlation ρ(A,B) across all perturbations is
the bonus diagnostic (ρ≈0 → scores are pure noise; ρ→1 → perturbations have real, reproducible
effects). **When it's safe to skip:** if you only report the *ensemble* number (selected on the
adaptation set, evaluated on the untouched test set), the train/test separation already protects
you — A/B only guards the *best-single-perturbation* claim. First E6 run: A/B OFF.

### Rungs
1. **RandOpt (contender):** perturb around the raw `dinov2-giant` backbone; score each
   perturbation by kNN top-1 on the adaptation set (gallery/query split inside M); top-K
   majority-vote ensemble on test, gallery = adaptation split. (A/B one-expert stat: optional,
   OFF in run 1.)
2. **Linear probe (Baseline 2):** re-train the 1000-way head on the M corrupted images (frozen
   backbone; precompute CLS features once, then it's a cheap logistic regression —
   `vision/train_linear_probe.py` may be reusable). Eval on test.
3. **Fine-tune (Baseline 1):** low-LR gradient FT on the same M images. Full-FT of 1.1B is heavy;
   default to **last-4-blocks + head** FT and label it "leashed FT" honestly (same caveat as E2),
   with full-FT as a stretch goal. Log ‖Δw‖ as equivalent-σ (E2 convention) so RandOpt's search
   radius and FT's movement are comparable.

### Pre-registered decision rules
| Outcome | Reading |
|---|---|
| RandOpt ensemble gain > noise floor and ≥ ~½ of probe's gain | Thicket exists around task-adapted vision centers → paper's story transfers once a trainable component anchors the init |
| RandOpt ≈ 0, probe & FT gain clearly | Needle persists even at a *classification*-adapted center at 1.1B → strengthens E1–E5's "no vision thicket" with the strongest possible setup |
| Probe ≈ FT ≈ 0 | Task has no adaptable signal at this budget → redesign (bigger M, different severity) before concluding anything about RandOpt |

## New code required (each new function passes the approval gate)
1. `data_handlers/imagenet_c.py` + reward file + registry entry (3-step guide in
   `CUSTOM_DATASET_GUIDE.md`) — download/parse, corruption+severity selection, stratified splits.
2. kNN scoring path for ImageNet-C: reuse the kNN machinery from `scripts/ssl_thicket_profile.py`
   (E3) rather than writing new code; new glue functions still pass the approval gate.
   (The `Dinov2ForImageClassification` engine option is deferred with the pretrained-head
   baseline — see `TASKS.md`.)
3. FT baseline script (Baseline 1) — new script, full gate.
4. Runner/glue: `randopt_vision.py` already does classification + majority vote; extend rather
   than fork if possible.

## Order of work
1. Data handler + local smoke on a few hundred images (CPU/1-GPU) — verify base top-1 per
   corruption and confirm headroom (checklist §2).
2. σ smoke scan (N=10/cell) per scope → freeze grids.
3. Probe + leashed-FT baselines (cheap, run first — they define the target RandOpt must chase).
4. RandOpt scan (N=300/cell), then scale the winning cell.
5. Update `results/report.html` (house policy) and the advisor artifact with an E6 section.

## Open questions for the user
- Metric comparability across rungs: evaluate all rungs by kNN (head-free everywhere), or let
  Baseline 2 use its trained head and report the mixed metric honestly?
- Which corruptions/severity? (default above: 4 corruptions, severity 3)
- Adaptation budget M? (default 2,000 — small enough to be "few-shot-ish", big enough for a
  stable probe)
- Is leashed FT (last-4 + head) acceptable for Baseline 1, or is full FT required?
