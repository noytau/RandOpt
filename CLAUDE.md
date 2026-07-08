# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

RandOpt implements **Neural Thickets** (paper: arxiv 2603.12228): instead of gradient-based fine-tuning, it randomly perturbs a pretrained LLM's weights using Gaussian noise, evaluates each perturbation on a small train set, selects the top-K perturbations by reward, and uses majority voting (ensemble) over these perturbed models to answer test queries. The key insight is that high-quality task-adapted models are densely packed around pretrained weights.

**Vision / SSL extension (branch `feature/vision-randopt`):** The same algorithm applied to DINOv2 — classification (CIFAR-10, CUB-200, FGVC-Aircraft) and semantic correspondence (SPair-71k, no head). See [`VISION.md`](VISION.md) and the **Vision & SSL Correspondence Experiments** section below. Headline finding: RandOpt does **not** transfer to DINOv2's SSL init — the reachable weight neighborhood is a flat plateau, not a thicket of better models (see findings below).

## Working Agreement (read first — set by the user 2026-07-08)

1. **Code approval gate:** every NEW function added to this project must be shown to and approved by the user BEFORE it is committed. Present the function (signature + body + why it's needed), wait for explicit approval, then commit. Small mechanical edits to existing code (renames, param defaults, log lines) don't need a gate, but new functions/scripts always do.
2. **The user is learning:** they have ML-engineering experience but are new to research practice. Explain research framings (baseline vs control, probe conventions, why a design choice exists) in plain language, define terms on first use, and be patient. The goal is for the user to learn, then drive the research themselves.
3. **Always return to the base experiment.** Every result, plan, or new idea must be framed against the project's anchor comparison, on the same task/metric/splits:
   | Rung | Method | Trains what |
   |---|---|---|
   | Baseline 1 | **Fine-tuning** the pretrained model (low LR) | backbone weights (gradient) |
   | Baseline 2 | **Linear probe** (train a linear head, frozen backbone) | head only (gradient) |
   | Contender | **RandOpt** (random perturbation + selection) | nothing (no gradients) |
   The research question is always: *how close does training-free RandOpt come to the two standard SSL adaptation baselines?* (Current state: E2 is a *small/leashed* FT used as a control, not a full Baseline-1; a proper linear-probe rung on the correspondence task doesn't exist yet — both would need user approval to build.)

## Running Experiments

**Local (no SLURM):**
```bash
bash scripts/local_run.sh
# or directly:
python3 randopt.py --dataset countdown --model_name Qwen/Qwen2.5-3B-Instruct \
  --num_engines 4 --population_size 500 --sigma_values "0.0005,0.001,0.002"
```

**On a cluster (SLURM):**
```bash
sbatch scripts/single_node.sh     # single node
sbatch scripts/multiple_nodes.sh  # multi-node
```
Fill in the SBATCH headers and path variables in those scripts before submitting.

**Resume from a previous run** (skips sampling, re-runs ensemble eval):
```bash
python3 randopt.py --resume_dir path/to/previous/experiment_dir --experiment_dir new_dir
```

**Docker:**
```bash
# Build from parent of RandOpt/
docker build -f RandOpt/docker/Dockerfile_vllm -t randopt-vllm:latest .
docker run -it --gpus all randopt-vllm:latest bash
```

**Install:**
```bash
pip install -r requirements.txt
```

## Key CLI Arguments

| Arg | Default | Notes |
|-----|---------|-------|
| `--dataset` | `gsm8k` | One of the registered datasets (see below) |
| `--model_name` | `Qwen/Qwen2.5-3B-Instruct` | HF model ID or local path |
| `--population_size` | `30` | Total perturbations to evaluate (paper uses 5000) |
| `--sigma_values` | `0.0001,...,0.01` | Noise scales to search over |
| `--top_k_ratios` | `0.01,0.05,0.1` | Fractions of population for ensemble sizes |
| `--num_engines` | `4` | vLLM engines = GPUs / TP |
| `--tp` | `1` | Tensor parallel size; use 2+ for 7B+ models |
| `--train_samples` | `200` | Samples used for perturbation scoring |

## Architecture

### Main loop (`randopt.py`)
1. **Load data** via `DatasetHandler`
2. **Evaluate base model** — records train reward and test accuracy
3. **Perturbation sampling** — for each of `population_size` (seed, sigma) pairs, perturbs weights in-place, generates on train set, restores weights
4. **Selection** — sorts all (seed, sigma) pairs by train reward, picks top-K
5. **Ensemble evaluation** — runs each top-K model on the test set, extracts answers, majority-votes per question

### Engine layer (`core/engine.py`)
- `launch_engines()` creates N Ray-managed vLLM instances, each on its own placement group. Engines are initialized in batches to avoid NFS contention.
- `WorkerExtension` (`utils/worker_extn.py`) is injected into vLLM workers via `worker_extension_cls`. It adds:
  - `perturb_self_weights(seed, sigma)` / `restore_self_weights(seed, sigma)` — stateless, deterministic perturbation using per-seed `torch.Generator`
  - `store_base_weights()` / `apply_perturbation()` / `reset_to_base_weights()` — alternative ensemble path that avoids add/subtract drift
  - `update_weights_from_seeds()` — ES-style gradient update from ranked seeds
  - `broadcast_all_weights()` / `save_self_weights_to_disk()` — multi-node and checkpointing utilities

### Dataset handlers (`data_handlers/`)
- All handlers inherit `DatasetHandler` (abstract base in `data_handlers/base.py`)
- Required methods: `load_data()`, `compute_reward()`, `extract_answer()`
- Optional overrides: `extract_answer_for_voting()`, `format_answer_for_check()`, `is_answer_correct()`, `is_voted_answer_correct()`
- Registered in `data_handlers/__init__.py` — add new datasets there

### Reward scoring (`utils/reward_score/`)
- One file per dataset; each exports `compute_score(response, ground_truth) -> float`
- Imported by the corresponding handler

### Adding a new dataset
Follow the 3-step guide in `CUSTOM_DATASET_GUIDE.md`:
1. Put data JSON in `data/your_dataset/`
2. Add reward function in `utils/reward_score/your_dataset.py` and handler in `data_handlers/your_dataset.py`
3. Register in `data_handlers/__init__.py`

## Vision & SSL Correspondence Experiments

Branch `feature/vision-randopt`. These apply RandOpt beyond LLMs, reusing the `DatasetHandler` + W&B plumbing but swapping vLLM for a PyTorch+Ray `VisionEngine` (`vision/engine.py`, launched via `vision/launch_vision_engines`).

### Entry points
| Script | What it does |
|--------|--------------|
| `randopt_vision.py` | DINOv2 + linear head **classification** (CIFAR-10, CUB-200, FGVC-Aircraft). Perturb backbone+head, ensemble by majority vote. |
| `randopt_correspondence.py` | DINOv2 **semantic correspondence** on SPair-71k. **No head** — reward = PCK@0.1 from cosine similarity of patch embeddings. Ensemble by averaging (N,256,256) similarity matrices then argmax. |
| `scripts/randopt_corr_thicket.py` | **Thicket-existence study**: scope×σ scan, each perturbation scored on two disjoint held-out splits A/B, reports unbiased "one-expert" gain + Spearman ρ(A,B). No selection/ensemble. |
| `scripts/grad_reachability.py` | **Gradient-reachability control (plateau vs needle)**: gradient ascent on a differentiable PCK surrogate (soft-argmax / InfoNCE) under the same scope/eval/A-B protocol as the thicket sweep; logs ‖Δw‖ as equivalent-σ. Distinguishes "no better model nearby" from "better models exist but isotropic sampling can't find them". Run via `scripts/run_grad_reachability.sh`. |
| `scripts/ssl_thicket_profile.py` | **Multi-task thicket profile (E3)**: each perturbation scored on a task panel (SPair PCK + kNN/retrieval-mAP on CUB-200 & FGVC-Aircraft, shared forwards) under per-task A/B honesty. Tests "thickets live along the axes the init was adapted for"; also logs cross-task gain correlations. Cells are `scope:sigma` tokens. Run via `scripts/run_ssl_profile.sh`. |
| `scripts/recenter_thicket.py` | **Thicket-emergence curve (E4)**: runs the E1 random-perturbation thicket protocol at several points along the E2 gradient trajectory (0/50/150/300 steps). Tests whether task-adaptation moves the init from a needle regime into a thicket — one-expert gain measured vs *each adapted center's* own PCK. Reuses E2's parity-checked eval; walks the trajectory incrementally. Run via `scripts/run_recenter_thicket.sh`. |
| `scripts/count_params.py` | CPU param inventory (per-block counts, validates `last_n_blocks` scope filter). |

### VisionEngine (`vision/engine.py`)
- `perturb_target`: `"all"` (86.58M backbone+head) · `"classifier"` (head only) · `"last_n_blocks"` (last N transformer blocks; DINOv2-base = 12 blocks named `encoder.layer.{i}`, ~7.09M each).
- `set_perturb_scope(target, n)` — switch scope on a **live** actor (no model reload); `count_perturb_params()` reports in-scope scalar count.
- `perturb_weights/restore_weights(seed, sigma)` — same per-seed `torch.Generator` scheme as the LLM path (bit-exact restore).
- `get_patch_features(imgs)` → `(N, 256, 768)` L2-normalized patch tokens (skips CLS token 0).
- `eval_pck(...)` — computes PCK@0.1 **inside the actor** and returns only the scalar. Bulk `.cpu()` once then CPU PCK — avoids both the ~157MB/​call Ray tensor transfer **and** the per-keypoint GPU→CPU sync that `int()` triggers on GPU tensors. (Note: on A5000/A6000 the forward pass still dominates at ~8s/perturbation for 800 imgs; batch size barely matters.)

### Datasets added (`data_handlers/`)
- `spair71k` — SPair-71k correspondence. Downloads from POSTECH (`SPair-71k.tar.gz`, use `wget -c`), parses `PairAnnotation/{trn,val,test}/*.json`; fields are `src_kps`/`trg_kps`, images under `JPEGImages/<category>/`, image size from PIL (`.size` = w,h; the JSON `*_imsize` is `[w,h,c]`). PCK threshold = `0.1·max(bbox_w,bbox_h)·GRID` (GRID=16).
- `cifar10`, `cub200`, `fgvc_aircraft` — classification handlers (classification track largely superseded by correspondence).

### Thicket study design
Each perturbation is scored on **two disjoint held-out splits A and B**. Rank on A, report the winner's **B**-score → unbiased one-expert gain (defeats the winner's-curse that inflated the `max`). ρ(PCK_A, PCK_B) across perturbations is the **selection-generalization** signal: ρ≈0 = differences are pure noise; ρ→1 = perturbations have real reproducible effects. This A/B split is our addition, not from the paper.

Run via `scripts/run_corr_thicket.sh`, tuned by env vars: `MODEL` (HF id, default `facebook/dinov2-base`; `dinov2-large`/`dinov2-giant` work — giant's SwiGLU FFN is supported by `Dinov2Model`, use `BATCH=32`), `SCOPES` (e.g. `all,last2,last1`), `NPOP` (N per cell), `SIGMAS`, `NA`/`NB` (held-out sizes), `BATCH`, `SEED` (**use different seeds across jobs** so draws are independent and pool to a larger effective N), `RUNNAME`, `SKIP_SYNC`.

### Key findings (2026-07, sweep COMPLETE — all 5 jobs Succeeded)
- **CIFAR-10:** DINOv2+probe base ~98.7% is at ceiling → RandOpt gain ~0 (no headroom).
- **SPair-71k correspondence:** base PCK@0.1 ~54–58%. Full-scope N=500 ensemble ≈0 gain; best single perturbation +2pp on the scoring set but **+0 on held-out** (winner's-curse).
- **Thicket sweep — FINAL (scope×σ, ~36k perturbations, N=1000 broad + N=3000 deep, σ 3e-4…1e-1, held-out A/B):** best unbiased one-expert gain across everything = **+0.34pp** (last1, σ=1e-3/3e-3, N=3000) — inside the ~0.8pp noise floor → **no expert, no thicket**. Every job printed *"no individual gain → hypothesis b"*.
  - **scope→cliff fully mapped:** `all` collapses (→~9%, random) between σ=1e-3 (−4.6pp) and σ=3e-3 (−45pp); `last2` tolerates up to ~1e-3 then degrades; `last1` is remarkably robust — only −2.7pp even at σ=0.1 (last-block contribution to correspondence is diffuse/redundant).
  - **ρ diagnostic:** where perturbations have real reproducible effect (high ρ, e.g. `all` σ=1e-3 ρ=0.63) the effect is **negative**; where harmless (low σ) ρ≈0.05 (noise). No regime has high ρ *and* positive gain.
  - **deep N=3000 settles it:** 4× more draws at the best cells moved the max from +0.22 → +0.34pp (still noise) — more sampling does not uncover an expert.
- **Conclusion (hypothesis "b", definitive):** DINOv2's SSL init is a **flat plateau** for frozen-backbone/no-head correspondence, *not* a thicket of better models — unlike instruct-LLMs which are already task-adapted. Making RandOpt work on SSL needs a **trainable component to perturb around** (trained correspondence head or partial fine-tune) so the base sits in a task-thicket. This is the recommended next direction; the raw-SSL-backbone approach is a confirmed dead end.
- **Gradient-reachability control (E2, 2026-07-06, `grad-reach-last1`): verdict = NEEDLE, not plateau.** Gradient ascent on differentiable PCK surrogates (soft-argmax / InfoNCE), same last1 scope, same eval, same A/B protocol (eval parity certified: base_B = 58.37% exact). **All 6 configs (2 losses × 3 LRs) gained +6.4 to +10.5pp held-out** — best: softargmax lr=3e-5, **+10.51pp (58.37→68.88%)** at ‖Δw‖ ≈ 1.9, i.e. *inside* the σ=1e-3 ball (radius 2.66) where the sweep's 3000+ draws found at most +0.34pp. So better task models **exist** at the searched radii but occupy negligible measure under isotropic sampling: **thickets require the good set to have non-negligible measure, not merely to exist.** Also a standalone side-finding: no head, last-block-only weight movement gives +10.5pp on SPair-71k. Numbers: `results/grad_reach_last1_results.json`, W&B run `grad-reach-last1`. Licensed next step: structured perturbation geometry (gradient/Fisher-subspace or low-rank ΔW sampling around init).
- **Multi-task profile (E3, 2026-07-07, `ssl-profile-{last1,last2,all}`): no thicket on ANY axis; H1 refuted.** Each perturbation scored on SPair PCK + kNN/retrieval-mAP for CUB-200 & FGVC-Aircraft (bases: pck 58.4, cub_knn 75, air_knn 25 — big headroom). Across all scope×σ cells, no task's one-expert held-out gain clears its noise floor. Only recurring positive is `air_knn` at high σ (+1.6…+2.6pp) but kNN-top1 on 500 queries has a ~2pp binomial floor (A/B bases themselves swing 1–3pp) → ≈1 SE, not real; `cub` collapses at high σ. **The predicted "thickets live along the discriminative pretext axes" is false — needle/plateau on spatial AND discriminative axes alike.** Numbers: `results/ssl-profile-*/results.json`.
- **Thicket-emergence curve (E4, 2026-07-07, `recenter-thicket`): needle PERSISTS across adaptation.** Ran the E1 random-perturbation protocol at 0/50/150/300 gradient steps along the E2 trajectory (parity base_B=58.37). Center PCK climbs 58.37→66.28→68.12→**68.69%** (adaptation works), but best thicket gain stays flat: +0.28/+0.20/+0.16/+0.37pp (all < 0.8pp floor, ρ≈0). **A genuinely task-adapted DINOv2 is still a needle** — gradient reaches a better *isolated* point, not a thicket. Directly answers "is there a thicket at the adapted center only gradient reaches": no. Numbers: `results/recenter-thicket/results.json`.
- **Unified conclusion (E1–E4):** the thicket phenomenon is absent from DINOv2's SSL landscape. Good models exist (E2 needle) but have negligible measure; that measure favours no task axis (E3) and does not grow with task-adaptation (E4). Open lever: **structured (non-isotropic) perturbation** — gradient/Fisher-subspace or low-rank ΔW sampling — the only untested way to reach the low-measure good set by random search. **⚠ Known confound: model scale (addressed by E5, in flight).** The paper's own scaling analysis (§6) says RandOpt *fails* at GPT-2 0.1B, gives small gains at Qwen 0.5B, and only works from **~1.5B** — and DINOv2-base is **0.086B, below their canonical failure case**. Until E5 lands, "SSL needs task-adaptation" and "thickets need scale" both fit the E1–E4 data; do NOT present the departure claim without this caveat.
- **Scale ladder (E5, 2026-07-08): FLAT AT EVERY TESTED SCALE — scale confound mostly killed; σ-extension in flight.** Large (0.304B, base_B=55.54): best +0.02pp, and σ=1e-3 shows the familiar high-ρ-negative regime (−1.37pp, ρ=0.65). Giant (1.136B, base_B=57.22 — raw giant is *worse* at correspondence than base, a known DINOv2 quirk): +0.08/−0.34/+0.54/**+1.04pp** at σ=3e-5…1e-3 — best cell inside its own 1.32pp A/B noise floor. **No thicket at the paper's ~1.5B working threshold.** ⚠ Wrinkle: giant's gain AND ρ rise monotonically with σ (ρ 0.05→0.20, p90 +1.07, max +2.01) and the grid stopped mid-climb — the down-shifted σ grid was a design miss: **giant is MORE robust than base** (positive mean at σ=1e-3 where base lost −4.6pp; the collapse cliff moves RIGHT with scale). Extension RESULT (2026-07-08): σ=3e-3 → +0.31pp with ρ=0.73 (reproducible-but-unhelpful, same signature as base's high-ρ regimes); σ=1e-2 → −49.8pp COLLAPSE. The rising trend broke at 3e-3 — **no scaling-law signal; giant's cliff sits at ~1e-2 (vs base's ~3e-3), grid mapped edge to edge; scale confound closed at N=300**. Paper-N campaign in flight: `e5g-1e3-d{1..5}` (5×N=1000, seeds 108–112, same splits) pools to N=5000 at the peak cell σ=1e-3; `randopt_corr_thicket.py` now dumps per-perturbation seeds/pcks_A/pcks_B so the pooled one-expert stat is recomputed honestly (rank on pooled A, report B). Numbers: `results/e5-*/results.json`, W&B `e5-*`.
- **Scale ladder (E5, launched 2026-07-07 overnight, superseded by result line above): `e5-giant-s{1..4}` + `e5-large`.** Exactly the E1 protocol with only `MODEL` changed — DINOv2-large (0.304B ≈ paper's "small gains" zone) and DINOv2-giant (1.136B ≈ paper's ~1.5B working threshold). Scope **all**, σ ∈ {3e-5, 1e-4, 3e-4, 1e-3} (grid shifted one notch down vs base — wider models collapse at smaller per-param noise), N=300/cell, seeds 101–105, A/B=200/200, K=1, BATCH=32 (giant) / 48 (large). Giant fanned out one σ per GPU (~40s/pert measured — much faster than the 130s FLOPs estimate; giant-smoke validated load/perturb/OOM, 1,136,482,305 params in scope). Pre-registered decision rules: flat at 1.14B → scale confound killed, SSL story stands; rises with size → paper's scaling law reproduced in vision (E1–E4 become the small-scale anchor); mixed → DINOv3-7B (largest pure-SSL backbone, HF-gated) is the decisive rung. Results → `results/e5-*/results.json`, W&B runs `e5-*`.
- **Presentation artifact (advisor-ready, E1–E5):** experiment log with per-experiment parameters, pre-registered decision tables, code snippets, paper-comparison table (N/K/σ vs Qwen runs), E3/E4 density heatmaps, interactive E4 map: https://claude.ai/code/artifact/d3b77bb8-4333-4b04-8cec-e4b198e55e5a
- Full per-cell numbers: `results/report.html` (SSL Correspondence section) and each run's `results/corr_thicket_*/results.json`; W&B project `randopt`, runs `thicket-*-night` / `thicket-*-deep`.

### Cluster run example (thicket)
```bash
# 1) ONE dedicated sync job (never let multiple jobs touch git at once):
runai submit pvc-sync -p raja -i noyhassid/randopt-vllm:latest --backoff-limit 0 \
  --existing-pvc claimname=storage,path=/storage --working-dir /storage/noy/RandOpt \
  --command -- bash -c "cd /storage/noy/RandOpt && git fetch origin -q && \
    git reset --hard origin/feature/vision-randopt"
# 2) then compute jobs with SKIP_SYNC=1:
runai submit thicket-last1 -p raja -i noyhassid/randopt-vllm:latest -g 1 --backoff-limit 0 \
  -e SKIP_SYNC=1 -e SCOPES=last1 -e NPOP=1000 -e BATCH=256 -e SEED=42 \
  -e SIGMAS=0.0003,0.001,0.003,0.01,0.03,0.1 -e RUNNAME=thicket-last1 \
  --existing-pvc claimname=storage,path=/storage \
  --command -- bash /storage/noy/RandOpt/scripts/run_corr_thicket.sh
```

## Baselines

PPO / GRPO / ES baselines live under `baselines/` (built on VERL). Separate conda env required:
```bash
conda create -n baseline python==3.12 && conda activate baseline
cd baselines && pip install --no-deps -e .
bash scripts/install_vllm_sglang_mcore.sh   # or sbatch install/ on cluster
```
Run scripts are in `baselines/run_jobs/`. See `baselines/README.md` for full details.

## RunAI Cluster

**Project:** `raja` | **Image:** `noyhassid/randopt-vllm:latest` | **Code on cluster:** `/storage/noy/RandOpt/` (Lustre PVC) | **Results:** `/storage/noy/RandOpt/results/`

### Workflow

```bash
# 1. Push code to GitHub
git push

# 2. Pull onto cluster (CPU job, ~30s)
bash scripts/update_cluster_code.sh

# 3. Submit experiment
bash scripts/submit_runai.sh --dataset countdown --model Qwen/Qwen2.5-3B-Instruct --gpus 4

# 4. Watch logs
runai logs <job-name> -f

# 5. Check status
runai workload list
```

### submit_runai.sh flags

```bash
bash scripts/submit_runai.sh \
  --dataset countdown \       # dataset name
  --model Qwen/Qwen2.5-7B-Instruct \
  --gpus 8 \                  # total GPUs (num_engines = gpus / tp)
  --tp 2 \                    # tensor parallel; use 2+ for 7B+
  --population 5000 \
  --sigma "0.0005,0.001,0.002" \
  --no_wandb                  # disable W&B for this run
```

### Smoke test (W&B verification)

```bash
bash scripts/submit_wandb_test.sh
# Runtime: ~10-15 min on 1x H100, N=20 perturbations, 0.5B model
# Then check: wandb.ai → project "randopt" → run "randopt-wandb-test"
```

### Node / GPU notes

- **H100 (node8)** is fastest but frequently grabbed by another queue → jobs pend. For small jobs (e.g. DINOv2-base), **drop `--node-type`** and take any free GPU; **node7 (RTX-6000 Ada)** is the fastest freely-available one. A5000 nodes (1,3,4,5,6) and A6000 (node2, node9) work fine. **node10 is usually Unschedulable.**
- On Mac CLI: use `-g 1` not `--gpu 1`.
- Use `runai list jobs -p raja` to list/check jobs (the `runai workload list` form is not available on this gateway; `runai describe job <name> -p raja` for status).
- **Always submit with `--backoff-limit 0`** — otherwise a crash auto-restarts in a loop (this caused overnight runaway restarts).
- **PVC code sync:** use `git fetch + git reset --hard origin/<branch>`; plain `git pull` fails on divergent PVC edits. Only ONE job may touch git at a time — a dedicated `pvc-sync` job, then compute jobs with `SKIP_SYNC=1` (concurrent git → `.git/index.lock` crash). **Verify the sync actually succeeded before any `SKIP_SYNC=1` submit** (check `runai logs pvc-sync` shows the expected HEAD + `ls` the new files in the sync command itself) — a sync submitted while the token was expired silently no-ops and dependent jobs fail with "No such file or directory" against stale code (bitten 2026-07-06).
- **Token expiry cadence:** the gateway's RunAI token has expired roughly daily during heavy use; symptoms are `invalid_grant "Session doesn't have required client"` or runai commands returning empty/timing out. Fix is interactive `runai login` **on the gateway with a real stdin** (the `!`-prefix or a piped ssh can't answer the verification-code prompt — it EOFs).
- Run Python with `-u` (unbuffered) or stdout won't stream through `tee`.
- Build Docker for `linux/amd64` (`docker buildx build --platform linux/amd64 --push`) — Mac ARM64 images fail to pull with `no match for platform`.
- The SSH gateway (Geoffry) is intermittently flaky and the RunAI token expires periodically — re-run `runai login` **on the gateway** to refresh.

### One-time cluster setup

```bash
# Store W&B key (run once from inside any container with /storage mounted)
echo "YOUR_WANDB_KEY" > /storage/noy/.wandb_api_key && chmod 600 /storage/noy/.wandb_api_key

# Wire GitHub token for git pull (run once from inside container)
git -C /storage/noy/RandOpt remote set-url origin \
  https://$(cat /storage/noy/.github_token)@github.com/noytau/RandOpt.git
```

### Rebuilding the Docker image

Only needed when dependencies change (not on code changes — code lives on PVC). **Must target `linux/amd64`** (Mac is ARM64; an ARM image fails to pull on the cluster with `no match for platform`):
```bash
# Build+push from the parent of RandOpt/ (buildx does both)
docker buildx build --platform linux/amd64 \
  -f RandOpt/docker/Dockerfile_vllm -t noyhassid/randopt-vllm:latest --push .
```
Alternatively trigger the GitHub Actions workflow `.github/workflows/docker-build.yml` (builds amd64 on a GH runner, needs `DOCKERHUB_TOKEN` secret). The image now bundles `wandb`, `transformers`, `torchvision`, `Pillow`; the vision run scripts still `pip install` them at runtime as a fallback.

## W&B Experiment Tracking

W&B is enabled by default. Logs are written to project `randopt`.

**New CLI args:**
- `--wandb_project randopt` — set to `""` to disable
- `--wandb_name <name>` — custom run name (defaults to `<experiment_dir>_<dataset>`)

**What gets logged per run:**

| Metric | Phase |
|--------|-------|
| `base/train_reward`, `base/test_accuracy` | After base model eval |
| `sampling/batch_mean_reward`, `sampling/batch_max_reward`, `sampling/samples_evaluated` | Each sampling batch |
| `sigma/<σ>/mean_reward` | End of sampling, one entry per sigma |
| `sampling/best_sigma` | End of sampling |
| `ensemble/k<K>/accuracy`, `ensemble/k<K>/gain_over_base` | Each K after ensemble eval |

**Reading results:**

- `base/test_accuracy` — greedy single-model baseline
- `ensemble/k<K>/accuracy` — majority-vote over top-K perturbed models
- `ensemble/k<K>/gain_over_base` — the actual improvement from RandOpt (+X%)
- `sampling/best_sigma` — which noise scale σ produced the best perturbations (most important hyperparameter)
- `sigma/*/mean_reward` — compare these to understand σ sensitivity

**Local results file** (`results/<run>/results.json`):
```json
{
  "base_test_accuracy": 0.10,
  "best_sigma": 0.002,
  "ensemble_results": {
    "10": {"accuracy": 18.0, "correct": 9},
    "4":  {"accuracy": 16.0, "correct": 8}
  },
  "top_k_perturbs": [[seed, sigma], ...]
}
```
The `top_k_perturbs` list is the input to distillation (`distillation/distill_data_gen.py`).

## Distillation

After a RandOpt run, distill the top-K perturbed models into a single model via SFT:
```bash
python distillation/distill_data_gen.py --seeds_file path/to/model_saves/top_k_seeds.json
sbatch distillation/distill_sft.sh
```

## 1D Experiments

`simple_1D_signals_expts/` contains small-scale experiments (no GPU needed) that demonstrate the core Neural Thickets intuition. Entry points: `expt_script_approximation.py` and `expt_script_generalization.py`.
