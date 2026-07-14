# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

RandOpt implements **Neural Thickets** (paper: arxiv 2603.12228): instead of gradient-based fine-tuning, it randomly perturbs a pretrained LLM's weights using Gaussian noise, evaluates each perturbation on a small train set, selects the top-K perturbations by reward, and uses majority voting (ensemble) over these perturbed models to answer test queries. The key insight is that high-quality task-adapted models are densely packed around pretrained weights.

**Vision / SSL extension (branch `feature/vision-randopt`):** The same algorithm applied to DINOv2. **Current experiment series: E1 — image classification on ImageNet-C with DINOv2-giant** (`experiments/E1_imagenet_c.md`, tasks in `TASKS.md`; see the **Vision Experiments** section below). Older vision experiments (CIFAR-10 classification, SPair-71k correspondence/PCK) are discarded and archived in `experiments/old_experiments.md` — do not build on them.

## Working Agreement (read first — set by the user 2026-07-08)

1. **Code approval gate:** every NEW function added to this project must be shown to and approved by the user BEFORE it is committed. Present the function (signature + body + why it's needed), wait for explicit approval, then commit. Small mechanical edits to existing code (renames, param defaults, log lines) don't need a gate, but new functions/scripts always do.
2. **The user is learning:** they have ML-engineering experience but are new to research practice. Explain research framings (baseline vs control, probe conventions, why a design choice exists) in plain language, define terms on first use, and be patient. The goal is for the user to learn, then drive the research themselves.
3. **Always return to the base experiment.** Every result, plan, or new idea must be framed against the project's anchor comparison, on the same task/metric/splits:
   | Rung | Method | Trains what |
   |---|---|---|
   | Baseline 1 | **Fine-tuning** the pretrained model (low LR) | backbone weights (gradient) |
   | Baseline 2 | **Linear probe** (train a linear head, frozen backbone) | head only (gradient) |
   | Contender | **RandOpt** (random perturbation + selection) | nothing (no gradients) |
   The research question is always: *how close does training-free RandOpt come to the two standard SSL adaptation baselines?*
4. **Method of work — every experiment (set by the user 2026-07-13):**
   - **Paper reference (keep handy):** Neural Thickets — arXiv 2603.12228, https://arxiv.org/abs/2603.12228
   - ① Write code → explicit user approval (gate in rule 1) → commit.
   - ② Run a **short POC experiment**: ONE simple fixed configuration, no val-based hyperparameter selection, logged to W&B.
   - ③ **Evaluate every experiment in two steps:** first verify it *worked* (job Succeeded, logs clean, expected W&B keys present, numbers sane), only then evaluate the *results*.
   - ④ If clear and explicitly approved → the **larger experiment** (hyperparameter sweeps, selected on the val split).
   - ⑤ **No pre-commitment of hyperparameters/training parameters — for ALL experiments.** Specs and plans never fix sweep ranges in advance; review ranges on the go during testing (and against the paper's choices where relevant) and propose them to the user before each larger tier.

### Pre-experiment checklist (set by the user 2026-07-11)

Every new RandOpt experiment spec must answer these four checks **in writing, before any job is submitted**:

1. **Scale check — is the model large enough?** State the total param count *and* the in-scope (perturbed) param count, and place them on the paper's scaling ladder (§6): RandOpt **fails** at 0.1B (GPT-2), gives **small gains** at 0.5B (Qwen), and **works from ~1.5B**. If the model sits below ~1B, say so explicitly and treat scale as a live confound.
2. **Base & improvement check — what exactly are we improving?** Define the **base**: the *unperturbed* model's score on the task test set — the number every gain is measured against (`base/test_accuracy` in W&B; the paper's analogue is the pretrained model's task performance, and its "pretrain NLL" is the pretrained LM's negative log-likelihood, used as a measure of init quality — for non-generative models substitute the base task metric). Then instantiate the full anchor table (Baseline 1 = fine-tune, Baseline 2 = linear probe, Contender = RandOpt) on the **same task, metric, splits, and adaptation budget**. Check the base has **headroom** (a base at ceiling cannot show a gain).
3. **Perturbation-scope check — which parameters move?** Name the scope (`all` / `classifier` / `last_n_blocks`), the scalar count it covers, and compare N, K, and the σ grid to the paper's runs (paper: all transformer weights, N=5000, σ≈5e-4–2e-3 on Qwen — verify against the paper before citing). Note where the collapse cliff is expected (it moves with scale — map it with a small σ scan before committing a grid).
4. **Ops check — logging and cluster flow.** W&B project `randopt` with a per-experiment run-name prefix; cluster flow per this file (push → `pvc-sync` job → compute jobs with `SKIP_SYNC=1`, always `--backoff-limit 0`); verify free GPUs, estimate GPU-hours and cost, and get **explicit user approval** before every submission.

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

## Vision Experiments — current: E1 (ImageNet-C classification)

Branch `feature/vision-randopt`. RandOpt beyond LLMs: `DatasetHandler` + W&B plumbing reused,
vLLM swapped for a PyTorch+Ray `VisionEngine` (`vision/engine.py`, launched via
`vision/launch_vision_engines`). **Spec: `experiments/E1_imagenet_c.md` · tasks: `TASKS.md` ·
discarded older experiments (CIFAR/PCK): `experiments/old_experiments.md`.**

### Entry points (E1: four rungs, same splits & metric — test top-1)
| Script | Rung |
|--------|------|
| `scripts/knn_imagenet_c.py` | kNN on backbone embeddings (base/headroom readout; POC fixed k, larger tier sweeps k on val) |
| `scripts/probe_imagenet_c.py` | linear classifier on frozen backbone (planned) |
| `scripts/finetune_imagenet_c.py` | finetune — some layers first, entire model in larger tier (planned) |
| `scripts/randopt_imagenet_c.py` | RandOpt contender: perturb backbone, score by kNN on train, top-K majority-vote ensemble on test |
| `scripts/run_e1_baselines.sh` | one cluster entry point for all tasks (`TASK=knn\|probe\|ft\|randopt`) |
| `scripts/count_params.py` | CPU param inventory (per-block counts, validates `last_n_blocks` scope filter) |

### Key pieces
- **`data_handlers/imagenet_c.py`** — ImageNet-C (50/class × 1000 classes per corruption/severity, 224×224). Zenodo selective download; splits `train/val/test = 25/10/15` per class (seeded, stratified, disjoint). Val is used only by larger tiers.
- **`vision/features.py`** — frozen-backbone feature cache (`cls` + `pooler` per image, one forward, fp16 on disk). kNN/probe consume it; FT/RandOpt must not use it inside loops that move weights.
- **`vision/engine.py` (`VisionEngine`)** — `perturb_target`: `"all"` · `"classifier"` · `"last_n_blocks"` (giant = 40 blocks named `encoder.layer.{i}`); `set_perturb_scope(target, n)` on a live actor; `perturb_weights/restore_weights(seed, sigma)` per-seed `torch.Generator` (bit-exact restore, same scheme as the LLM path); `eval_global` (weighted-vote kNN + mAP) and `knn_predict` (per-query labels for ensemble voting). Giant's SwiGLU FFN is supported by `Dinov2Model`; use `BATCH=32`.

### Cluster run example (E1)
```bash
# 1) ONE dedicated sync job (never let multiple jobs touch git at once):
runai submit pvc-sync -p raja -i noyhassid/randopt-vllm:latest --backoff-limit 0 \
  --existing-pvc claimname=storage,path=/storage --working-dir /storage/noy/RandOpt \
  --command -- bash -c "cd /storage/noy/RandOpt && git fetch origin -q && \
    git reset --hard origin/feature/vision-randopt"
# 2) then compute jobs with SKIP_SYNC=1:
runai submit e1-knn-poc -p raja -i noyhassid/randopt-vllm:latest -g 1 --backoff-limit 0 \
  -e SKIP_SYNC=1 -e TASK=knn -e CORRUPTION=gaussian_noise -e SEVERITY=3 \
  --existing-pvc claimname=storage,path=/storage \
  --command -- bash /storage/noy/RandOpt/scripts/run_e1_baselines.sh
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

- **PVC access:** `/storage` (code, data, results on the cluster) is reachable **only from inside RunAI jobs** — not from the gateway (Geoffry) and not from the Mac. When asked to query job state or results, **try `runai` first** (`runai list jobs -p raja`, `runai logs <job>` — via `ssh Geoffry` since the Mac's token is separate). If the token is expired, the fallback readout is the **W&B API** (`~/.netrc` credentials exist on both Mac and Geoffry); `results/*.json` on the PVC stays unreachable until re-login.
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
