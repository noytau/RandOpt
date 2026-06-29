# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

RandOpt implements **Neural Thickets** (paper: arxiv 2603.12228): instead of gradient-based fine-tuning, it randomly perturbs a pretrained LLM's weights using Gaussian noise, evaluates each perturbation on a small train set, selects the top-K perturbations by reward, and uses majority voting (ensemble) over these perturbed models to answer test queries. The key insight is that high-quality task-adapted models are densely packed around pretrained weights.

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

- Always use `--node-type NVIDIA-H100-80GB-HBM3` to land on node8 (H100). A6000 nodes (9, 2, 10) have a 42 GiB zombie GPU memory issue that causes OOM.
- On Mac CLI: use `-g 1` not `--gpu 1`.
- `runai workload list` (not the deprecated `runai list jobs`).

### One-time cluster setup

```bash
# Store W&B key (run once from inside any container with /storage mounted)
echo "YOUR_WANDB_KEY" > /storage/noy/.wandb_api_key && chmod 600 /storage/noy/.wandb_api_key

# Wire GitHub token for git pull (run once from inside container)
git -C /storage/noy/RandOpt remote set-url origin \
  https://$(cat /storage/noy/.github_token)@github.com/noytau/RandOpt.git
```

### Rebuilding the Docker image

Only needed when dependencies change (not on code changes — code lives on PVC):
```bash
# Build from the parent of RandOpt/
docker build -f RandOpt/docker/Dockerfile_vllm -t noyhassid/randopt-vllm:latest .
docker push noyhassid/randopt-vllm:latest
```
`wandb` is currently installed at job runtime (`pip install wandb --quiet` in submit scripts) until the image is next rebuilt.

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
