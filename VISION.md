# RandOpt for Vision (DINOv2 + CIFAR-10)

Branch: `feature/vision-randopt`

Extends the Neural Thickets algorithm from LLMs to SSL vision models. Tests whether the same "dense packing of task-adapted models around pretrained weights" hypothesis holds for DINOv2 on image classification.

## Architecture

```
randopt_vision.py         ← Entry point (mirrors randopt.py, uses VisionEngine instead of vLLM)
vision/
  engine.py               ← VisionEngine Ray actor: DINOv2 backbone + linear head + perturb/restore
  __init__.py             ← launch_vision_engines() / cleanup_vision_engines()
  train_linear_probe.py   ← Optional: train linear probe for warm init (~5 min)
data_handlers/
  cifar10.py              ← CIFAR10Handler (torchvision, auto-downloads)
utils/reward_score/
  vision.py               ← compute_score(pred_id, gt_id) → 0.0 or 1.0
```

No changes to the LLM path (`randopt.py`, `core/`, `utils/worker_extn.py`).

## How It Works

Same 4-phase algorithm as the LLM path:

1. **Base eval** — run DINOv2 + random linear head on train/test, record accuracy
2. **Sampling** — for each of `population_size` (seed, σ) pairs: perturb all weights → forward pass on train images → restore weights → record top-1 accuracy
3. **Selection** — sort by train accuracy, keep top-K
4. **Ensemble** — run each top-K model on test set, majority-vote predicted class IDs per image

### Perturbation

`VisionEngine` uses the same deterministic seed scheme as `WorkerExtension` in the LLM path: for each parameter, a fresh `torch.Generator` is seeded with `seed`, producing reproducible Gaussian noise. `restore_weights` regenerates the exact same noise and subtracts it.

Full backbone + linear head are perturbed (equivalent to the LLM setting).

## Running

### Cluster smoke test (1 GPU, ~10 min)

```bash
runai training submit randopt-vision-smoke \
  --project raja \
  --image noyhassid/randopt-vllm:latest \
  -g 1 \
  --node-type NVIDIA-H100-80GB-HBM3 \
  --existing-pvc claimname=storage,path=/storage \
  --working-dir /storage/noy/RandOpt \
  --command -- bash -c "
    pip install wandb transformers torchvision --quiet
    export HF_HOME=/storage/noy/.cache/huggingface
    [ -f /storage/noy/.wandb_api_key ] && export WANDB_API_KEY=\$(cat /storage/noy/.wandb_api_key)
    python randopt_vision.py \
      --dataset cifar10 \
      --model_name facebook/dinov2-small \
      --num_engines 1 --cuda_devices 0 \
      --population_size 10 --train_samples 100 --test_samples 200 \
      --sigma_values '0.001,0.01,0.1' --top_k_ratios '0.2,0.5' \
      --wandb_project randopt --wandb_run_name vision-smoke-test
  "
```

### Full run (4 GPUs)

```bash
runai training submit randopt-vision-cifar10 \
  --project raja \
  --image noyhassid/randopt-vllm:latest \
  -g 4 \
  --node-type NVIDIA-H100-80GB-HBM3 \
  --existing-pvc claimname=storage,path=/storage \
  --working-dir /storage/noy/RandOpt \
  --command -- bash -c "
    pip install wandb transformers torchvision --quiet
    export HF_HOME=/storage/noy/.cache/huggingface
    [ -f /storage/noy/.wandb_api_key ] && export WANDB_API_KEY=\$(cat /storage/noy/.wandb_api_key)
    python randopt_vision.py \
      --dataset cifar10 \
      --model_name facebook/dinov2-base \
      --num_engines 4 --cuda_devices 0,1,2,3 \
      --population_size 500 --train_samples 1000 \
      --sigma_values '0.0001,0.001,0.01,0.1' --top_k_ratios '0.01,0.05,0.1' \
      --wandb_project randopt --wandb_run_name vision-cifar10-n500
  "
```

### Local (no GPU, CPU only — slow but works for debugging)

```bash
python randopt_vision.py \
  --dataset cifar10 \
  --model_name facebook/dinov2-small \
  --num_engines 1 --cuda_devices "" \
  --population_size 3 --train_samples 20 --test_samples 50 \
  --sigma_values "0.01" --top_k_ratios "0.5"
```

## Key CLI Arguments

| Arg | Default | Notes |
|-----|---------|-------|
| `--dataset` | `cifar10` | Currently only `cifar10` |
| `--model_name` | `facebook/dinov2-base` | Any HF DINOv2 variant: `dinov2-small`, `dinov2-base`, `dinov2-large` |
| `--num_classes` | `10` | 10 for CIFAR-10 |
| `--linear_init_path` | `None` | Path to pretrained linear head `.pt` (see below) |
| `--inference_batch_size` | `64` | Images per GPU forward pass — reduce if OOM |
| `--population_size` | `50` | Total perturbations to evaluate |
| `--sigma_values` | `0.0001,...,0.1` | Noise scales — vision needs wider range than LLMs |
| `--num_engines` | `1` | GPUs to use (1 engine per GPU) |
| `--wandb_project` | `None` | Set to `randopt` to enable W&B logging |

## Warm Init: Linear Probe

By default the linear head is randomly initialized (~10% accuracy on CIFAR-10). To start from a strong base:

```bash
# Train linear probe once (~5 min on 1 GPU)
python vision/train_linear_probe.py \
  --model_name facebook/dinov2-base \
  --data_dir data/cifar10 \
  --output_path data/cifar10/linear_probe_dinov2base.pt \
  --epochs 10

# Then pass to randopt_vision.py:
python randopt_vision.py ... --linear_init_path data/cifar10/linear_probe_dinov2base.pt
```

Expected linear probe accuracy: ~82% (DINOv2-base on CIFAR-10).

## W&B Metrics

Same metric names as the LLM path:

| Metric | Phase |
|--------|-------|
| `base/train_reward`, `base/test_accuracy` | After base model eval |
| `sampling/batch_mean_reward`, `sampling/batch_max_reward` | Each sampling batch |
| `sigma/<σ>/mean_reward` | End of sampling |
| `sampling/best_sigma` | End of sampling |
| `ensemble/k<K>/accuracy`, `ensemble/k<K>/gain_over_base` | Each K after ensemble eval |

## Results Format

Same `results.json` schema as the LLM path — compatible with the same analysis scripts.

## Adding a New Vision Dataset

1. Add data handler in `data_handlers/your_dataset.py` inheriting `DatasetHandler`:
   - `load_data()` returns dicts with `image_tensor` (float32, C×H×W, [0,1]), `ground_truth`, `class_id`
   - Override `postprocess_outputs(predictions: List[int], task_datas)` to compute accuracy from class IDs
2. Register in `data_handlers/__init__.py`
3. Pass `--dataset your_dataset --num_classes N` to `randopt_vision.py`
