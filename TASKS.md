# RandOpt Vision (SSL) — Plan & Tasks

Living plan for the DINOv2/DINOv3 RandOpt series. Any agent/session working on
this project reads this first and updates it when state changes (check the box,
add the result inline). Ground truth for results: W&B project `randopt`;
per-run `results/<name>/results.json` on the PVC.

## Current state (updated 2026-08-01)

Protocol: score N Gaussian perturbations (all layers + head, σ={2,5,10}e-4,
seed 42) on class-balanced clean ImageNet-train draws; select top-K by score;
majority-vote on class-balanced ImageNet-C (gaussian_noise/3) test draws.
Bases: d2 = 83.34/82.80 (5k/1k eval), d3 = 84.86.

| # | Model | N | score/eval | K=50 Δ | status |
|---|-------|---|-----------|--------|--------|
| 1 | d2 | 2000 | 1k/5k | +0.08 | done |
| 2 | d2 | 5000 | 1k/5k | +0.06 | done |
| 3 | d2 | 2000 | 5k/1k | −0.10 | done |
| 4 | d2 | 5000 | 5k/1k | +0.10 | done |
| 5 | d3 | 2000 | 1k/5k | −0.04 | done |

DINOv3 experiments 6–8 DROPPED (2026-08-01): #5's clean null makes their info
value low, and d3 jobs kept getting killed on the shared cluster (root cause
found — see "d3 post-mortem" below — fixable, but not worth the GPU-days now).

Earlier series: IC-scored N-sweep (σ={1,2,3}e-3) uniformly −0.3 to −0.5;
local N=200 scope study ±0.1. Headline: **replicated null within ±0.1pp across
model scale (1.1B & 6.7B), N (2k/5k), and scoring/eval splits; σ=2e-4 wins
everywhere** (matches FT displacement σ-equivalent ~2e-4).

## d3 post-mortem (2026-08-01)

- d3-n2000 was NOT stuck: it completed fully (results recovered from the W&B
  console log + results.json); only the W&B heartbeat died at teardown.
- d3-n5000 was killed by **Ray's node-level memory monitor**: node RAM hit 95%
  from NEIGHBOR pods while our engines used 52 GB of our reserved 128 GB. Ray's
  monitor ignores cgroup boundaries and killed our worker.
- **Fix for any future Ray job**: add `export RAY_memory_monitor_refresh_ms=0`
  to the job command (kernel cgroup OOM killer remains the real guardrail).

## Open tasks

- [ ] **FT matched baseline** (scripts/ft_matched_baseline.py): fine-tune the
      released DINOv2 backbone+head on the SAME class-balanced clean draws as
      RandOpt scoring, eval on clean holdout + IC draws. Two configs:
      tr1k/eval5k and tr5k/eval1k. Run on the new GPU servers once connected
      (else cluster A6000). ~1–2 h each.
- [ ] **New GPU servers bring-up** (132.66.150.56 = `gpu56`,
      132.66.150.55 = `gpu55`; user noy, Geoffry password):
      - [ ] key auth: USER runs `ssh-copy-id noy@132.66.150.56` (+ .55)
      - [ ] inspect: GPUs, CUDA driver, disk, home layout
      - [ ] code: clone noytau/RandOpt; python env (venv like Geoffry)
      - [ ] data: pull imagenet train-subset + val + IC gaussian_noise/3 from
            Geoffry (server→server scp after keying), regen manifests locally
      - [ ] smoke: engine load + 1-perturbation round trip
- [ ] Design tier (brainstorm with Nimrod, never auto-implement): kNN-centered
      scoring arm; unadapted-center setup; two-stage scoring
- [ ] Docker image rebuild (only if cluster still used): bake in `wandb
      torchmetrics termcolor omegaconf ftfy submitit scikit-learn regex`
- [ ] Report expired cluster-api TLS cert (`hpc-vm-runai.eng.tau.ac.il`) to
      admins — blocks `runai logs`/`exec`; workaround: W&B run `output.log`
- [ ] Adapt or delete stale `scripts/dinov2_hub_inference.py` (uncommitted)

## Job venue policy (updated 2026-08-01)

**Prefer the standalone GPU servers `gpu56`/`gpu55` over RunAI** for new
jobs once they're set up (large GPUs, no scheduler roulette, no daily token,
no noisy-neighbor OOM kills). Fall back to RunAI for multi-node scale only.

## Cluster (RunAI) operating rules — for fallback use

- Pin big-model jobs: `--node-pools raja` (48G A6000s). hpc-node3 = 24 GB.
- ALWAYS `--cpu-memory-request` (64G ViT-g / 128G d3-2-engine jobs).
- Add `export RAY_memory_monitor_refresh_ms=0` to every Ray job command.
- `-g N` cannot combine with `--gpu-request-type memory`; no exclude-node.
- `--backoff-limit 0` always; runai token expires daily (`runai login`).
- W&B name convention: `randopt-<d2|d3>-n<N>-k<K>-tr-<set><size>-te-<set><size>`.
