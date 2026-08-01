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
| 6 | d3 | 5000 | 1k/5k | — | DIED (host-RAM OOM @1052/5000) — resubmit decision open |
| 7 | d3 | 2000 | 5k/1k | — | held (user) |
| 8 | d3 | 5000 | 5k/1k | — | held (user) |

Earlier series: IC-scored N-sweep (σ={1,2,3}e-3) uniformly −0.3 to −0.5;
local N=200 scope study ±0.1. Headline: **replicated null within ±0.1pp across
model scale (1.1B & 6.7B), N (2k/5k), and scoring/eval splits; σ=2e-4 wins
everywhere** (matches FT displacement σ-equivalent ~2e-4).

## Open experiment tasks

- [ ] Decide on #6 resubmit (1 engine or bigger RAM reservation; ~5 d; low
      expected info given #5's null)
- [ ] Decide on #7–8 (multi-day d3 runs; same consideration)
- [ ] Protocol-matched FT baseline (offered, not approved): fine-tune backbone+
      released head on the SAME 1k/5k clean scoring draw, eval on the same IC
      draw — the gradient-based counterpart to the RandOpt null (~1 h, 1 GPU)
- [ ] Design tier (brainstorm with Nimrod, never auto-implement): kNN-centered
      scoring arm; unadapted-center setup (head not converged on task);
      two-stage scoring

## Infra / chores

- [ ] Docker image rebuild: bake in `wandb torchmetrics termcolor omegaconf
      ftfy submitit scikit-learn regex` (now pip-installed per job)
- [ ] Report expired cluster-api TLS cert (`hpc-vm-runai.eng.tau.ac.il`) to
      admins — blocks `runai logs`/`exec`; workaround: W&B run file
      `output.log`
- [ ] Adapt or delete stale `scripts/dinov2_hub_inference.py` (uncommitted)

## Cluster operating rules (hard-won; details in Claude memory)

- Pin big-model jobs: `--node-pools raja` (48G A6000s). hpc-node3 = 24 GB.
- ALWAYS `--cpu-memory-request` (64G ViT-g jobs / 128G d3-2-engine jobs);
  unreserved jobs get OOM-killed by neighbors.
- `-g N` cannot combine with `--gpu-request-type memory`; no exclude-node.
- `--backoff-limit 0` always; runai token expires daily (`runai login`).
- W&B name convention: `randopt-<d2|d3>-n<N>-k<K>-tr-<set><size>-te-<set><size>`.
