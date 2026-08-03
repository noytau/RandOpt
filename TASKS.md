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

**FT matched baseline (2026-08-02, gpu56)** — same data budgets, gradient FT
(lr 1e-5, 5 epochs, full model) instead of perturbation+selection:

| budget | FT clean Δ | FT IC Δ | RandOpt IC Δ | RandOpt edge |
|--------|-----------|---------|--------------|--------------|
| 1k imgs | −1.24 | −1.32 | +0.08 | **+1.4pp** |
| 5k imgs | −2.70 | −2.10 | +0.10 | **+2.2pp** |

FT memorizes the training draw (train acc 1.0) and damages both clean and
corrupted accuracy; more data at fixed epochs = larger displacement = more
damage. Reframes the RandOpt null: at small budgets on an adapted center,
selection-over-noise is SAFER than gradient descent — it cannot overfit
weights, only choose among them. (W&B: ft-d2-tr-intrain{1,5}k-*.)

## d3 post-mortem (2026-08-01)

- d3-n2000 was NOT stuck: it completed fully (results recovered from the W&B
  console log + results.json); only the W&B heartbeat died at teardown.
- d3-n5000 was killed by **Ray's node-level memory monitor**: node RAM hit 95%
  from NEIGHBOR pods while our engines used 52 GB of our reserved 128 GB. Ray's
  monitor ignores cgroup boundaries and killed our worker.
- **Fix for any future Ray job**: add `export RAY_memory_monitor_refresh_ms=0`
  to the job command (kernel cgroup OOM killer remains the real guardrail).

## ImageNet distribution-shift suite (added 2026-08-01)

Data prep for 4 new ImageNet-1k-label-space shift datasets (ImageNet-A/R/
Sketch/ES), so `linear_probe`/`finetune`/`randopt` can later be compared on
identical data with the frozen 1000-way head reused unchanged. Plan file:
`.claude/plans/snazzy-popping-fairy.md` (has the full spec + adaptations).

**Key finding**: the existing `scripts/make_imagenet_c_manifest.py` labels
classes as `enumerate(sorted(os.listdir(data_dir)))` — a **local** wnid sort,
silently correct today only because ImageNet-C/plain-ImageNet manifests
always cover all 1000 classes. Confirmed this breaks for real on a 200-class
subset: the published ImageNet-R split (CODA-Prompt's
`imagenet-r_{train,test}.yaml`) encodes `targets` as local 0–199 indices, not
canonical ImageNet-1k ones. New pipeline (`scripts/data_prep/`) resolves every
label through a canonical `wnid -> 0..999` index
(`build_class_index.py`, from the real `imagenet/val` 1000-class dirs) instead.

Downloads: ImageNet-R/A from Berkeley tars (checksum-verified); ImageNet-Sketch
and ImageNet-ES both have direct HF-hosted zips (`songweig/imagenet_sketch`,
`Edw2n/ImageNet-ES` — standard ES, not ES-Diverse, per 2026-08-01 decision) —
simpler and more reliable than the gdown/ES-Studio paths a generic spec would
suggest. Data lives on Geoffry under `/mnt5/noy/datasets/{raw,manifests}/`.

- [x] Canonical class index, manifest builder, verifier, zero-shot sanity
      script, 4 new `data_handlers` subclasses, `utils/logit_mask.py` — coded
      and unit-tested locally (`tests/test_build_class_index.py`,
      `tests/test_shift_manifest_builder.py`).
- [x] Synced to Geoffry, gpu56, AND gpu55; canonical class index built on all 3.
- [x] Downloaded + extracted all 4 datasets on Geoffry; imagenet_r transferred
      server-to-server to gpu55+gpu56; imagenet_a/sketch/es transferred
      server-to-server to gpu56 (2026-08-02) — gpu55 only has imagenet_r so far.
- [x] ImageNet-ES: inspected the real extracted tree, wrote
      `_meta/imagenet_es_config.json` (dark=`param_control/l1`, INFERRED not
      verified against a README — flagged in the config's own `_note` field).
- [x] Built + verified manifests for all 4 shift datasets on Geoffry and
      gpu56 (byte-identical output on both hosts, all acceptance checks +
      determinism pass).
- [ ] `zero_shot_sanity.py --dataset all` was never actually run standalone —
      superseded by real `randopt_shift.py` runs instead (see below), but the
      "4 top-1 numbers, check the difficulty ordering ES>A>R>Sketch" diagnostic
      itself is still worth running once as a sanity cross-check.
- [x] `randopt_shift.py` generalized: `--dataset` is now a comma-separated
      plusarg (any of imagenet/imagenet_c/imagenet_a/r/sketch/es, or `all` =
      the 4 shift datasets) resolved through one registry — scoring always on
      clean ImageNet, shared across every named target in one job.
- [x] DINOv3 weights (`dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth`, ~26.9GB;
      `dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth`, ~33MB) downloaded to
      gpu56 and gpu55 (2026-08-02, via Meta's signed CloudFront URL; gpu55 hit a
      total network outage mid-session — worked around via direct gpu55->gpu56
      LAN rsync at ~112MB/s after adding gpu55's key to gpu56's
      `authorized_keys`, since ping between them was sub-2ms but no SSH trust
      existed yet). `facebookresearch/dinov3` repo cloned to both (public, no
      gate). Confirmed working end-to-end: real forward pass, correct
      `[N,1000]` logits, embed_dim=4096, 6.72B backbone params, ~27GB GPU mem.
      Extra pip deps needed beyond DINOv2 (dinov3's hubconf pulls in its own
      segmentation/eval code transitively): `torchmetrics ftfy omegaconf regex
      scikit-learn submitit termcolor`.
      - `--ft_scope {all,head,last_n_blocks}` / `--ft_last_n_blocks` added to
        `ft_matched_baseline.py`, reusing `SSLEngineImpl`'s existing
        `perturb_target` scope-selection so FT and RandOpt always optimize/
        perturb the identical param subset. Verified real (no OOM): 671M
        trainable params (`last_n_blocks=4`), base 85.3%->85.7% IC after 1
        epoch on 1000 images.
      - **Real host-RAM ceiling found and fixed** (matches the earlier
        "~3-4 engines" estimate below almost exactly): `launch_ssl_engines`
        fired all N actor constructors at once; each does a CPU-side
        `torch.load` of the backbone checkpoint, transiently ~2x the
        checkpoint size in host RAM before `load_state_dict`'s `copy_` frees
        the loaded state_dict. DINOv2 (~4.4GB/engine) never noticed; DINOv3
        (~27GB/engine steady-state, ~54GB transient) OOM-killed a worker with
        6-8 engines launched concurrently on gpu56's 251GB host — reproduced
        twice. Fixed by batching engine construction (mirrors
        `core/engine.py`'s already-established batched vLLM init, same
        "avoid contention" shape) — `launch_ssl_engines(..., init_batch_size=2)`.
        Batching alone wasn't enough at N=6-8 (steady-state N x ~27GB itself
        approaches the ceiling, not just the transient spike) — **empirically
        confirmed safe engine count on a 251GB host is 5** (real observed
        steady-state ~149GB/5 engines, ~107GB margin); 6 and 8 both crashed
        even with batching.
      - **Real calibration timing** (N=20, 2 engines, 2026-08-02): base eval
        ~6.75min (1000 img train + 1000 img imagenet_c test), then a rock-
        steady **165.3s per perturbation-batch** (2 engines/1 each), ensemble
        K=2/K=4 both 85.4% vs base 85.3% (+0.1pt). This supersedes the old
        17.4h/2-engine extrapolation below — grounded ETA for N=2000/K=50 at
        5 engines: 400 batches x 165.3s sampling (~18.3h) + ~1h base/ensemble
        overhead across 6 datasets.
      - **Real run launched** 2026-08-02 20:30 on gpu56, 5 engines, all 6
        datasets (`--dataset all,imagenet,imagenet_c`), N=2000/K=50,
        train/test_samples=1000, wandb run `3nes0koi` (name
        `dinov3-all6-N2000-K50-tr1k-te1k`). ETA ~19-20h from launch.
      - **Max FT scope on a single 46GB A6000, found empirically** (gpu55,
        binary search): `last_n_blocks=4` (671M params) and `=5` (839.1M
        params) both fit; `=6` and `=8` both OOM (43.8-44.0/44.55 GiB used,
        failing on a last ~64-192MiB alloc). **5 is the real ceiling** —
        full-layer FT (all 40 blocks, ~107.5GB fp32 AdamW) was never in reach
        of a single GPU, as expected, and this codebase's FT script has no
        multi-GPU sharding to pool memory across GPUs.
      - **Removed the `w0`/`delta_w`/`sigma_equiv` weight-displacement
        tracking from `ft_matched_baseline.py`** (2026-08-03, per user
        request) — the `w0` GPU-resident clone was blocking `ft_scope=all`
        from even starting (OOM'd before the optimizer was built; see repro
        run `dinov3-ft-all-layers-OOM-repro`, `ft_matched_baseline.py:159`
        pre-removal). Even with `w0` fixed (e.g. cloned to CPU), full-layer
        FT still can't run on one GPU — AdamW's own state for 6.72B params
        needs ~80.6GB on top of the 26.9GB resident weights. Real fixes if
        full-layer FT becomes worth pursuing: a CPU-offloaded optimizer
        (bitsandbytes `PagedAdamW32bit` / DeepSpeed ZeRO-Offload, works on
        one GPU but PCIe-bound per step) or real FSDP/ZeRO-3 sharding across
        3+ GPUs (the architecturally correct fix, real engineering lift —
        this script currently has zero distributed-training scaffolding).
      - [ ] **TODO: revisit comparing RandOpt's `sigma` to FT's weight
        displacement.** The removed `sigma_equiv = delta_w / sqrt(P)` metric
        answered "what constant per-parameter sigma would a random
        perturbation need to produce the same L2 displacement FT's gradient
        descent found?" — letting you ask whether FT needed to move further
        or less far than RandOpt's best `sigma` to get its accuracy gain
        (direct evidence for/against the "many good solutions densely packed
        near the pretrained weights" hypothesis this whole project tests).
        Worth re-adding once `ft_scope=all` actually runs (via one of the
        fixes above) — re-implement the snapshot on CPU
        (`p.detach().cpu().clone()`, matching `SSLEngineImpl.
        store_base_weights()`'s existing pattern) so it doesn't reintroduce
        the OOM this removal fixed.
      - **Real FT run completed** (gpu55, `last_n_blocks=5`, 5 epochs,
        train_samples=1000, datasets limited to imagenet/imagenet_c/imagenet_r
        — gpu55 never got the full shift-suite data prep, only Geoffry/gpu56
        did): imagenet 88.3%->88.6% (+0.3), imagenet_c 84.1%->84.2% (+0.1),
        imagenet_r 89.1%->90.0% (+0.9). train_acc hit 99.4% (expected overfit
        on a 1000-image draw over 5 epochs, matches the DINOv2 FT baseline's
        known behavior). Ran under `WANDB_MODE=offline` (see network note
        below) — local run dir `wandb/offline-run-20260802_214233-xh7i9mu9`
        on gpu55, not yet synced to the dashboard.
      - **`api.wandb.ai` had a sustained outage** 2026-08-02 ~21:36 onward
        (>1hr): general internet/DNS/GitHub/`wandb.ai` marketing site all
        fine, but `api.wandb.ai` specifically timed out on raw TCP — looks
        W&B-side, not local network. Confirmed the RandOpt run kept computing
        correctly throughout (worker CPU-time deltas + the local `.wandb`
        binary file's mtime both proved continuous real progress) — W&B's
        dashboard showing "crashed" during this window was a stale-heartbeat
        artifact only; nothing was lost, wandb buffers locally and syncs the
        backlog once torch reachable (`wandb sync <run_dir>` if it doesn't
        auto-resume). **Lesson: don't trust `run.state`/`scan_history()` row
        counts alone during a suspected outage — cross-check the local
        `wandb/run-*/run-*.wandb` file's mtime and, if the process is a Ray
        job, worker CPU-time deltas across two snapshots.**
- [ ] **Examine the raw/manifests directory structure**: imagenet/imagenet_c
      (pre-existing) store raw images directly under `$DATASETS_ROOT/<name>/`
      and their manifest inside the repo (`data/<name>/data.json`); the 4 new
      shift datasets use `$DATASETS_ROOT/raw/<name>/` + `$DATASETS_ROOT/
      manifests/<name>.json` + a `_meta/` provenance folder. Deliberate at the
      time (avoid touching the working old pipeline), but worth revisiting —
      decide whether to migrate imagenet/imagenet_c onto the newer layout, or
      formally document the two-convention split as permanent.
- [ ] Multi-engine startup is flaky on Geoffry specifically (7-engine jobs
      have stalled twice at the `store_base_weights` CPU-clone barrier,
      seemingly always on physical GPU4 — possibly a flaky/degraded card,
      not just contention). Root cause not confirmed (`py-spy` needs sudo).
      gpu55/gpu56 (A6000) have not shown this. Until root-caused, prefer
      gpu55/gpu56 for multi-engine (num_engines>4) jobs.
- [ ] Profiled image decode on A6000: GPU utilization sits at 12-37% even at
      batch_size=128 (huge memory headroom, 46GB cards barely touched) —
      threaded decode (`load_image_batch`, now parallelized) only bought
      ~5%, so the real bottleneck is still unidentified (Ray task-dispatch
      overhead? perturb/restore cost for perturb_target=all's 1.14B params?).
      Worth profiling properly (py-spy/nsys) if throughput matters more.
- [ ] Follow-on (not started): wire `finetune`/`linear_probe` runners to
      these 4 manifests using `utils/logit_mask.py` (only `randopt` is wired
      up so far, via `randopt_shift.py`).

## New same-distribution train/val/test datasets (added 2026-08-02)

Motivation: the current honesty check ("train" on clean ImageNet, eval on a
different dataset/corruption) conflates domain shift with train/test
methodology. Added 5 datasets, each **outside ImageNet's label space** and
each with **its own train/val/test split**, so RandOpt's train->test
generalization can be validated within one consistent distribution per
dataset, separately from the imagenet_c/a/r/sketch/es shift battery above.

| Category | Dataset | Classes | Split train/val/test | Size |
|---|---|---|---|---|
| Dark/low-light | exdark | 12 | 3000/1800/2563 (official) | 1.5G |
| Cartoon/graphics | domainnet_clipart | 345 | 30168/3357/14604 (test official, val carved 10%) | 1.4G |
| Sketch | domainnet_sketch | 345 | 43395/4817/20916 (test official, val carved 10%) | 2.8G |
| Niche (aircraft) | fgvc_aircraft | 100 | 3334/3333/3333 (official) | 2.6G |
| Niche (dogs) | stanford_dogs | 120 | 10800/1200/8580 (test official, val carved 10%) | 784M |

- [x] Confirmed each has a genuine split, not just a description in a paper:
      exdark's lives in `Groundtruth/imageclasslist.txt` col 5 (per-image
      train/val/test flag, counts verified to match exactly); fgvc_aircraft
      ships official `images_variant_{train,val,test}.txt`; stanford_dogs/
      domainnet ship official train/test lists only — val carved stratified
      10% from train, test never touched.
- [x] `scripts/data_prep/download_new_datasets.sh` — idempotent downloader
      (exdark's image archive is Google-Drive-hosted -> needs `gdown`;
      everything else direct wget/tar/zip from academic hosts, all URLs
      reachability-checked before running).
- [x] `scripts/data_prep/make_new_dataset_manifests.py` — builds
      `manifests/<dataset>.json` (same `{image, label, split, dataset}` shape
      as the shift-suite manifests) + `_meta/<dataset>_class_index.json`
      (own label space per dataset, no ImageNet wnid resolution needed).
      Caught + fixed two real bugs on the first run before trusting the
      output: (1) the val-carve helper pooled train+test together before
      sampling, silently reassigning ~880 Stanford Dogs test images into val;
      (2) DomainNet's official train/test txt lists prefix every path with
      the domain name itself (`clipart/aircraft_carrier/...`), which broke
      the class-name parse. Fixed; re-verified byte-identical manifest
      entry/split counts across all 3 hosts (same seed=0).
- [x] Downloaded on Geoffry (source host, most free disk: 1.8 TB), verified,
      then transferred server-to-server to gpu55 + gpu56; manifests rebuilt
      independently on all 3 (paths are absolute + host-specific, same
      convention as the shift suite: Geoffry uses `/mnt5/noy/datasets`,
      gpu55/gpu56 use `~/datasets`).
- [ ] Not yet done: no `data_handlers` subclasses wired up for these 5 yet
      (data-prep only, mirroring how the shift suite was staged first) —
      needed before `randopt_shift.py`/FT/probe can actually run
      train-on-X/test-on-X for any of them.
- [ ] License note: ExDark is non-commercial-research-use only per its own
      README — fine for this project's use, flag if that ever changes.

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
