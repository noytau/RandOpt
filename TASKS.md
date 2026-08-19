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
      - **`ft_scope=all` (full 6.72B-param FT) SOLVED on a single 46GB A6000**
        (2026-08-03) via three stacked fixes, each one found necessary by a
        real crash from the previous attempt — no guessing:
        1. **bf16-stored weights+grad** (`engine.backbone.to(bfloat16)`,
           `predict()` in `ssl_engine.py` made dtype-agnostic to not break
           the existing fp32 paths) — halves fp32's 107.5GB total need to
           ~53.8GB. Confirmed insufficient alone by a real run: OOM'd
           *inside* `opt.step()`, allocating the 4th of 4 required tensors
           (`exp_avg_sq`) at 44.51/44.55GiB.
        2. **8-bit optimizer moments** (`bitsandbytes.optim.AdamW8bit`,
           scoped to `ft_scope=="all"` only) — cuts `exp_avg`/`exp_avg_sq`
           to ~1 byte/param, ~40.3GB total. Real test revealed this wasn't
           actually the dominant cost: it OOM'd *mid-forward-pass* inside a
           block's MLP at ~40.8GB, *before* the optimizer state was ever
           touched — meaning activations, not optimizer state, were the
           real remaining wall.
        3. **Gradient checkpointing**, monkey-patched onto each
           `engine.backbone.blocks[i].forward` at runtime (can't edit the
           pinned dinov3 repo's block-loop code per CLAUDE.md) — trades
           compute for memory by recomputing each block's activations
           during backward instead of keeping all 40 blocks' activations
           resident at once (unlike `last_n_blocks` scope, where the frozen
           early blocks' outputs never require_grad and their activations
           are never saved — full-layer FT gets none of that savings for
           free, which is why this step was the one that actually closed
           the gap). **With all three combined: real, complete 1-epoch
           full-layer FT run finished clean** — `dinov3-ft-all-layers-ckpt-
           repro`, imagenet_c base 85.3%→85.2% (-0.1pt after 1 epoch,
           lr=1e-5, batch_size=16 — expected near-zero movement, this was a
           feasibility check not a real experiment).
      - [x] **Real full-layer FT run, all 6 datasets, 5 epochs, N=2000/K=50
        RandOpt run — all three finished 2026-08-03.** Complete comparison:

        | Dataset | Base | FT `last_n_blocks=5` (839M) | FT `all` (6.72B) | RandOpt N=2000/K=50 |
        |---|---|---|---|---|
        | imagenet | 87.5% | +0.9 | +0.3 | -0.2 |
        | imagenet_c | 85.1% | +0.1 | +0.6 | -0.1 |
        | imagenet_a | 79.8% | +1.0 | +0.25 | 0.0 |
        | imagenet_r | 88.6% | +1.0 | +0.7 | -0.1 |
        | imagenet_sketch | 69.5% | +0.8 | +0.1 | 0.0 |
        | imagenet_es | 83.1% | 0.0 | 0.0 | +0.2 |

        Two findings worth flagging: (1) RandOpt is flat/noise here (±0.2pt
        max) while both FT variants show small, consistent positive gains —
        the *opposite* of the DINOv2 pattern above (FT damages, RandOpt
        holds flat and wins by comparison); for DINOv3 at this N/sigma,
        RandOpt has nothing to "win" against, gradient descent finds real
        (if small) improvements random search doesn't. (2) Scoped FT
        (`last_n_blocks=5`, 839M params) beats full FT (`all`, 6.72B params)
        on 4/6 datasets at the same lr/epochs, despite training 8x fewer
        params — not explained, just observed; plausibly the same nominal
        step size produces a larger effective per-parameter update when
        concentrated in fewer params. `--dataset all,imagenet,imagenet_c`,
        train/test_samples=1000, 5 engines (RandOpt), batch_size=16 (FT).
        Wandb: `dinov3-ft-all-layers-all6-real`, `dinov3-ft-all6-blocks5-
        tr1k`, `dinov3-all6-N2000-K50-tr1k-te1k`.
      - [x] **`delta_w`/`sigma_equiv` re-added** (2026-08-04) — `w0` snapshot
        now clones to CPU (`p.detach().cpu().clone()`, matching
        `SSLEngineImpl.store_base_weights()`'s pattern), diffed back on CPU
        too, so it never touches GPU memory regardless of scope. Motivation:
        DINOv2's full-model FT and DINOv3's both used the same `lr=1e-5`,
        but Adam's update rule normalizes per-parameter step size roughly
        independent of total param count — so "same lr" isn't obviously
        "same push," and the milder/positive DINOv3 result could mean
        genuine robustness, or just too small a displacement at this lr to
        show damage at all. `delta_w` now lets that be answered directly
        instead of guessed at — compare DINOv3's displacement at a given lr
        against DINOv2's already-measured `delta_w=3.27`.
      - [x] **LR sweep result (2026-08-04): confirmed — DINOv3 isn't more
        robust, lr=1e-5 was just too small a displacement to show damage.**
        `dinov3-ft-all-layers-all6-lr1e-4` (10x higher lr, otherwise
        identical config): `delta_w=45.6` (~14x DINOv2's measured
        `delta_w=3.27`), `train_acc=1.0` (full memorization, same
        overfitting signature as DINOv2). Damage reappears and is *worse*
        than DINOv2's: imagenet -0.7 (was +0.3), imagenet_c -1.3 (was +0.6),
        imagenet_a -2.37 (was +0.25), **imagenet_es -21.2** (was 0.0 —
        DINOv2's worst casualty at this dataset was -8.6, this is 2.5x
        worse). imagenet_r still +1.0, imagenet_sketch flat at 0.0. Same
        gradient-descent-overfits-a-random-selection-can't story as DINOv2,
        just needed a big enough displacement to surface at this model
        scale — the earlier "DINOv3 doesn't get damaged" result was an
        lr artifact, not a real scale-dependent finding.
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
- [x] **exdark self-contained pipeline: coded AND run** (2026-08-03/04, by
      the concurrent session flagged below — code was left uncommitted when
      that session's terminal closed; recovered + verified 2026-08-12).
      **Correction (2026-08-16): the "not yet run" note below was wrong** —
      checked gpu55 directly and the full series already completed
      2026-08-02 to 08-04, before the session closed: `checkpoints/
      exdark_head.pth` (fitted head) and `exdark_ft_all_backbone.pth`
      (4.5GB FT'd backbone) exist on disk; `results/randopt-ssl-self-
      exdark-{N2000,N2000-holdout,N3000-holdout}/results.json` are real,
      finished results (verified `randopt-ssl-self-exdark-N3000-holdout`:
      base_test_accuracy=0.8471, ensemble K=50=84.74%, matching
      `results/report.html` exactly). Full results (RandOpt N=2000/3000,
      FT at 3 learning rates, 6-dataset forgetting cross-eval) are already
      written up in `results/report.html` sections 1-7 and the published
      artifact, generated 2026-08-05. Nothing left to run for exdark itself
      — see "New same-distribution train/val/test datasets" section's
      remaining open item (fgvc_aircraft/stanford_dogs/domainnet_*, not
      started) for what's actually still open in this series.
      `data_handlers/exdark.py`
      (`ExDarkHandler`, registered in `data_handlers/__init__.py`) + new
      `perturb_target="backbone"` scope in `vision/ssl_engine.py` (perturbs
      backbone only, excluding head — lets a (seed, sigma) pair be replayed
      against a *different* head) + `load_backbone_state_dict()`. New
      scripts: `vision/head_probe.py` (`fit_linear_head` — generic trainer,
      reused by both fit and FT below), `scripts/fit_head.py` (one-time
      fresh 12-way head fit, since exdark has no pretrained head the way
      ImageNet-1k does), `scripts/randopt_selfcontained.py` (RandOpt scored
      + evaluated entirely on exdark's own train/test, not clean ImageNet —
      `_SELFCONTAINED_REGISTRY`, exdark only so far; fgvc_aircraft/
      stanford_dogs/domainnet_{clipart,sketch} are one registry line +
      handler each, later), `scripts/ft_selfcontained.py` (matched FT
      baseline, same head/data/eval as RandOpt for a fair comparison),
      `scripts/eval_backbone_on_imagenet.py` (catastrophic-forgetting check:
      replay an exdark-adapted backbone against the ORIGINAL frozen
      ImageNet-1k head on imagenet/imagenet_c/a/r/sketch/es).
      `scripts/data_prep/make_new_dataset_manifests.py` also grew a
      `train_holdout` split (disjoint from what `fit_head.py` trains on) so
      RandOpt scoring never re-measures images the head already memorized.
- [ ] License note: ExDark is non-commercial-research-use only per its own
      README — fine for this project's use, flag if that ever changes.

## Multi-step perturbation + layer-scope follow-on (added 2026-08-09)

Motivation: the DINOv3 N=2000/K=50 real run (table above) showed RandOpt flat
(±0.2pt) while FT found small real gains — user's hypothesis is that a single
perturbation step may not shift the model far enough at this sigma; chaining
P perturbations (same sigma each step, only the chain's *final* state scored)
might reach a more informative region. Second follow-on: scope perturbations
to only the top T% of blocks instead of `all`, mirroring FT's existing
`--ft_scope last_n_blocks` machinery.

- [x] `scripts/randopt_shift.py`: added `--perturb_steps` (P, default 1).
      `run_sampling` draws P-seed chains per candidate (same sigma every
      step), applies all P steps before scoring once, restores in reverse.
      `run_ensemble_multi` normalizes each top-K entry via `_as_seed_chain`
      so it transparently accepts either the original flat `(seed, sigma)`
      format (still used by `scripts/eval_backbone_on_imagenet.py` and every
      pre-existing `results.json`) or the new chain format — P=1 draws the
      exact same seeds/sigmas as the pre-change code (verified: identical
      rng draws). Committed `85abd56`.
- [x] **Calibration (N=20, 5 engines, imagenet_c, matched configs) before
      committing GPU time**: per-batch sampling time barely moves with P —
      P=1: 165.4s avg, P=2: 166.2s, P=3: 166.6s, P=5: 168.1s. The extra
      perturb/restore round-trips (Gaussian noise regen over all 6.72B
      backbone params) cost <1s each — negligible next to the ~165s/batch
      forward-pass cost. Projected real-run ETA for P=2/3/5 at N=2000/K=50:
      ~18.5-18.7h sampling + ~1h overhead ≈ same ~19.5-20h as the existing
      P=1 baseline, not a P× multiple.
- [ ] **Real sweep launched** 2026-08-09 19:34 UTC on gpu56, 5 engines, all 6
      datasets (`--dataset all,imagenet,imagenet_c`), N=2000/K=50,
      train/test_samples=1000, sequential wrapper (`~/run_multistep_sweep.sh`
      on gpu56) runs P=2 → P=3 → P=5 back-to-back since only one 5-engine
      DINOv3 job fits in the 251GB host RAM ceiling at a time. Wandb names
      `dinov3-all6-N2000-K50-P{2,3,5}-tr1k-te1k`; results land in
      `results/dinov3-all6-N2000-K50-P{2,3,5}-tr1k-te1k/results.json`. ETA
      ~58-60h total (~2.5 days) from launch.
      - **Both attempts crashed to real, confirmed kernel OOMs from OTHER lab
        members' unrelated jobs (2026-08-10)** — `gpu56`/`gpu55` are shared
        department servers, NOT exclusive to this project as CLAUDE.md's
        table implies. gpu56's P2 leg died at batch 267/400 (66% through
        sampling): user `omertaub`'s 8-GPU diffusion job + SynthSeg
        processes pushed the 251GB host over the edge, `dmesg`-confirmed OOM
        kill of one `ray::SSLEngineImpl` worker. Retried on gpu55 with a
        reduced `num_engines=2` (pinned via `CUDA_VISIBLE_DEVICES` to the 2
        actually-free GPUs, since 6/8 were already held by another user's
        job) — also `dmesg`-confirmed OOM-killed, this time from user
        `timorhalabi`'s `cellonseg` job's `/dev/shm` shard cache (fluctuates
        50-100+GB) eating the remaining ~30GB margin. `RAY_memory_monitor_
        refresh_ms=0` does not prevent this — the kernel OOM killer fires
        regardless. See memory `project_gpu56_gpu55_contention.md`. **Paused
        at user's request (2026-08-10)** — user is checking on server
        access/capacity before any further relaunch attempt.
      - **RAM fix coded 2026-08-12, verified 2026-08-14** (commit `8965fa5`,
        recovered + reviewed after the session that wrote it was closed
        mid-work — see "New same-distribution train/val/test datasets"
        section for that recovery): `store_base_weights`/
        `reset_to_base_weights` (`vision/ssl_engine.py`) no longer keep a
        persistent full-backbone CPU clone alive per engine for DINOv3 —
        only the (tiny) head is cloned into RAM; the backbone is instead
        reloaded from its on-disk checkpoint the one time
        `reset_to_base_weights` actually runs (once per engine, right
        before ensemble eval). Old behavior: every engine held a ~27GB CPU
        snapshot for its whole lifetime just as a drift-safety net used
        once at the very end — 5 engines' idle clones ≈135GB, plausibly the
        dominant host-RAM cost that made the two OOMs above easier to
        trigger (contention from other users' jobs was the proximate cause
        either way, so this reduces exposure, not a guaranteed fix).
        Doesn't touch the DINOv2 path (no on-disk checkpoint to reload
        from, so unaffected).
        **Verification run** (gpu55, `imagenet_c`, N=2000/K=50, P=3,
        `num_engines=2` pinned to the only 2 GPUs free at the time — gpu56
        was down the whole window with its driver purged by another lab
        member, unrelated, see below): launched 2026-08-12 13:59, finished
        cleanly 2026-08-14 14:50 (~49h, in line with the ~46-47h estimate
        for this reduced scale), zero errors/OOM/kills in the log, host RAM
        back to 162/220GB available after exit. `base_test_accuracy=0.853`,
        `best_sigma=0.001`, `ensemble K=50 accuracy=85.3%` (+0.0pp vs base —
        flat, consistent with the existing DINOv3-RandOpt-is-flat finding
        above; this run's purpose was the RAM fix, not a new result).
        wandb: `dinov3-imagenet_c-N2000-K50-P3-2eng-tr1k-te1k`. **RAM fix
        confirmed working** — full 5-engine relaunch of the original
        P=2→3→5/N=2000/all6 sweep is unblocked whenever gpu56 (or more free
        GPUs on gpu55) are available.
      - **gpu56 driver outage, 2026-08-11 to ~2026-08-15/16** (separate,
        unrelated incident, discovered mid-verification-run when SSH to
        both gpu56 AND gpu55 started timing out and had to be diagnosed):
        another lab member, `erez`, ran `apt-get purge` on the entire
        NVIDIA driver stack on gpu56 (2026-08-11 09:59, per
        `/var/log/apt/history.log`), followed by 3 reboots landing on a new
        kernel (`6.8.0-137-generic`); no driver was reinstalled for days,
        leaving gpu56 with 0 usable GPUs (`nvidia-smi` gone,
        `torch.cuda.is_available()` → False). Confirmed reinstalled and
        healthy again as of 2026-08-16 (`torch.cuda.device_count()` → 8).
        The multi-day SSH-unreachable window for gpu55 in between was
        separate and never root-caused (both hosts timed out identically,
        consistent with a network/VPN-side issue rather than a second host
        failure — gpu55's uptime showed no reboot across that window, and
        the verification run above completed successfully in the middle of
        it, confirming gpu55 itself was fine throughout).
- [ ] **Layer-scope sweep, next** (after the multi-step sweep frees gpu56):
      RandOpt with `--perturb_target last_n_blocks --last_n_blocks N` at
      N=4/8/12/20 (10/20/30/50% of DINOv3's 40 blocks), same N=2000/K=50, all
      6 datasets, sequential (~19-20h each, ~80h total). No new code needed
      — `last_n_blocks` scoping already exists and is exercised by
      `ft_matched_baseline.py`'s `--ft_scope`.
- [x] Item raised alongside these two: a separate concurrent Claude session
      was running RandOpt-vs-FT on DINOv2 + new datasets (exdark, etc.) — its
      terminal got closed 2026-08-12 before it committed; work recovered +
      verified, see "New same-distribution train/val/test datasets" section
      above for what it left coded and what's still needed to run it.

## ExDark on DINOv3 (added 2026-08-16)

ExDark so far only ran on DINOv2 (see above) — user asked for the same
FT-vs-RandOpt comparison on DINOv3, as two experiments: (1) adapt on clean
ImageNet, check transfer/damage on ExDark; (2) adapt on ExDark itself, check
transfer/damage on the 6 ImageNet-family datasets (mirrors the DINOv2
ExDark series' own forgetting check). Both: FT lr=1e-5/all-layers, RandOpt
N=2000/K=50. **All 8 runs complete as of 2026-08-19.**

- [x] Tested the DINOv3+ExDark combination for the first time (toy scale,
      2 engines, 24 samples): `fit_head.py`, `randopt_selfcontained.py`
      (full loop incl. the reset-from-disk RAM-fix path), and
      `ft_selfcontained.py` all confirmed working.
- [x] Found + fixed a real bug during that test: DINOv3's default
      `weights_path`/`head_path` (`vision/ssl_engine.py`'s `_FAMILIES`)
      pointed at the RunAI cluster location only, no valid fallback on
      gpu55/gpu56. Never bit a real run before (every launch script passes
      the flag explicitly) but broke ad-hoc testing. Fixed the default.
- [x] Found + fixed a second gap while preparing experiment (2)'s FT half:
      `ft_selfcontained.py`/`vision/head_probe.py`'s `fit_linear_head` had
      never been exercised at `ft_scope="all"` on a DINOv3-scale backbone —
      would OOM (same wall `ft_matched_baseline.py` already solved: fp32
      AdamW needs ~107.5GB for 6.72B params). Ported that script's bf16 +
      gradient-checkpointing + 8-bit-AdamW approach into `fit_linear_head`
      rather than duplicating it. Also fixed `ft_selfcontained.py`'s
      `delta_w` snapshot (`w0`) to be CPU-resident, matching a fix
      `ft_matched_baseline.py` already had — was silently GPU-resident
      (~27GB extra) on top of the now-tight all-scope footprint.
- [x] `scripts/ft_matched_baseline.py`: added `--backbone_out` — it never
      saved the fine-tuned backbone before (measured accuracy, discarded
      the weights), so the earlier "DINOv3 FT lr=1e-5/all-layers on
      ImageNet" run's numbers exist but the adapted weights don't. Needs a
      rerun to get a reusable checkpoint for experiment (1)'s FT half.
- [x] `scripts/eval_backbone_on_exdark.py`: new script, mirrors
      `eval_backbone_on_imagenet.py` in the opposite direction — takes an
      ImageNet-adapted backbone (FT or RandOpt) and evaluates it against a
      freshly-fit ExDark head instead of the reverse. `--mode base/randopt/
      ft`, reuses `run_ensemble_multi` and the `perturb_target="backbone"`
      replay mechanism already proven for the ExDark→ImageNet direction.
**Results (2026-08-16 to 08-19, gpu56).** All steps below run
`--backbone_family dinov3`, ExDark head = `checkpoints/exdark_head.pth`
(fit fresh: 2400 train images, lr=1e-3, 10 epochs, train_acc=0.9971,
held-out test_acc=0.8521 — healthy gap, not memorization, on par with the
DINOv2 series' equivalent 0.998 train / 0.8471 test).

| Step | What | Result |
|---|---|---|
| Exp 1, RandOpt (reused `dinov3-all6-N2000-K50-tr1k-te1k`, no rerun needed — RandOpt's adapted state is fully reconstructible from its saved `(seed,σ)` list, unlike FT's actual trained weights) | ImageNet-trained RandOpt backbone → ExDark | 85.21% vs 85.21% base — **flat, +0.00pp** |
| Exp 1, FT (`ft_matched_baseline.py --ft_scope all --lr 1e-5`, rerun to capture `--backbone_out`) | ImageNet-trained FT backbone → ExDark | 85.06% vs 85.21% base — **−0.16pp** (small damage, out-of-distribution) |
| Exp 2, RandOpt (`randopt_selfcontained.py --dataset exdark`, N=2000/K=50, 7 engines) | ExDark-trained RandOpt | 85.17% vs 85.21% base — **−0.04pp** (flat). wandb `dinov3-exdark-N2000-K50-7eng`, results `results/randopt-ssl-dinov3-self-exdark-N2000/results.json` |
| Exp 2, FT (`ft_selfcontained.py --dataset exdark --ft_scope all --lr 1e-5`) | ExDark-trained FT | 85.68% vs 85.21% base — **+0.47pp** (real gain) |
| Exp 2, RandOpt forgetting-check (6 ImageNet datasets) | ExDark-RandOpt backbone vs original ImageNet head | max deviation **±0.30pp** — imagenet −0.1, imagenet_c −0.1, imagenet_a +0.0, imagenet_r −0.1, sketch +0.2, es +0.3. Flat, same pattern as every other RandOpt result on DINOv3 |
| Exp 2, FT forgetting-check (6 ImageNet datasets) | ExDark-FT backbone vs original ImageNet head | max deviation **±0.25pp** across all 6 — imagenet +0.0, imagenet_c +0.2, imagenet_a −0.25, imagenet_r +0.2, sketch +0.0, es +0.2. Much milder than the DINOv2 ExDark series' own forgetting check (which found −8.1pp on imagenet_es) |

**Final pattern, all 8 runs complete:** RandOpt stays essentially flat on
DINOv3 everywhere it was tested — both training directions (ImageNet→ExDark,
ExDark→ImageNet-family) and both primary-task and forgetting-check numbers
never exceed ±0.30pp. FT finds a real gain when trained on ExDark itself
(+0.47pp) but shows small damage when trained on a different domain and
evaluated cold on ExDark (−0.16pp) — direction matters for FT, not for
RandOpt. FT's forgetting footprint on DINOv3 (±0.25pp max) is far smaller
than the equivalent DINOv2 result (−8.1pp concentrated on imagenet_es) —
whatever made DINOv2's FT damage concentrated and severe on that one
dataset doesn't reproduce at DINOv3's scale, at least at this lr/budget.

## Open tasks

- [x] **FT matched baseline** (scripts/ft_matched_baseline.py): done and
      superseded — DINOv2 tr1k/tr5k results landed 2026-08-02 (see top of
      this file), and the fuller DINOv3 FT comparison (last_n_blocks=5 +
      full model, 3 lr's, all 6 datasets) landed 2026-08-03/04 (see "DINOv3
      Experiments" work below and the published reports).
- [x] **New GPU servers bring-up** (gpu56/gpu55) — done. Both have been in
      active use for weeks: key auth working, code synced (`myfork` =
      noytau/RandOpt.git on both), data present, dozens of real runs
      completed (N=2000/3000 DINOv3 sweeps, ExDark series, multi-step
      verification run 2026-08-14). gpu56 hit an unrelated driver outage
      2026-08-11 to ~08-15 (another lab member purged NVIDIA drivers,
      confirmed fixed 2026-08-16) — not a bring-up gap, a later incident.
- [ ] Design tier (brainstorm with Nimrod, never auto-implement): kNN-centered
      scoring arm; unadapted-center setup; two-stage scoring
- [ ] Docker image rebuild (only relevant if RunAI is used again — fallback
      only per policy below, not touched in weeks): bake in `wandb
      torchmetrics termcolor omegaconf ftfy submitit scikit-learn regex`
- [ ] Report expired cluster-api TLS cert (`hpc-vm-runai.eng.tau.ac.il`) to
      admins — blocks `runai logs`/`exec`; workaround: W&B run `output.log`
      (low priority while RunAI is unused)
- [ ] Adapt or delete stale `scripts/dinov2_hub_inference.py` — still sitting
      uncommitted and untouched, unrelated to any current work thread

## Job venue policy (updated 2026-08-16)

**Standalone GPU servers `gpu56`/`gpu55` are the default now** (bring-up is
done, both actively used) — large GPUs, no scheduler roulette, no daily
token, no noisy-neighbor OOM kills (though real contention from other lab
members' jobs on these SHARED servers is still possible — check
`nvidia-smi` before every submission). Fall back to RunAI for multi-node
scale only.

## Cluster (RunAI) operating rules — for fallback use

- Pin big-model jobs: `--node-pools raja` (48G A6000s). hpc-node3 = 24 GB.
- ALWAYS `--cpu-memory-request` (64G ViT-g / 128G d3-2-engine jobs).
- Add `export RAY_memory_monitor_refresh_ms=0` to every Ray job command.
- `-g N` cannot combine with `--gpu-request-type memory`; no exclude-node.
- `--backoff-limit 0` always; runai token expires daily (`runai login`).
- W&B name convention: `randopt-<d2|d3>-n<N>-k<K>-tr-<set><size>-te-<set><size>`.
