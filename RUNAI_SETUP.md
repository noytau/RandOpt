# RandOpt RunAI Setup Guide

## Prerequisites
- Docker Desktop (Mac)
- RunAI CLI installed and logged in
- kubectl installed
- Docker Hub account (`noyhassid`)
- GitHub fork: `https://github.com/noytau/RandOpt`

---

## 1. Build & Push the Docker Image (AMD64)

The image must be built for `linux/amd64` — Mac is ARM64 and local cross-compilation is too slow/unreliable. Use GitHub Actions instead.

### One-time GitHub Actions setup
1. Generate a Docker Hub access token: hub.docker.com → Account Settings → Security → **New Access Token** (Read & Write)
2. Add it as a GitHub secret: `github.com/noytau/RandOpt` → Settings → Secrets and variables → Actions → New secret → name: `DOCKERHUB_TOKEN`
3. Trigger the build: Actions tab → "Build & Push Docker image (linux/amd64)" → **Run workflow**

The workflow lives at [`.github/workflows/docker-build.yml`](.github/workflows/docker-build.yml) and pushes to `docker.io/noyhassid/randopt-vllm:latest`.

### Re-building after Dockerfile changes
Commit and push the updated `docker/Dockerfile_vllm` to `noytau/RandOpt`, then trigger the workflow again.

---

## 2. Create a RunAI Workspace

Run from your local machine (Mac or Geoffrey).

**New CLI (Mac):**
```bash
runai workspace submit randopt-ws \
  --image docker.io/noyhassid/randopt-vllm:latest \
  --project raja \
  -g 1 \
  --existing-pvc claimname=storage,path=/storage
```

**Old CLI (Geoffrey):**
```bash
runai submit randopt-ws \
  --image docker.io/noyhassid/randopt-vllm:latest \
  --project raja \
  -g 1 \
  --interactive \
  --existing-pvc claimname=storage,path=/storage
```

> **Important:** `--existing-pvc claimname=storage,path=/storage` is required. Without it, `/storage` is empty and not persistent.

Wait for the pod to be Running:
```bash
runai workspace list -p raja    # new CLI
runai list -p raja              # old CLI
```

---

## 3. Connect to the Workspace

**New CLI:**
```bash
runai workspace exec randopt-ws -p raja -it -- bash
```

**Old CLI (Geoffrey):**
```bash
runai bash randopt-ws -p raja
```

---

## 4. First-Time Setup Inside the Container

Run once after creating a fresh workspace. The repo and data persist on `/storage` across pod restarts **as long as the same PVC is mounted**.

### Clone the repo
```bash
cd /storage/noy/RandOpt
git init
git remote add origin https://github.com/noytau/RandOpt.git
git pull origin main
```

### Download and extract datasets
```bash
pip install gdown -q
gdown "https://drive.google.com/uc?id=1PiAYvjZOk3VuEyGIeft7d4HynCK1lrur" -O /tmp/data.zip
unzip -q /tmp/data.zip -d /tmp/data_extracted
mv /tmp/data_extracted/data/* /storage/noy/RandOpt/data/
rm -rf /tmp/data.zip /tmp/data_extracted
```

Verify:
```bash
ls /storage/noy/RandOpt/data
# Expected: countdown  gsm8k  gqa  math500  mbpp  olympiadbench  rocstories  uspto50k
```

---

## 5. Run the Benchmark

```bash
cd /storage/noy/RandOpt
export WANDB_API_KEY=<your_key>   # get from wandb.ai → Settings → API keys
CUDA_DEVICES=0 bash scripts/local_run.sh
```

> `CUDA_DEVICES=0` overrides the default `0,1,2,3` since the workspace has 1 GPU.  
> W&B project defaults to `randopt`. Override with `WANDB_PROJECT=my-project`.

Results are saved to `randopt-experiment-local/` on `/storage`.

---

## 6. Manage the Workspace

```bash
# Delete workspace
runai delete job randopt-ws -p raja

# View logs (non-interactive job)
runai logs randopt-run -p raja --follow
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `no match for platform in manifest` | Image is ARM64, cluster needs AMD64 | Rebuild via GitHub Actions |
| `ImagePullBackOff` | Wrong platform or image not pushed | Check Docker Hub manifest with `docker buildx imagetools inspect docker.io/noyhassid/randopt-vllm:latest` |
| `/storage/noy/RandOpt` is empty | PVC not mounted | Re-submit with `--existing-pvc claimname=storage,path=/storage` |
| GPU memory full on startup | Leftover CUDA context from a crashed run | Delete and recreate the workspace (do NOT kill processes inside) |
| `data/countdown/countdown.json` not found | Data zip extracted with extra nesting | Run `mv /storage/noy/RandOpt/data/data/* /storage/noy/RandOpt/data/` |
