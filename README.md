# MolmoAct2 Fine-Tuning

Fine-tune [MolmoAct2](https://huggingface.co/allenai/MolmoAct2-SO100_101) on a LeRobot v3 dataset with 90/10 train/eval split and WandB logging.

## Dataset

- Source: [`dhirajdg/record-test_20260607_235017`](https://huggingface.co/datasets/dhirajdg/record-test_20260607_235017)
- Robot: SO follower, 6-DOF joint positions, top + wrist cameras

## Setup

```bash
cp .env.example .env   # or export HF_TOKEN, GIT_* and run sync below
./scripts/sync_secrets.sh
cd scripts
pip install -r requirements.txt
```

`sync_secrets.sh` writes credentials from your shell environment or `.env` to:

- `~/.cache/huggingface/token` (HF Hub)
- `.git/credentials` + repo-local git credential helper (GitHub push)
- `git config --global user.name` / `user.email` (commit identity)

It runs automatically at the start of `run_eval.sh`, `push_code.sh`, and the Docker entrypoint.

`GIT_TOKEN` must be a [GitHub Personal Access Token](https://github.com/settings/tokens) with `repo` scope (account passwords no longer work over HTTPS).

## Push code

After committing locally:

```bash
./scripts/push_code.sh
```

Loads `GIT_USERNAME` and `GIT_TOKEN` from `.env` and pushes to `origin` without prompting.

## Usage

```bash
# 1. Download, split 90/10, recompute train normalization stats
python prepare_dataset.py

# 2. Fine-tune (always use run_train.sh — runs detached via nohup)
./run_train.sh
```

Training takes ~55 minutes and must run detached so it is not killed when the IDE session ends. **Do not run `python train.py` directly** unless you are debugging a single step.

Monitor progress:

```bash
tail -f ../outputs/train.log
```

Stop training:

```bash
kill "$(cat ../outputs/train.pid)"
```

Edit hyperparameters in `scripts/config.py`.

## Offline evaluation

Evaluate a deployed Hugging Face checkpoint on the held-out val split (no robot required):

```bash
cd scripts
./run_eval.sh --checkpoint dhirajdg/molmoact2-record-test-step3000-eval03562-20260625
```

`run_eval.sh` loads `HF_TOKEN` from the environment, project `.env`, or `~/.cache/huggingface/token` (after `hf auth login`).

Results are written to `outputs/offline_eval.json`.

Quick smoke test (first 5 batches):

```bash
./run_eval.sh --checkpoint dhirajdg/molmoact2-record-test-step3000-eval03562-20260625 --max-batches 5
```

## Docker (reproducible on ephemeral instances)

Build once, then run eval on any GPU host without reinstalling dependencies:

```bash
# From repo root
docker build -t molmoact2-eval .

docker run --rm --gpus all \
  -e HF_TOKEN="$HF_TOKEN" \
  -e GIT_USERNAME="$GIT_USERNAME" \
  -e GIT_TOKEN="$GIT_TOKEN" \
  -e GIT_USER_NAME="$GIT_USER_NAME" \
  -e GIT_USER_EMAIL="$GIT_USER_EMAIL" \
  -v molmoact2-hf-cache:/tmp/huggingface \
  -v "$(pwd)/outputs:/app/outputs" \
  molmoact2-eval \
  --checkpoint dhirajdg/molmoact2-record-test-step3000-eval03562-20260625
```

Copy `.env.example` to `.env` and set your token, or pass `-e HF_TOKEN=...` directly. The volume mount keeps checkpoint downloads across container restarts.

## Project layout

```
specs/PLAN.md          # Training plan and loss monitoring guide
scripts/
  config.py            # All paths and hyperparameters
  prepare_dataset.py   # Dataset split + stats
  train.py             # Training + eval loss logging
  run_train.sh         # Start training via nohup (use this, not train.py directly)
  run_eval.sh          # Offline eval from a local or Hub checkpoint
  eval_offline.py      # Eval implementation (teacher-forcing + open-loop MSE)
  env_setup.py         # HF cache on /tmp for large checkpoints
  requirements.txt
Dockerfile             # Reproducible GPU eval image
```

See `specs/PLAN.md` for what to watch in training/eval loss curves.
