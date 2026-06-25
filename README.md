# MolmoAct2 Fine-Tuning

Fine-tune [MolmoAct2](https://huggingface.co/allenai/MolmoAct2-SO100_101) on a LeRobot v3 dataset with 90/10 train/eval split and WandB logging.

## Dataset

- Source: [`dhirajdg/record-test_20260607_235017`](https://huggingface.co/datasets/dhirajdg/record-test_20260607_235017)
- Robot: SO follower, 6-DOF joint positions, top + wrist cameras

## Setup

```bash
cd scripts
pip install -r requirements.txt
hf auth login
```

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

## Project layout

```
specs/PLAN.md          # Training plan and loss monitoring guide
scripts/
  config.py            # All paths and hyperparameters
  prepare_dataset.py   # Dataset split + stats
  train.py             # Training + eval loss logging
  run_train.sh         # Start training via nohup (use this, not train.py directly)
  env_setup.py         # HF cache on /tmp for large checkpoints
  requirements.txt
```

See `specs/PLAN.md` for what to watch in training/eval loss curves.
