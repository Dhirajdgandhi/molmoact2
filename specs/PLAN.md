# Plan

Fine-tune **MolmoAct2** on the LeRobot v3 dataset `dhirajdg/record-test_20260607_235017` with a 90/10 episode split and WandB loss tracking.

## Dataset summary

| Property | Value |
|----------|-------|
| Episodes | 10 |
| Frames | 2,299 |
| Robot | SO follower (`so_follower`) |
| Action/state dim | 6 joint positions |
| Cameras | `observation.images.top`, `observation.images.wrist` |
| Task | Brush solar panel and return to base |

**90/10 split (by episode):** 9 train episodes (1,899 frames) + 1 eval episode (400 frames, episode 9).

## Tasks

1. ✅ **Project setup** — Create `molmoact2-record-test/` with scripts and config.
2. ✅ **Prepare dataset** — Split 90/10, recompute normalization stats from train split only. _(Script: `scripts/prepare_dataset.py`)_
3. 🔄 **Fine-tune MolmoAct2** — Action-expert-only, batch_size=2 on L4 GPU. _(Script: `scripts/train.py`)_
4. 🔄 **Monitor losses** — Log train/eval loss to WandB and console.

## What to watch during training

### Metrics logged

| Metric | Meaning |
|--------|---------|
| `train/loss` | Total training loss (flow + discrete if `action_mode=both`) |
| `train/action_flow_loss` | Flow-matching MSE on continuous actions |
| `train/discrete_ce_loss` | Cross-entropy on discrete action tokens |
| `eval/loss` | Same losses on held-out episode(s) — **primary overfitting signal** |
| `train/lr` | Learning rate (cosine decay after warmup) |
| `train/grad_norm` | Gradient norm — spikes may mean instability |

### Healthy patterns (small dataset)

- **Train loss** should trend down over the first 500–2,000 steps.
- **Eval loss** should follow train loss at roughly the same scale. A small gap is normal.
- **Warning signs:**
  - Train loss keeps dropping but eval loss rises → overfitting (stop early or reduce steps).
  - Loss flat from step 0 → LR too low, wrong normalization, or data issue.
  - Loss is NaN → check `meta/stats.json` for zero-variance dimensions.
  - Eval loss much higher than train from the start → split or stats mismatch.

### Recommended actions

- For 10 demos, prefer **LoRA VLM** (`enable_lora_vlm=true`) over full fine-tune.
- Start with **~3,000 steps**, `batch_size=8`, `eval_every=100` steps.
- Save checkpoints every 500 steps; pick the checkpoint with **lowest eval loss**, not the last step.
- Real success is measured on the **physical robot**, not eval loss alone.

## File layout

```
molmoact2-record-test/
├── specs/PLAN.md
├── scripts/
│   ├── config.py           # All hyperparameters and paths
│   ├── prepare_dataset.py  # 90/10 split + stats recompute
│   ├── train.py            # Training + eval loss logging
│   └── requirements.txt
├── data/                   # Local train/val datasets (created by prepare)
└── outputs/                # Checkpoints and logs (created by train)
```

## Usage

```bash
cd molmoact2-record-test/scripts
pip install -r requirements.txt
python prepare_dataset.py   # once
python train.py
```
