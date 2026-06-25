---
name: project-flow-diagram
description: >-
  Generates mermaid flow diagrams and config tables for ML fine-tuning projects
  (LeRobot/MolmoAct2-style layouts with config.py, prepare_dataset.py, train.py).
  Use when the user asks for a repo map, flow diagram, architecture overview,
  how prepare_dataset works, where configs live, or current hyperparameter values.
disable-model-invocation: true
---

# Training Project Flow Diagram

Produce a visual map of a fine-tuning project: repo layout, data prep flow,
training loop, config sources, and **live values read from disk** (never guess).

## When to run

User mentions: flow diagram, repo map, project structure, how prepare works,
where config lives, current values, molmo/lerobot project overview.

## Step 1 — Resolve project root

Search in order:

1. Directory from conversation context (e.g. `molmoact2-record-test/`)
2. Nearest `*/scripts/config.py` + `*/scripts/train.py`
3. Ask user if ambiguous

Set `PROJECT_ROOT` = parent of `scripts/`.

## Step 2 — Read these files (required)

| File | Purpose |
|------|---------|
| `scripts/config.py` | All hyperparameters and paths |
| `scripts/prepare_dataset.py` | Split/stats/idempotency logic |
| `scripts/train.py` | Training loop, eval, checkpoints |
| `scripts/env_setup.py` | HF cache, CUDA env (if present) |
| `scripts/requirements.txt` | Dependencies |
| `specs/PLAN.md` | Human docs (optional) |
| `data/train/.prepared` | Live split stats (if exists) |
| `data/val/.prepared` | Live split stats (if exists) |

Use `Read` only — do not execute scripts.

## Step 3 — Extract config values

From `config.py`, build tables grouped as:

- **Paths & datasets** — repos, fractions, DATA_DIR, OUTPUT_DIR
- **Model** — CHECKPOINT_PATH, NORM_TAG, SETUP_TYPE, CONTROL_MODE, IMAGE_KEYS
- **Training** — STEPS, BATCH_SIZE, workers, LOG/EVAL/SAVE_FREQ, LoRA/action-expert flags, ACTION_MODE, dtypes
- **WandB** — enable, project, entity
- **Environment** — HF_HOME, cache paths from env_setup.py

If `.prepared` markers exist, add an **On-disk state** table (episodes, frames per split).

Note when `PLAN.md` disagrees with `config.py` — **config.py wins**.

## Step 4 — Output structure (always use this order)

### 4.1 Repo layout diagram

Mermaid `flowchart TB` with subgraphs:

- `docs` — PLAN.md, README.md
- `scripts` — config.py (single source of truth), env_setup, prepare_dataset, train, requirements
- `data` — source/, train/, val/, .prepared markers
- `outputs` — checkpoints, wandb, logs
- `external` — HF dataset + base model

### 4.2 End-to-end flow

Mermaid `flowchart LR`:

1. User runs `train.py`
2. `is_prepared()` check → auto `prepare_dataset` or skip
3. prepare steps: snapshot_download → split_dataset → recompute_stats → .prepared
4. Build MolmoAct2Config / policy config from config.py
5. Load base model from HF
6. Train loop with eval + checkpoint + WandB intervals from config

Include a table: how prepare is invoked (standalone, `--force`, auto from train).

### 4.3 Config map diagram

Mermaid showing: `config.py` → policy config + train pipeline config + dataset config → make_policy / make_dataset / training loop.

State clearly: **edit config.py for almost everything**.

### 4.4 Current values tables

Markdown tables from Step 3. Label section **Current values from config.py**.

### 4.5 Training loop sequence diagram

Mermaid `sequenceDiagram`: train.py ↔ prepare_dataset ↔ datasets ↔ policy ↔ WandB.

Note if built-in LeRobot `eval_freq=-1` and custom `evaluate()` handles validation (read train.py to confirm).

### 4.6 prepare_dataset data flow

Mermaid `flowchart TD`: HF Hub → data/source → split 90/10 → train/val → recompute_stats(train only) → .prepared.

One sentence: **why stats are recomputed on train only** (no val leakage in normalization).

### 4.7 Command cheat sheet

```bash
cd <PROJECT_ROOT>/scripts
python prepare_dataset.py          # optional; train auto-runs if needed
python train.py
python prepare_dataset.py --force  # rebuild from scratch
```

### 4.8 One-sentence mental model

`config.py` → `prepare_dataset.py` (HF → local splits + stats) → `train.py` (fine-tune + eval + WandB + checkpoints).

## Rules

- **Read files first** — never invent current hyperparameter values.
- Use **mermaid** for all diagrams (flowchart TB/LR/TD, sequenceDiagram).
- Keep prose concise; diagrams carry the structure.
- Do not modify any project files unless user explicitly asks.
- If project differs from MolmoAct2 (no prepare_dataset, different layout), adapt diagrams to match actual files found.
- Mention `env_setup.py` import side-effect if present (runs before HF imports).

## Optional follow-up

If user asks, diagram the **LeRobot library layer** under the wrapper scripts
(MolmoAct2Config, make_policy, make_dataset) by grepping installed `lerobot` package.
