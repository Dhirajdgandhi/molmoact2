---
name: training-eval-analysis
description: >-
  Writes a 2–3 page research-advisor report after MolmoAct2 training or offline
  eval completes: hyperparameter table, loss trends, delta vs prior run, paper-grounded
  interpretation, and actionable suggestions. Saves to outputs/analyses/. Use when
  training finishes, offline eval completes, run_eval.sh returns, the user asks for
  loss analysis, iteration comparison, or advisor-style conclusions.
---

# Training & Offline Eval Analysis

Produce a **saved markdown report** (2–3 pages max) and a **short chat summary** whenever training or offline eval completes.

## When to run

- User says training finished, eval done, analyze losses, compare iterations, advisor review
- `outputs/offline_eval.json` or `train.log` just updated
- User invokes this skill after `./run_train.sh` or `./run_eval.sh`

## Workflow

```
1. Collect metrics snapshot (required)
2. Read reference.md for paper context & heuristics
3. Compare vs previous iteration (if any)
4. Write report from templates/report-template.md
5. Save + present summary
```

### Step 1 — Collect data (do not guess)

From project root:

```bash
python scripts/collect_metrics_snapshot.py --event training --out outputs/metrics_snapshot.json
# or
python scripts/collect_metrics_snapshot.py --event offline_eval --out outputs/metrics_snapshot.json
```

Also read when present:

| Source | Use |
|--------|-----|
| `outputs/metrics_snapshot.json` | Config, best_meta, eval history, train log curve |
| `outputs/offline_eval.json` | Latest offline metrics |
| `outputs/eval_runs/*.json` | Prior offline runs (newest last) |
| `outputs/analyses/*.md` | Prior advisor reports |
| `specs/PLAN.md` | Dataset counts, healthy-loss guidance |
| `scripts/config.py` | Verify any value missing from snapshot |

If WandB is enabled, mention the project URL pattern `wandb.ai/<entity>/molmoact2-record-test` but do not block on API access.

### Step 2 — Compare to previous iteration

| Event | Prior baseline |
|-------|----------------|
| **offline_eval** | Previous file in `outputs/eval_runs/` (second-to-last), or last `outputs/analyses/*-offline_eval-*.md` |
| **training** | Previous `best_meta.json` if resumed, prior analysis, or last eval curve minimum |

Report **deltas** with sign and % where meaningful:

- `eval/loss`, `eval/action_flow_loss`
- `open_loop/action_mse`, `open_loop/action_rmse`
- Per-joint MSE (joint 4 often highest on SO arms)
- Best training eval loss @ step

State **improved / regressed / unchanged** per metric. If no prior run exists, say so once and skip delta section filler.

### Step 3 — Write the report

- Follow [templates/report-template.md](templates/report-template.md)
- **Hard limit: 2–3 pages** (~1,200–1,800 words). Cut repetition before cutting the hyperparameter table or advisor conclusion.
- Tone: research advisor — precise, honest about small-data limits, not hype.
- Ground claims in [reference.md](reference.md) (MolmoAct2, flow matching, small-demo fine-tuning). Cite papers in a short **References** block (3–5 entries max).

### Step 4 — Save

```text
outputs/analyses/YYYYMMDD-HHMMSS-<event>-<short-checkpoint-slug>.md
```

Example: `outputs/analyses/20260701-001045-offline_eval-step3000-eval03562.md`

Append a one-line index to `outputs/analyses/README.md` if the file exists; create it if not:

```markdown
- 2026-07-01 offline_eval step3000 — eval/loss 0.377 → [report](20260701-001045-offline_eval-step3000-eval03562.md)
```

### Step 5 — Present to user

In chat (not the full report):

1. One-sentence verdict (ready for robot / needs more data / overfitting risk)
2. Top 3 metric deltas vs prior run
3. Top 2–3 actionable suggestions
4. Link/path to saved report

## Metric semantics (this project)

| Metric | Meaning |
|--------|---------|
| `train/loss`, `eval/loss` | Flow-matching loss (continuous actions); primary training signal |
| `eval/action_flow_loss` | Same component on val episode(s) |
| `open_loop/action_mse` | Predicted vs recorded actions without teacher forcing |
| `open_loop/joint_k_mse` | Per-DOF error; joint 4 often dominates on reach motions |

**Offline eval loss ≈ training eval loss** only when the same checkpoint and val split are used; small gaps are normal.

## Suggestions library (pick 3–5, tailored)

Use snapshot + reference.md; do not dump the full list.

- **More demos**: 10 episodes is far below MolmoAct2 paper scale; collect 50–200 episodes before architecture changes.
- **LoRA VLM**: `ENABLE_LORA_VLM=True` when action-expert-only plateaus (see PLAN.md).
- **Stop early**: If train ↓ but eval ↑ after ~1500 steps, use best checkpoint not last step.
- **Robot eval**: Paper success is real-world; run 10–20 on-robot trials even if offline loss looks good.
- **Joint 4**: High wrist/elbow MSE → check camera visibility of wrist cam during reach-to-panel phase.
- **Hub checkpoint**: Deploy `best` by eval loss, not final step; confirm with `./scripts/show_latest_checkpoint.sh`.

## Anti-patterns

- Do not invent hyperparameters — always from snapshot/config
- Do not claim sim-to-real success from offline loss alone
- Do not exceed 3 pages
- Do not skip saving the report file

## Additional resources

- Paper context & citations: [reference.md](reference.md)
- Report structure: [templates/report-template.md](templates/report-template.md)
