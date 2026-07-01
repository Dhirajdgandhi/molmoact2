# MolmoAct2 Run Analysis — {{TITLE}}

**Date:** {{ISO_DATE}}  
**Event:** {{training | offline_eval}}  
**Checkpoint:** {{CHECKPOINT_OR_HUB_REPO}}  
**Analyst:** Research advisor (automated)

---

## Executive summary

{{2–4 sentences: what ran, headline result, vs prior iteration, deployment readiness verdict}}

---

## Experimental configuration

### Dataset

| Field | Value |
|-------|-------|
| Source repo | |
| Train / val repos | |
| Episodes (train / val / total) | |
| Frames (train / val / total) | |
| Split method | episode-level |
| Task | |
| Robot / action dim | |
| Cameras | |

### Model & training hyperparameters

| Field | Value |
|-------|-------|
| Base checkpoint | |
| Fine-tune mode | action-expert-only / LoRA VLM |
| Steps | |
| Batch size | |
| Effective batch (× grad accum if any) | |
| Learning rates (global / ViT / connector / action expert) | |
| Warmup steps | |
| Scheduler | cosine (after warmup) |
| Seed | |
| Action chunk / flow steps | CHUNK_SIZE, N_ACTION_STEPS, NUM_FLOW_TIMESTEPS |
| Dtype / gradient checkpointing | |
| Log / eval / save frequency | LOG_FREQ, EVAL_EVERY, SAVE_FREQ |
| Checkpoint policy | best + keep N periodic |
| Hub push | repo prefix, unique best repos |

### Hardware & environment

| Field | Value |
|-------|-------|
| GPU | (from context if known, e.g. L4 23GB) |
| HF cache | /tmp/huggingface |
| WandB project | |

---

## Results

### Training (if applicable)

| Metric | Value | @ step |
|--------|-------|--------|
| Best eval loss | | |
| Best action flow loss | | |
| Final train loss (if logged) | | |

Brief trend: {{1 short paragraph on eval curve — improving, plateau, overfit}}

### Offline evaluation (if applicable)

| Metric | Value |
|--------|-------|
| eval/loss | |
| eval/action_flow_loss | |
| open_loop/action_mse | |
| open_loop/action_rmse | |
| joint_0 … joint_5 MSE | |

Val coverage: {{episodes}} episodes, {{frames}} frames

---

## Comparison to previous iteration

| Metric | Previous | Current | Δ | Verdict |
|--------|----------|---------|---|---------|
| | | | | improved / regressed / n/a |

If no prior run: state "First tracked iteration."

---

## Interpretation (paper-grounded)

{{1–2 paragraphs tying results to MolmoAct2 flow-matching action expert, small SO-100 fine-tune, and episode-held-out val limitations. Reference arXiv:2605.02881 where relevant.}}

---

## Recommendations

1. {{highest priority}}
2. {{second}}
3. {{third}}

Optional: data collection, hyperparameter, robot eval protocol, checkpoint/HF workflow.

---

## Advisor conclusion

**Recommendation:** {{Proceed to robot eval | Iterate training | Collect more demos | Hold deployment}}

{{1 short paragraph: confidence, caveats, what would change your mind}}

---

## References

1. Fang et al., MolmoAct2: Action Reasoning Models for Real-world Deployment, arXiv:2605.02881, 2026.
2. {{1–3 additional only if cited in text — e.g. Diffusion Policy, RT-1, π₀}}
