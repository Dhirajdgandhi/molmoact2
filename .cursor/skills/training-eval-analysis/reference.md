# Reference: MolmoAct2 & fine-tuning interpretation

Read this when writing the analysis report. Keep citations in the saved report brief.

## Primary papers & resources

### MolmoAct2 (current base model family)

- **Paper**: Fang et al., *MolmoAct2: Action Reasoning Models for Real-world Deployment*, arXiv:2605.02881, 2026.  
  https://arxiv.org/abs/2605.02881
- **Key ideas**: Molmo2-ER VLM backbone; flow-matching **continuous action expert** with per-layer KV conditioning; OpenFAST tokenizer; SO-100/101 among released embodiments.
- **Relevance to this project**: Base checkpoint `allenai/MolmoAct2-SO100_101` is embodiment-specific; fine-tuning adapts the action expert (and optionally VLM via LoRA) to your teleop distribution.

### MolmoAct (predecessor ARM)

- **Blog / ARM framing**: Action Reasoning Models reason about 3D structure before acting (depth / spatial grounding).  
  https://allenai.org/blog/molmoact2
- **Relevance**: Explains why multi-camera + spatial tasks benefit from wrist + top views; offline joint errors on “reach” DOFs often reflect viewpoint / occlusion, not just capacity.

### Related VLAs (comparison context only)

| Work | Note for advisor text |
|------|----------------------|
| **π₀ / π₀.₅** (Physical Intelligence) | Flow/diffusion-style actions; strong baselines in MolmoAct2 eval tables |
| **OpenVLA / OpenVLA-OFT** | Open VLA line; MolmoAct2 reports higher real-world fine-tune scores on partner benchmark |
| **RT-1 / RT-2** | Classic autoregressive discrete actions; MolmoAct2 uses continuous flow expert instead |
| **Diffusion Policy** | Flow matching in MolmoAct2 is related but coupled to VLM via KV bridge |

Do not over-claim comparisons — this project is **small-data SO-100 fine-tune**, not full MolmoAct2 pretraining.

## Architecture ↔ logged metrics

| Config | Role |
|--------|------|
| `TRAIN_ACTION_EXPERT_ONLY=True` | Only action expert trains; VLM frozen — faster, less risk of destroying pretrained vision on tiny data |
| `ACTION_MODE=continuous` | `action_flow_loss` / flow matching MSE |
| `NUM_FLOW_TIMESTEPS=8` | Denoising steps for flow head |
| `CHUNK_SIZE` / `N_ACTION_STEPS=30` | Predict 30-step action chunks |
| `NORMALIZE_GRIPPER=True` | Gripper in stats; joint 5 often low MSE if nearly constant |

## Small-data fine-tuning heuristics (10 demos)

From project PLAN + general IL/VLA practice:

1. **~10 episodes** → high variance in eval loss; treat ±0.02–0.05 eval loss swings cautiously.
2. **Episode-held-out val** (1/10 episodes) → eval loss is noisy; one episode dominates.
3. **Teacher-forcing eval** can look better than **open-loop** — report both; open-loop is closer to deployment.
4. **Best checkpoint ≠ last step** — MolmoAct2 paper-scale training uses early stopping; same applies here.
5. **Physical robot** remains gold standard (brush task contact, return-to-base).

## Healthy vs concerning patterns

| Pattern | Interpretation |
|---------|----------------|
| Train & eval both fall, eval tracks train | Learning signal present |
| Train falls, eval flat then rises | Overfitting — stop earlier, more data, or LoRA with care |
| Eval flat from step 0 | LR, normalization, or split bug |
| Open-loop RMSE ↓ but eval loss ↑ | Possible exposure bias; still may help on robot |
| Joint 4 MSE ≫ others | Reach joint; check wrist camera framing |
| Offline eval > training best eval | Different code path OK; large gap warrants checkpoint mismatch check |

## Advisor conclusion rubric

End every report with a clear **Recommendation** (one of):

- **Proceed to robot eval** — offline metrics improved or stable; no overfit signature
- **Iterate training** — underfit or unstable; suggest concrete hyperparameter/data change
- **Collect more demonstrations** — variance too high or eval regressed; data is the bottleneck
- **Hold deployment** — overfitting or offline regression; do not deploy current checkpoint

Always qualify: *offline metrics are necessary but not sufficient for contact-rich manipulation.*
