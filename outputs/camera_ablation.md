# Camera ablation — 20260701T010531Z

**Checkpoint:** `dhirajdg/molmoact2-record-test-step3000-eval03562-20260625`

## Camera ablation study

| Metric | baseline | no_top | Δ (vs baseline) |
|---|---|---|---|
| `eval/action_flow_loss` | 0.387528 | 0.445783 | +0.058255 |
| `eval/loss` | 0.387528 | 0.445783 | +0.058255 |
| `open_loop/action_mse` | 0.014069 | 0.013375 | -0.000694 |
| `open_loop/action_rmse` | 0.118614 | 0.115649 | -0.002964 |
| `open_loop/joint_0_mse` | 0.015384 | 0.016577 | +0.001193 |
| `open_loop/joint_1_mse` | 0.001274 | 0.001129 | -0.000145 |
| `open_loop/joint_2_mse` | 0.014865 | 0.011933 | -0.002932 |
| `open_loop/joint_3_mse` | 0.013669 | 0.011697 | -0.001972 |
| `open_loop/joint_4_mse` | 0.039076 | 0.038775 | -0.000301 |
| `open_loop/joint_5_mse` | 0.000148 | 0.000138 | -0.000010 |

### Conditions
- **baseline**: both cameras active (observation.images.top, observation.images.wrist)
- **no_top**: top camera zeroed (`observation.images.top`); wrist active
