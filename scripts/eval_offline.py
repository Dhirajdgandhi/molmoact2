#!/usr/bin/env python3
"""Offline evaluation on the val dataset — no robot or simulation required.

Runs two complementary checks on held-out demonstration frames:
1. Teacher-forcing loss (same metric as training eval)
2. Open-loop action MSE (predict actions from images, compare to recorded actions)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import env_setup  # noqa: F401

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
import checkpoint_utils
import prepare_dataset
from train import _make_dataloader, _release_cuda_cache, evaluate

try:
    from lerobot.utils.constants import OBS_STATE
except ImportError:
    OBS_STATE = "observation.state"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Short names accepted by --ablate-camera (maps to LeRobot observation keys).
CAMERA_ALIASES: dict[str, str] = {
    "top": "observation.images.top",
    "wrist": "observation.images.wrist",
}


def _candidate_checkpoints_dirs() -> list[Path]:
    from lerobot.utils.constants import CHECKPOINTS_DIR

    candidates: list[Path] = []
    env_output = os.environ.get("MOLMOACT2_OUTPUT_DIR")
    if env_output:
        candidates.append(Path(env_output) / config.JOB_NAME / CHECKPOINTS_DIR)
    candidates.extend(
        [
            config.PROJECT_ROOT / "outputs" / config.JOB_NAME / CHECKPOINTS_DIR,
            config.OUTPUT_DIR / config.JOB_NAME / CHECKPOINTS_DIR,
        ]
    )
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def resolve_checkpoint_path(explicit: str | None) -> str | Path:
    checkpoint = explicit or os.environ.get("MOLMOACT2_CHECKPOINT")
    if checkpoint:
        return checkpoint

    checkpoints_dir = checkpoint_utils.resolve_checkpoints_dir(_candidate_checkpoints_dirs())
    if checkpoints_dir is not None:
        local = checkpoints_dir / config.BEST_CHECKPOINT_DIR / "pretrained_model"
        if checkpoint_utils.is_best_model_complete(checkpoints_dir / config.BEST_CHECKPOINT_DIR):
            return local

    for checkpoints_dir in _candidate_checkpoints_dirs():
        hub_meta = checkpoint_utils.load_best_hub_push_meta(checkpoints_dir)
        if hub_meta and hub_meta.get("repo_id"):
            return hub_meta["repo_id"]

    latest = checkpoint_utils.find_latest_hub_best_repo(config.HUB_BEST_REPO_PREFIX)
    if latest:
        logger.info("Auto-selected latest Hub best checkpoint: %s", latest)
        return latest

    raise FileNotFoundError(
        "No checkpoint found. Pass --checkpoint, set MOLMOACT2_CHECKPOINT, or push a best checkpoint to Hub."
    )


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def resolve_ablated_cameras(names: list[str]) -> set[str]:
    """Resolve CLI camera names to full observation image keys."""
    resolved: set[str] = set()
    valid = set(CAMERA_ALIASES) | set(CAMERA_ALIASES.values()) | set(config.IMAGE_KEYS)
    for name in names:
        key = CAMERA_ALIASES.get(name, name)
        if key not in valid:
            raise ValueError(
                f"Unknown camera {name!r}. Choose from: {', '.join(sorted(CAMERA_ALIASES))} "
                f"or full keys {config.IMAGE_KEYS}."
            )
        resolved.add(key)
    return resolved


def _apply_ablation(
    batch: dict,
    ablated_cameras: set[str],
    ablate_proprio: bool,
) -> dict:
    """Zero out ablated camera frames and/or proprioceptive state."""
    if not ablated_cameras and not ablate_proprio:
        return batch
    batch = dict(batch)
    for key in ablated_cameras:
        if key in batch and isinstance(batch[key], torch.Tensor):
            batch[key] = torch.zeros_like(batch[key])
    if ablate_proprio and OBS_STATE in batch and isinstance(batch[OBS_STATE], torch.Tensor):
        batch[OBS_STATE] = torch.zeros_like(batch[OBS_STATE])
    return batch


class _AblationLoader:
    """DataLoader wrapper that applies camera and/or proprio ablation per batch."""

    def __init__(
        self,
        loader: DataLoader,
        ablated_cameras: set[str],
        ablate_proprio: bool = False,
    ) -> None:
        self.loader = loader
        self.ablated_cameras = ablated_cameras
        self.ablate_proprio = ablate_proprio

    @property
    def dataset(self):
        return self.loader.dataset

    def __iter__(self):
        for batch in self.loader:
            yield _apply_ablation(batch, self.ablated_cameras, self.ablate_proprio)


# Backward-compatible alias
_AblatedLoader = _AblationLoader


def _prepare_batch(batch: dict, dataset, preprocessor, device: torch.device) -> dict:
    for cam_key in dataset.meta.camera_keys:
        if cam_key in batch and batch[cam_key].dtype == torch.uint8:
            batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
    batch = preprocessor(batch)
    return _to_device(batch, device)


@torch.no_grad()
def evaluate_open_loop_actions(
    policy,
    dataloader: DataLoader,
    preprocessor,
    device: torch.device,
) -> dict[str, float]:
    """Predict action chunks from observations and compare to recorded actions."""
    policy.eval()
    total_sse = 0.0
    total_elements = 0
    per_dim_sse = None

    for batch in tqdm(dataloader, desc="Open-loop action eval", leave=False):
        batch = _prepare_batch(batch, dataloader.dataset, preprocessor, device)
        policy.reset()
        pred = policy.predict_action_chunk(batch, inference_action_mode="continuous")
        gt = batch["action"]
        steps = min(pred.shape[1], gt.shape[1])
        action_dim = pred.shape[-1]
        pred = pred[:, :steps, :action_dim]
        gt = gt[:, :steps, :action_dim]

        time_valid = torch.ones(steps, dtype=torch.bool, device=pred.device)
        if "action_horizon_is_pad" in batch:
            time_valid = ~batch["action_horizon_is_pad"][:, :steps].to(dtype=torch.bool)

        dim_valid = torch.ones(action_dim, dtype=torch.bool, device=pred.device)
        if "action_dim_is_pad" in batch:
            dim_valid = ~batch["action_dim_is_pad"][:, :action_dim].to(dtype=torch.bool)

        valid = time_valid.unsqueeze(-1) & dim_valid.unsqueeze(1)
        sq_err = (pred - gt).pow(2) * valid
        total_sse += sq_err.sum().item()
        total_elements += valid.sum().item()
        if per_dim_sse is None:
            per_dim_sse = sq_err.sum(dim=(0, 1))
        else:
            per_dim_sse += sq_err.sum(dim=(0, 1))

    if total_elements == 0:
        return {}

    metrics = {
        "open_loop/action_mse": total_sse / total_elements,
        "open_loop/action_rmse": (total_sse / total_elements) ** 0.5,
    }
    if per_dim_sse is not None and total_elements > 0:
        counts = total_elements / action_dim
        per_dim_mse = (per_dim_sse / counts).tolist()
        for idx, value in enumerate(per_dim_mse):
            metrics[f"open_loop/joint_{idx}_mse"] = float(value)
    return metrics


def load_policy_and_data(checkpoint: str | Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from train import _build_policy_config

    if not prepare_dataset.is_prepared():
        logger.info("Prepared splits not found — running prepare_dataset first.")
        prepare_dataset.prepare()

    val_dataset = LeRobotDataset(
        config.VAL_DATASET_REPO,
        root=str(config.VAL_DATASET_ROOT),
        video_backend=config.VIDEO_BACKEND,
    )
    policy_cfg = _build_policy_config()
    policy_cfg.pretrained_path = checkpoint
    policy = make_policy(policy_cfg, ds_meta=val_dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg,
        pretrained_path=str(checkpoint),
    )
    return policy, preprocessor, postprocessor, val_dataset


def _metric_keys(results: dict[str, float]) -> list[str]:
    return sorted(key for key in results if not key.startswith("_"))


def _format_ablation_table(
    rows: list[tuple[str, dict[str, float]]],
    metric_keys: list[str] | None = None,
    condition_notes: dict[str, str] | None = None,
) -> str:
    """Render a markdown table comparing metrics across ablation conditions."""
    if not rows:
        return ""
    keys = metric_keys or _metric_keys(rows[0][1])
    names = [name for name, _ in rows]
    baseline_metrics = rows[0][1]
    delta_headers = [f"Δ {name}" for name in names[1:]]
    header = "| Metric | " + " | ".join(names + delta_headers) + " |"
    sep = "|" + "|".join(["---"] * (len(names) + len(delta_headers) + 1)) + "|"
    lines = ["## Ablation study", "", header, sep]
    for key in keys:
        values = [metrics.get(key) for _, metrics in rows]
        cells = [f"{value:.6f}" if value is not None else "—" for value in values]
        baseline_val = baseline_metrics.get(key)
        for value in values[1:]:
            if baseline_val is not None and value is not None:
                cells.append(f"{value - baseline_val:+.6f}")
            else:
                cells.append("—")
        lines.append(f"| `{key}` | " + " | ".join(cells) + " |")
    lines.append("")
    if condition_notes:
        lines.append("### Conditions")
        for label, note in condition_notes.items():
            lines.append(f"- **{label}**: {note}")
    return "\n".join(lines)


def _wrap_loader(
    loader: DataLoader,
    ablated_cameras: set[str],
    ablate_proprio: bool,
) -> DataLoader:
    if ablated_cameras or ablate_proprio:
        return _AblationLoader(loader, ablated_cameras, ablate_proprio)
    return loader


def _log_condition(
    label: str,
    ablated_cameras: set[str],
    ablate_proprio: bool,
) -> None:
    parts: list[str] = []
    if ablated_cameras:
        parts.append(f"zeroed camera(s): {', '.join(sorted(ablated_cameras))}")
    if ablate_proprio:
        parts.append(f"zeroed proprio (`{OBS_STATE}`)")
    if parts:
        logger.info("Condition %r: %s", label, "; ".join(parts))
    else:
        logger.info(
            "Condition %r: all cameras active (%s); proprio active",
            label,
            ", ".join(config.IMAGE_KEYS),
        )


def _run_eval_pass(
    *,
    policy,
    val_dataset,
    batch_size: int,
    max_batches: int,
    preprocessor,
    accelerator: Accelerator,
    device: torch.device,
    skip_open_loop: bool,
    ablated_cameras: set[str],
    ablate_proprio: bool,
    condition_label: str,
) -> dict[str, float]:
    val_loader = _make_dataloader(
        val_dataset,
        batch_size,
        shuffle=False,
        num_workers=config.EVAL_NUM_WORKERS,
    )
    if max_batches > 0:
        val_loader = _LimitedLoader(val_loader, max_batches)
    val_loader = _wrap_loader(val_loader, ablated_cameras, ablate_proprio)
    _log_condition(condition_label, ablated_cameras, ablate_proprio)

    loss_metrics = evaluate(policy, val_loader, preprocessor, accelerator)
    _release_cuda_cache()
    logger.info("[%s] Teacher-forcing metrics:", condition_label)
    for key, value in sorted(loss_metrics.items()):
        logger.info("  %s = %.6f", key, value)

    results = dict(loss_metrics)
    if not skip_open_loop:
        open_loop_loader = _make_dataloader(
            val_dataset,
            batch_size,
            shuffle=False,
            num_workers=config.EVAL_NUM_WORKERS,
        )
        if max_batches > 0:
            open_loop_loader = _LimitedLoader(open_loop_loader, max_batches)
        open_loop_loader = _wrap_loader(open_loop_loader, ablated_cameras, ablate_proprio)
        action_metrics = evaluate_open_loop_actions(
            policy,
            open_loop_loader,
            preprocessor,
            device,
        )
        logger.info("[%s] Open-loop action metrics:", condition_label)
        for key, value in sorted(action_metrics.items()):
            logger.info("  %s = %.6f", key, value)
        results.update(action_metrics)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline val evaluation (no robot).")
    parser.add_argument(
        "--checkpoint",
        help=(
            "Local pretrained_model path or Hub repo id. "
            "Default: MOLMOACT2_CHECKPOINT env, local best, last Hub push, or latest Hub best."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Use 1 to minimize VRAM during eval.")
    parser.add_argument(
        "--skip-open-loop",
        action="store_true",
        help="Only run teacher-forcing eval loss (faster).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Limit batches for a quick smoke test (0 = full val set).",
    )
    parser.add_argument(
        "--ablate-camera",
        action="append",
        default=[],
        metavar="CAM",
        help=(
            "Zero out a camera stream at eval time (repeatable). "
            "Use short names: top, wrist. Example: --ablate-camera top"
        ),
    )
    parser.add_argument(
        "--no-proprio",
        action="store_true",
        help=f"Zero out proprioceptive state (`{OBS_STATE}`) at eval time.",
    )
    parser.add_argument(
        "--ablation-study",
        action="store_true",
        help=(
            "Run baseline, no-top-camera, and no-proprio ablations; "
            "write comparison table to outputs/ablation_study.md."
        ),
    )
    args = parser.parse_args()

    try:
        checkpoint = resolve_checkpoint_path(args.checkpoint)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Checkpoint: %s", checkpoint)
    policy, preprocessor, _postprocessor, val_dataset = load_policy_and_data(checkpoint)

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    policy.to(device)

    logger.info(
        "Val set: %d episodes, %d frames",
        val_dataset.num_episodes,
        val_dataset.num_frames,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outputs_dir = config.PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", str(checkpoint).replace("/", "_"))

    if args.ablation_study:
        conditions: list[tuple[str, set[str], bool]] = [
            ("baseline", set(), False),
            ("no_top", {CAMERA_ALIASES["top"]}, False),
            ("no_proprio", set(), True),
        ]
        study_rows: list[tuple[str, dict[str, float]]] = []
        study_payload: dict[str, object] = {
            "checkpoint": str(checkpoint),
            "timestamp": timestamp,
            "val_episodes": val_dataset.num_episodes,
            "val_frames": val_dataset.num_frames,
            "conditions": {},
        }
        for label, ablated_cameras, ablate_proprio in conditions:
            metrics = _run_eval_pass(
                policy=policy,
                val_dataset=val_dataset,
                batch_size=args.batch_size,
                max_batches=args.max_batches,
                preprocessor=preprocessor,
                accelerator=accelerator,
                device=device,
                skip_open_loop=args.skip_open_loop,
                ablated_cameras=ablated_cameras,
                ablate_proprio=ablate_proprio,
                condition_label=label,
            )
            study_rows.append((label, metrics))
            study_payload["conditions"][label] = {
                "ablated_cameras": sorted(ablated_cameras),
                "ablate_proprio": ablate_proprio,
                "active_cameras": [k for k in config.IMAGE_KEYS if k not in ablated_cameras],
                "metrics": metrics,
            }

        metric_keys = _metric_keys(study_rows[0][1])
        condition_notes = {
            "baseline": f"both cameras + proprio (`{OBS_STATE}`)",
            "no_top": f"top camera zeroed (`{CAMERA_ALIASES['top']}`); wrist + proprio active",
            "no_proprio": f"proprio zeroed (`{OBS_STATE}`); both cameras active",
        }
        table_md = _format_ablation_table(study_rows, metric_keys, condition_notes)
        print("\n" + table_md)

        ablation_md_path = outputs_dir / "ablation_study.md"
        ablation_md_path.write_text(
            f"# Ablation study — {timestamp}\n\n"
            f"**Checkpoint:** `{checkpoint}`\n\n"
            f"{table_md}\n"
        )
        ablation_json_path = outputs_dir / "ablation_study.json"
        ablation_json_path.write_text(json.dumps(study_payload, indent=2) + "\n")
        # Legacy paths for camera-only consumers
        (outputs_dir / "camera_ablation.md").write_text(ablation_md_path.read_text())
        (outputs_dir / "camera_ablation.json").write_text(ablation_json_path.read_text())

        archive_path = outputs_dir / "eval_runs" / f"{slug}_ablation_{timestamp}.json"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_text(json.dumps(study_payload, indent=2) + "\n")

        logger.info("Wrote %s", ablation_md_path)
        logger.info("Wrote %s", ablation_json_path)
        logger.info("Archived %s", archive_path)
        return 0

    ablated_cameras = resolve_ablated_cameras(args.ablate_camera)
    ablate_proprio = args.no_proprio
    if ablated_cameras or ablate_proprio:
        condition_label = "ablated"
    else:
        condition_label = "baseline"
    results = _run_eval_pass(
        policy=policy,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        preprocessor=preprocessor,
        accelerator=accelerator,
        device=device,
        skip_open_loop=args.skip_open_loop,
        ablated_cameras=ablated_cameras,
        ablate_proprio=ablate_proprio,
        condition_label=condition_label,
    )

    results["_meta"] = {
        "checkpoint": str(checkpoint),
        "timestamp": timestamp,
        "val_episodes": val_dataset.num_episodes,
        "val_frames": val_dataset.num_frames,
        "ablated_cameras": sorted(ablated_cameras),
        "ablate_proprio": ablate_proprio,
        "active_cameras": [k for k in config.IMAGE_KEYS if k not in ablated_cameras],
    }

    archive_path = outputs_dir / "eval_runs" / f"{slug}_{timestamp}.json"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_text(json.dumps(results, indent=2) + "\n")

    out_path = outputs_dir / "offline_eval.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    logger.info("Wrote %s", out_path)
    logger.info("Archived %s", archive_path)
    return 0


class _LimitedLoader:
    """Wrap a DataLoader and stop after N batches."""

    def __init__(self, loader: DataLoader, max_batches: int) -> None:
        self.loader = loader
        self.max_batches = max_batches

    @property
    def dataset(self):
        return self.loader.dataset

    def __iter__(self):
        for idx, batch in enumerate(self.loader):
            if idx >= self.max_batches:
                break
            yield batch


if __name__ == "__main__":
    sys.exit(main())
