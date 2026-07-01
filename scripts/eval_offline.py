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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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
    args = parser.parse_args()

    try:
        checkpoint = resolve_checkpoint_path(args.checkpoint)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Checkpoint: %s", checkpoint)
    policy, preprocessor, _postprocessor, val_dataset = load_policy_and_data(checkpoint)

    val_loader = _make_dataloader(
        val_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=config.EVAL_NUM_WORKERS,
    )
    if args.max_batches > 0:
        val_loader = _LimitedLoader(val_loader, args.max_batches)

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    policy.to(device)

    logger.info(
        "Val set: %d episodes, %d frames",
        val_dataset.num_episodes,
        val_dataset.num_frames,
    )

    loss_metrics = evaluate(policy, val_loader, preprocessor, accelerator)
    _release_cuda_cache()
    logger.info("Teacher-forcing metrics:")
    for key, value in sorted(loss_metrics.items()):
        logger.info("  %s = %.6f", key, value)

    results = dict(loss_metrics)
    if not args.skip_open_loop:
        action_metrics = evaluate_open_loop_actions(
            policy,
            val_loader,
            preprocessor,
            device,
        )
        logger.info("Open-loop action metrics:")
        for key, value in sorted(action_metrics.items()):
            logger.info("  %s = %.6f", key, value)
        results.update(action_metrics)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results["_meta"] = {
        "checkpoint": str(checkpoint),
        "timestamp": timestamp,
        "val_episodes": val_dataset.num_episodes,
        "val_frames": val_dataset.num_frames,
    }

    outputs_dir = config.PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", str(checkpoint).replace("/", "_"))
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
