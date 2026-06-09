#!/usr/bin/env python3
"""Prepare train/val splits from the source LeRobot dataset.

Splits episodes 90/10, writes local datasets under data/, and recomputes
normalization statistics from the train split only (required for correct
inference after fine-tuning).
"""

from __future__ import annotations

import env_setup  # noqa: F401  # must run before huggingface imports

import json
import logging
import shutil
import sys
from pathlib import Path

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _marker_path(root: Path) -> Path:
    return root / ".prepared"


def is_prepared() -> bool:
    return _marker_path(config.TRAIN_DATASET_ROOT).exists() and _marker_path(
        config.VAL_DATASET_ROOT
    ).exists()


def _write_marker(root: Path, payload: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with open(_marker_path(root), "w") as f:
        json.dump(payload, f, indent=2)


def prepare(force: bool = False) -> tuple[Path, Path]:
    """Download, split 90/10 by episode, and recompute train stats."""
    if is_prepared() and not force:
        logger.info("Dataset already prepared at %s", config.DATA_DIR)
        return config.TRAIN_DATASET_ROOT, config.VAL_DATASET_ROOT

    if force and config.DATA_DIR.exists():
        logger.info("Removing existing data dir: %s", config.DATA_DIR)
        shutil.rmtree(config.DATA_DIR)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    from lerobot.datasets.dataset_tools import recompute_stats, split_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source_root = config.DATA_DIR / "source"
    if not (source_root / "meta" / "info.json").exists():
        logger.info("Downloading source dataset: %s", config.SOURCE_DATASET_REPO)
        snapshot_download(
            repo_id=config.SOURCE_DATASET_REPO,
            repo_type="dataset",
            local_dir=str(source_root),
        )
    else:
        logger.info("Using cached source dataset at %s", source_root)

    logger.info("Loading source dataset: %s", config.SOURCE_DATASET_REPO)
    source = LeRobotDataset(config.SOURCE_DATASET_REPO, root=str(source_root))
    logger.info(
        "Source: %d episodes, %d frames",
        source.meta.total_episodes,
        source.meta.total_frames,
    )

    splits = {
        "train": config.TRAIN_FRACTION,
        "val": config.VAL_FRACTION,
    }
    logger.info("Splitting dataset: %s", splits)
    split_datasets = split_dataset(
        source,
        splits=splits,
        output_dir=config.DATA_DIR,
    )

    train_ds = split_datasets["train"]
    val_ds = split_datasets["val"]

    logger.info(
        "Train split: %d episodes, %d frames -> %s",
        train_ds.meta.total_episodes,
        train_ds.meta.total_frames,
        train_ds.root,
    )
    logger.info(
        "Val split: %d episodes, %d frames -> %s",
        val_ds.meta.total_episodes,
        val_ds.meta.total_frames,
        val_ds.root,
    )

    logger.info("Recomputing normalization stats from train split only...")
    recompute_stats(train_ds, skip_image_video=True)

    train_meta = {
        "repo_id": config.TRAIN_DATASET_REPO,
        "root": str(train_ds.root),
        "episodes": train_ds.meta.total_episodes,
        "frames": train_ds.meta.total_frames,
        "split": "train",
        "source": config.SOURCE_DATASET_REPO,
    }
    val_meta = {
        "repo_id": config.VAL_DATASET_REPO,
        "root": str(val_ds.root),
        "episodes": val_ds.meta.total_episodes,
        "frames": val_ds.meta.total_frames,
        "split": "val",
        "source": config.SOURCE_DATASET_REPO,
    }
    _write_marker(config.TRAIN_DATASET_ROOT, train_meta)
    _write_marker(config.VAL_DATASET_ROOT, val_meta)

    logger.info("Dataset preparation complete.")
    logger.info("  Train: %s", config.TRAIN_DATASET_ROOT)
    logger.info("  Val:   %s", config.VAL_DATASET_ROOT)
    return config.TRAIN_DATASET_ROOT, config.VAL_DATASET_ROOT


def main() -> None:
    force = "--force" in sys.argv
    prepare(force=force)


if __name__ == "__main__":
    main()
