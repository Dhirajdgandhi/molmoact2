#!/usr/bin/env python3
"""Push the local best MolmoAct2 checkpoint to Hugging Face Hub."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import env_setup  # noqa: F401

import config
import checkpoint_utils

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
    # Preserve order but drop duplicates.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description="Push best checkpoint to Hugging Face Hub.")
    parser.add_argument(
        "--repo-id",
        help="Target Hub repo id. Default: unique name from config prefix + step + eval loss + date.",
    )
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=config.HUB_PRIVATE,
        help="Create/update repo as private (default: from config).",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        help="Override checkpoints directory containing best/ and best_meta.json.",
    )
    args = parser.parse_args()

    if not checkpoint_utils.verify_hub_auth():
        logger.error("Hugging Face authentication failed. Run `hf auth login` first.")
        return 1

    checkpoints_dir = args.checkpoints_dir
    if checkpoints_dir is None:
        checkpoints_dir = checkpoint_utils.resolve_checkpoints_dir(_candidate_checkpoints_dirs())
    if checkpoints_dir is None:
        logger.error("No complete best checkpoint found. Checked: %s", _candidate_checkpoints_dirs())
        return 1

    best_dir = checkpoints_dir / config.BEST_CHECKPOINT_DIR
    best_eval_loss, best_step = checkpoint_utils.load_best_meta(checkpoints_dir)
    if best_step <= 0:
        logger.error("best_meta.json missing or invalid in %s", checkpoints_dir)
        return 1

    repo_id = args.repo_id
    if repo_id is None:
        if config.HUB_USE_UNIQUE_BEST_REPO:
            repo_id = checkpoint_utils.generate_best_hub_repo_id(
                prefix=config.HUB_BEST_REPO_PREFIX,
                best_step=best_step,
                best_eval_loss=best_eval_loss,
            )
        else:
            repo_id = config.HUB_REPO_ID

    logger.info(
        "Pushing best checkpoint from %s (step %d, eval %.4f) to %s",
        best_dir,
        best_step,
        best_eval_loss,
        repo_id,
    )

    try:
        repo_url = checkpoint_utils.push_best_checkpoint_folder(
            best_dir,
            repo_id=repo_id,
            private=args.private,
            best_step=best_step,
            best_eval_loss=best_eval_loss,
            checkpoints_dir=checkpoints_dir,
        )
    except Exception as exc:
        logger.error("Hub push failed: %s: %s", type(exc).__name__, exc)
        return 1

    print(f"Pushed to {repo_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
