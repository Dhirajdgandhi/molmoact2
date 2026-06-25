"""Checkpoint retention, resume discovery, and Hugging Face Hub uploads."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import HfApi
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from lerobot.utils.constants import (
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies import PreTrainedPolicy

logger = logging.getLogger(__name__)

BEST_META_FILE = "best_meta.json"
HUB_RESUME_PREFIX = "resume"


def is_complete_checkpoint(ckpt_dir: Path) -> bool:
    """Return True when the checkpoint finished saving model weights and training state."""
    model_file = ckpt_dir / PRETRAINED_MODEL_DIR / SAFETENSORS_SINGLE_FILE
    training_step_file = ckpt_dir / TRAINING_STATE_DIR / TRAINING_STEP
    return model_file.is_file() and training_step_file.is_file()


def remove_incomplete_checkpoints(checkpoints_dir: Path) -> None:
    """Drop partial checkpoint dirs left by failed saves (e.g. disk full)."""
    if not checkpoints_dir.exists():
        return
    for path in checkpoints_dir.iterdir():
        if not path.is_dir() or path.name in {LAST_CHECKPOINT_LINK, "best"}:
            continue
        if path.name.isdigit() and not is_complete_checkpoint(path):
            logger.warning("Removing incomplete checkpoint: %s", path)
            shutil.rmtree(path)


def list_step_checkpoint_dirs(checkpoints_dir: Path) -> list[tuple[int, Path]]:
    if not checkpoints_dir.exists():
        return []
    step_dirs = [
        (int(path.name), path)
        for path in checkpoints_dir.iterdir()
        if path.is_dir() and path.name.isdigit() and is_complete_checkpoint(path)
    ]
    return sorted(step_dirs)


def find_resume_checkpoint(checkpoints_dir: Path) -> Path | None:
    """Return the latest complete periodic checkpoint for training resume."""
    if not checkpoints_dir.exists():
        return None

    last_link = checkpoints_dir / LAST_CHECKPOINT_LINK
    if last_link.is_symlink():
        target = (checkpoints_dir / last_link.readlink()).resolve()
        if target.is_dir() and is_complete_checkpoint(target):
            return target
    if last_link.is_dir() and is_complete_checkpoint(last_link):
        return last_link

    step_dirs = list_step_checkpoint_dirs(checkpoints_dir)
    if not step_dirs:
        return None
    return step_dirs[-1][1]


def prune_old_checkpoints(
    checkpoints_dir: Path,
    *,
    keep_last_n: int,
    protected: frozenset[str] = frozenset({LAST_CHECKPOINT_LINK, "best"}),
) -> None:
    """Keep only the newest `keep_last_n` numbered checkpoints."""
    step_dirs = list_step_checkpoint_dirs(checkpoints_dir)
    keep_steps = {step for step, _ in step_dirs[-keep_last_n:]}
    for step, path in step_dirs:
        if step in keep_steps:
            continue
        logger.info("Removing old checkpoint: %s", path)
        shutil.rmtree(path)


def load_best_meta(checkpoints_dir: Path) -> tuple[float, int]:
    meta_path = checkpoints_dir / BEST_META_FILE
    if not meta_path.is_file():
        return float("inf"), 0
    data = json.loads(meta_path.read_text())
    return float(data.get("best_eval_loss", float("inf"))), int(data.get("best_step", 0))


def save_best_meta(checkpoints_dir: Path, best_eval_loss: float, best_step: int) -> None:
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    meta_path = checkpoints_dir / BEST_META_FILE
    meta_path.write_text(
        json.dumps({"best_eval_loss": best_eval_loss, "best_step": best_step}, indent=2) + "\n"
    )


def push_best_to_hub(
    policy: PreTrainedPolicy,
    train_cfg: TrainPipelineConfig,
    *,
    repo_id: str,
    private: bool,
) -> None:
    """Upload the current policy weights to the Hub repo root (deployable model)."""
    policy.config.repo_id = repo_id
    policy.config.private = private
    logger.info("Pushing best checkpoint to Hub: %s", repo_id)
    policy.push_model_to_hub(train_cfg)


def push_resume_to_hub(
    ckpt_dir: Path,
    *,
    repo_id: str,
    private: bool,
) -> None:
    """Upload the latest full checkpoint under `resume/` for cross-machine resume."""
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    try:
        api.delete_folder(repo_id=repo_id, path_in_repo=HUB_RESUME_PREFIX, repo_type="model")
    except Exception:
        pass
    logger.info("Pushing resume checkpoint to Hub: %s/%s", repo_id, HUB_RESUME_PREFIX)
    api.upload_folder(
        folder_path=ckpt_dir,
        path_in_repo=HUB_RESUME_PREFIX,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Update resume checkpoint (step {ckpt_dir.name})",
    )
