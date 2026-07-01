#!/usr/bin/env python3
"""Collect training/offline-eval metrics into one JSON snapshot for analysis reports."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import env_setup  # noqa: F401

import checkpoint_utils
import config


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


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _config_snapshot() -> dict:
    keys = [
        "SOURCE_DATASET_REPO",
        "TRAIN_DATASET_REPO",
        "VAL_DATASET_REPO",
        "TRAIN_FRACTION",
        "VAL_FRACTION",
        "CHECKPOINT_PATH",
        "NORM_TAG",
        "SETUP_TYPE",
        "CONTROL_MODE",
        "IMAGE_KEYS",
        "JOB_NAME",
        "SEED",
        "STEPS",
        "BATCH_SIZE",
        "NUM_WORKERS",
        "EVAL_NUM_WORKERS",
        "LOG_FREQ",
        "EVAL_EVERY",
        "SAVE_FREQ",
        "KEEP_LAST_N_CHECKPOINTS",
        "SAVE_BEST_CHECKPOINT",
        "SAVE_BEST_MODEL_ONLY",
        "PUSH_TO_HUB",
        "HUB_REPO_ID",
        "HUB_BEST_REPO_PREFIX",
        "HUB_USE_UNIQUE_BEST_REPO",
        "ENABLE_LORA_VLM",
        "TRAIN_ACTION_EXPERT_ONLY",
        "GRADIENT_CHECKPOINTING",
        "NORMALIZE_GRIPPER",
        "ACTION_MODE",
        "CHUNK_SIZE",
        "N_ACTION_STEPS",
        "NUM_FLOW_TIMESTEPS",
        "MODEL_DTYPE",
        "OPTIMIZER_LR",
        "OPTIMIZER_VIT_LR",
        "OPTIMIZER_CONNECTOR_LR",
        "OPTIMIZER_ACTION_EXPERT_LR",
        "SCHEDULER_WARMUP_STEPS",
        "WANDB_ENABLE",
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "VIDEO_BACKEND",
    ]
    return {key: getattr(config, key) for key in keys}


def _parse_train_log(log_path: Path) -> dict:
    if not log_path.is_file():
        return {}

    text = log_path.read_text(errors="replace")
    eval_points: list[dict] = []
    for match in re.finditer(
        r"Step (\d+) \| eval loss=([\d.]+) flow=([\d.]+)(?: discrete=([\d.]+))?",
        text,
    ):
        point = {
            "step": int(match.group(1)),
            "eval_loss": float(match.group(2)),
            "action_flow_loss": float(match.group(3)),
        }
        if match.group(4) is not None:
            point["discrete_ce_loss"] = float(match.group(4))
        eval_points.append(point)

    best_matches = list(
        re.finditer(r"New best eval loss ([\d.]+) at step (\d+)", text)
    )
    complete = re.search(
        r"Training complete\. Best eval loss: ([\d.]+) at step (\d+)", text
    )

    result: dict = {"eval_curve": eval_points}
    if best_matches:
        last = best_matches[-1]
        result["last_new_best"] = {
            "eval_loss": float(last.group(1)),
            "step": int(last.group(2)),
        }
    if complete:
        result["training_complete"] = {
            "eval_loss": float(complete.group(1)),
            "step": int(complete.group(2)),
        }
    return result


def _list_eval_runs(outputs_dir: Path) -> list[dict]:
    runs_dir = outputs_dir / "eval_runs"
    if not runs_dir.is_dir():
        return []
    runs = sorted(runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    out = []
    for path in runs:
        data = _read_json(path) or {}
        out.append(
            {
                "path": str(path.relative_to(config.PROJECT_ROOT)),
                "mtime": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
                "metrics": {k: v for k, v in data.items() if not k.startswith("_")},
                "meta": data.get("_meta", {}),
            }
        )
    return out


def _list_analyses(outputs_dir: Path) -> list[dict]:
    analyses_dir = outputs_dir / "analyses"
    if not analyses_dir.is_dir():
        return []
    return [
        {
            "path": str(p.relative_to(config.PROJECT_ROOT)),
            "mtime": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(),
        }
        for p in sorted(analyses_dir.glob("*.md"), key=lambda x: x.stat().st_mtime)
    ]


def collect(event: str) -> dict:
    outputs_dir = config.PROJECT_ROOT / "outputs"
    checkpoints_dir = checkpoint_utils.resolve_checkpoints_dir(_candidate_checkpoints_dirs())

    log_candidates = [
        config.OUTPUT_DIR / "train.log",
        config.PROJECT_ROOT / "outputs" / "train.log",
        Path(os.environ.get("MOLMOACT2_OUTPUT_DIR", "")) / "train.log",
    ]
    train_log = next((p for p in log_candidates if p.is_file()), None)

    best_meta = None
    hub_push = None
    if checkpoints_dir:
        best_meta_path = checkpoints_dir / "best_meta.json"
        if best_meta_path.is_file():
            best_meta = _read_json(best_meta_path)
        hub_push = checkpoint_utils.load_best_hub_push_meta(checkpoints_dir)

    offline_eval = _read_json(outputs_dir / "offline_eval.json")
    eval_runs = _list_eval_runs(outputs_dir)
    prior_analyses = _list_analyses(outputs_dir)

    latest_hub = None
    try:
        latest_hub = checkpoint_utils.find_latest_hub_best_repo(config.HUB_BEST_REPO_PREFIX)
    except Exception:
        pass

    snapshot = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "config": _config_snapshot(),
        "dataset_plan": {
            "episodes_total": 10,
            "frames_total": 2299,
            "train_episodes": 9,
            "train_frames": 1899,
            "val_episodes": 1,
            "val_frames": 400,
            "task": "Brush solar panel and return to base",
            "robot": "so_follower",
            "action_dim": 6,
            "cameras": config.IMAGE_KEYS,
        },
        "training": {
            "train_log": str(train_log) if train_log else None,
            "train_log_parsed": _parse_train_log(train_log) if train_log else {},
            "best_meta": best_meta,
            "hub_best_push": hub_push,
            "checkpoints_dir": str(checkpoints_dir) if checkpoints_dir else None,
            "latest_hub_repo": latest_hub,
        },
        "offline_eval": {
            "latest": offline_eval,
            "history": eval_runs,
        },
        "prior_analyses": prior_analyses,
    }
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect metrics snapshot for analysis reports.")
    parser.add_argument(
        "--event",
        choices=("training", "offline_eval", "auto"),
        default="auto",
        help="What triggered collection (default: auto-detect).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Write JSON here (default: stdout).",
    )
    args = parser.parse_args()

    event = args.event
    if event == "auto":
        outputs = config.PROJECT_ROOT / "outputs"
        has_offline = (outputs / "offline_eval.json").is_file()
        has_train = any(
            p.is_file()
            for p in (
                config.OUTPUT_DIR / "train.log",
                outputs / "train.log",
            )
        )
        if has_offline and has_train:
            event = "offline_eval"
        elif has_train:
            event = "training"
        else:
            event = "offline_eval" if has_offline else "training"

    snapshot = collect(event)
    payload = json.dumps(snapshot, indent=2) + "\n"

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
