#!/usr/bin/env python3
"""Fine-tune MolmoAct2 on the prepared train split with eval loss logging.

Runs prepare_dataset automatically if splits are missing.
Logs train and eval losses to WandB and stdout.
"""

from __future__ import annotations

import env_setup  # noqa: F401  # must run before huggingface imports

import logging
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
import prepare_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _build_policy_config():
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

    return MolmoAct2Config(
        checkpoint_path=config.CHECKPOINT_PATH,
        norm_tag=config.NORM_TAG,
        setup_type=config.SETUP_TYPE,
        control_mode=config.CONTROL_MODE,
        image_keys=config.IMAGE_KEYS,
        action_mode=config.ACTION_MODE,
        chunk_size=config.CHUNK_SIZE,
        n_action_steps=config.N_ACTION_STEPS,
        num_flow_timesteps=config.NUM_FLOW_TIMESTEPS,
        model_dtype=config.MODEL_DTYPE,
        gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        enable_lora_vlm=config.ENABLE_LORA_VLM,
        train_action_expert_only=config.TRAIN_ACTION_EXPERT_ONLY,
        freeze_embedding=True,
        normalize_gripper=config.NORMALIZE_GRIPPER,
        optimizer_lr=config.OPTIMIZER_LR,
        optimizer_vit_lr=config.OPTIMIZER_VIT_LR,
        optimizer_connector_lr=config.OPTIMIZER_CONNECTOR_LR,
        optimizer_action_expert_lr=config.OPTIMIZER_ACTION_EXPERT_LR,
        scheduler_warmup_steps=config.SCHEDULER_WARMUP_STEPS,
        push_to_hub=False,
        device="cuda",
    )


def _make_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    *,
    num_workers: int | None = None,
):
    from lerobot.datasets import EpisodeAwareSampler
    from lerobot.utils.collate import lerobot_collate_fn

    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None

    sampler = None
    shuffle_loader = shuffle
    if shuffle and hasattr(dataset, "meta") and dataset.meta.episodes is not None:
        policy_cfg = _build_policy_config()
        drop_n = max(0, policy_cfg.chunk_size - 1)
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=list(range(dataset.meta.total_episodes)),
            drop_n_last_frames=drop_n,
            shuffle=True,
        )
        shuffle_loader = False

    workers = config.NUM_WORKERS if num_workers is None else num_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_loader,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def evaluate(
    policy,
    dataloader: DataLoader,
    preprocessor,
    accelerator: Accelerator,
) -> dict[str, float]:
    """Compute average eval losses over the validation dataloader."""
    policy.eval()
    totals: dict[str, float] = {}
    count = 0

    for batch in dataloader:
        for cam_key in dataloader.dataset.meta.camera_keys:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
        batch = preprocessor(batch)
        with accelerator.autocast():
            _, metrics = policy.forward(batch)
        count += 1
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)

    policy.train()
    if count == 0:
        return {}
    return {f"eval/{k}": v / count for k, v in totals.items()}


def train() -> Path:
    from lerobot.common.wandb_utils import WandBLogger
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.transforms import ImageTransformsConfig
    from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
    from lerobot.utils.random_utils import set_seed
    from lerobot.utils.utils import cycle, init_logging

    init_logging()

    if not prepare_dataset.is_prepared():
        logger.info("Prepared splits not found — running prepare_dataset first.")
        prepare_dataset.prepare()

    set_seed(config.SEED)

    output_dir = config.OUTPUT_DIR / config.JOB_NAME
    if output_dir.exists() and not any(output_dir.iterdir()):
        output_dir.rmdir()

    policy_cfg = _build_policy_config()
    policy_cfg.apply_norm_tag_metadata()

    train_cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=config.TRAIN_DATASET_REPO,
            root=str(config.TRAIN_DATASET_ROOT),
            video_backend=config.VIDEO_BACKEND,
            image_transforms=ImageTransformsConfig(enable=True),
        ),
        policy=policy_cfg,
        output_dir=output_dir,
        job_name=config.JOB_NAME,
        seed=config.SEED,
        num_workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        steps=config.STEPS,
        log_freq=config.LOG_FREQ,
        eval_freq=-1,
        save_checkpoint=True,
        save_freq=config.SAVE_FREQ,
        wandb=WandBConfig(
            enable=config.WANDB_ENABLE,
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
        ),
    )
    train_cfg.validate()
    train_cfg.optimizer = policy_cfg.get_optimizer_preset()
    train_cfg.scheduler = policy_cfg.get_scheduler_preset()

    accelerator = Accelerator(mixed_precision="bf16")
    is_main = accelerator.is_main_process

    wandb_logger = None
    if train_cfg.wandb.enable and is_main:
        wandb_logger = WandBLogger(train_cfg)

    train_dataset = make_dataset(train_cfg)
    val_dataset = LeRobotDataset(
        config.VAL_DATASET_REPO,
        root=str(config.VAL_DATASET_ROOT),
        video_backend=config.VIDEO_BACKEND,
    )

    policy = make_policy(policy_cfg, ds_meta=train_dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg,
        dataset_stats=train_dataset.meta.stats,
    )
    optimizer, lr_scheduler = make_optimizer_and_scheduler(train_cfg, policy)

    train_loader = _make_dataloader(train_dataset, config.BATCH_SIZE, shuffle=True)
    val_loader = _make_dataloader(
        val_dataset,
        config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.EVAL_NUM_WORKERS,
    )

    policy, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        policy, optimizer, train_loader, lr_scheduler
    )

    policy.train()
    dl_iter = cycle(train_loader)

    train_metrics = {
        "loss": AverageMeter("loss", ":.4f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
    }
    tracker = MetricsTracker(
        config.BATCH_SIZE,
        train_dataset.num_frames,
        train_dataset.num_episodes,
        train_metrics,
        accelerator=accelerator,
    )

    best_eval_loss = float("inf")
    best_step = 0

    if is_main:
        logger.info("Output dir: %s", output_dir)
        logger.info(
            "Train: %d episodes, %d frames | Val: %d episodes, %d frames",
            train_dataset.num_episodes,
            train_dataset.num_frames,
            val_dataset.num_episodes,
            val_dataset.num_frames,
        )
        logger.info("Steps: %d | Batch size: %d | Eval every: %d", config.STEPS, config.BATCH_SIZE, config.EVAL_EVERY)
        pbar = tqdm(total=config.STEPS, desc="Training", unit="step")

    for step in range(config.STEPS):
        batch = next(dl_iter)
        for cam_key in train_dataset.meta.camera_keys:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
        batch = preprocessor(batch)

        policy.train()
        with accelerator.autocast():
            loss, output_dict = policy.forward(batch)

        accelerator.backward(loss)
        if policy_cfg.optimizer_grad_clip_norm > 0:
            grad_norm = accelerator.clip_grad_norm_(
                policy.parameters(), policy_cfg.optimizer_grad_clip_norm
            )
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float("inf"))

        optimizer.step()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()

        tracker.loss = loss.item()
        tracker.grad_norm = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
        tracker.lr = optimizer.param_groups[0]["lr"]
        tracker.step()

        is_log = (step + 1) % config.LOG_FREQ == 0
        is_eval = (step + 1) % config.EVAL_EVERY == 0
        is_save = (step + 1) % config.SAVE_FREQ == 0

        if is_main and is_log:
            log_dict = tracker.to_dict()
            if output_dict:
                log_dict.update({f"train/{k}": v for k, v in output_dict.items()})
            if wandb_logger:
                wandb_logger.log_dict(log_dict, step + 1)
            pbar.set_postfix(loss=f"{tracker.loss.avg:.4f}", lr=f"{tracker.lr.avg:.2e}")

        if is_eval:
            accelerator.wait_for_everyone()
            if is_main:
                eval_metrics = evaluate(
                    accelerator.unwrap_model(policy),
                    val_loader,
                    preprocessor,
                    accelerator,
                )
                eval_loss = eval_metrics.get("eval/loss", float("nan"))
                logger.info(
                    "Step %d | eval loss=%.4f flow=%.4f discrete=%.4f",
                    step + 1,
                    eval_loss,
                    eval_metrics.get("eval/action_flow_loss", float("nan")),
                    eval_metrics.get("eval/discrete_ce_loss", float("nan")),
                )
                if wandb_logger:
                    wandb_logger.log_dict(eval_metrics, step + 1, mode="eval")
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_step = step + 1
                    logger.info("New best eval loss %.4f at step %d", best_eval_loss, best_step)

        if is_save and is_main:
            from lerobot.common.train_utils import get_step_checkpoint_dir, save_checkpoint

            ckpt_dir = get_step_checkpoint_dir(output_dir, config.STEPS, step + 1)
            save_checkpoint(
                ckpt_dir,
                step + 1,
                train_cfg,
                accelerator.unwrap_model(policy),
                optimizer,
                lr_scheduler,
                preprocessor,
                postprocessor,
            )
            logger.info("Saved checkpoint: %s", ckpt_dir)

        if is_main:
            pbar.update(1)

    if is_main:
        pbar.close()
        logger.info("Training complete. Best eval loss: %.4f at step %d", best_eval_loss, best_step)
        logger.info("Checkpoints and logs: %s", output_dir)
        if wandb_logger:
            wandb_logger.log_dict(
                {"best_eval_loss": best_eval_loss, "best_step": best_step},
                config.STEPS,
            )

    accelerator.end_training()
    return output_dir


def main() -> None:
    try:
        train()
    except KeyboardInterrupt:
        logger.info("Training interrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
