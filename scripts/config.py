"""Training configuration for MolmoAct2 fine-tuning on record-test dataset."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SOURCE_DATASET_REPO = "dhirajdg/record-test_20260607_235017"
TRAIN_DATASET_REPO = "dhirajdg/record-test_20260607_235017_train"
VAL_DATASET_REPO = "dhirajdg/record-test_20260607_235017_val"

TRAIN_DATASET_ROOT = DATA_DIR / "train"
VAL_DATASET_ROOT = DATA_DIR / "val"

# ---------------------------------------------------------------------------
# Dataset split
# ---------------------------------------------------------------------------
TRAIN_FRACTION = 0.9
VAL_FRACTION = 0.1
# 10 episodes -> 9 train, 1 val (episode 9 has 400 frames)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = "allenai/MolmoAct2-SO100_101"
NORM_TAG = "so100_so101_molmoact2"

SETUP_TYPE = "single so100/so101 robotic arm in molmoact2"
CONTROL_MODE = "absolute joint pose"
IMAGE_KEYS = [
    "observation.images.top",
    "observation.images.wrist",
]

# ---------------------------------------------------------------------------
# Training hyperparameters (tuned for L4 23GB + small dataset)
# ---------------------------------------------------------------------------
JOB_NAME = "molmoact2-record-test"
SEED = 42

STEPS = 3_000
BATCH_SIZE = 2  # L4 23GB: bs=8 OOMs with LoRA MolmoAct2
NUM_WORKERS = 2
VIDEO_BACKEND = "pyav"  # avoid torchcodec/ffmpeg issues in DataLoader workers
EVAL_NUM_WORKERS = 0
LOG_FREQ = 20
EVAL_EVERY = 100
SAVE_FREQ = 500

# Checkpoint retention (~22 GB each). Keeps 1 periodic + 1 best on disk.
KEEP_LAST_N_CHECKPOINTS = 1
SAVE_BEST_CHECKPOINT = True
BEST_CHECKPOINT_DIR = "best"

# Hugging Face Hub (requires `hf auth login`). Mirrors local prune+best strategy:
# - best eval → repo root (deployable model)
# - latest periodic → resume/ (full checkpoint for cross-machine resume)
PUSH_TO_HUB = True
HUB_REPO_ID = "dhirajdg/molmoact2-record-test"
HUB_PRIVATE = True

# Memory-friendly preset for L4 23GB. Action-expert-only uses less VRAM than LoRA+both.
ENABLE_LORA_VLM = False
TRAIN_ACTION_EXPERT_ONLY = True
GRADIENT_CHECKPOINTING = True
NORMALIZE_GRIPPER = True  # required when gripper.pos is not in [-1, 1]
ACTION_MODE = "continuous"  # required when train_action_expert_only=True
CHUNK_SIZE = 30
N_ACTION_STEPS = 30
NUM_FLOW_TIMESTEPS = 8
MODEL_DTYPE = "bfloat16"

# Learning rates (LoRA VLM preset from MolmoAct2 docs)
OPTIMIZER_LR = 5e-5
OPTIMIZER_VIT_LR = 5e-5
OPTIMIZER_CONNECTOR_LR = 5e-5
OPTIMIZER_ACTION_EXPERT_LR = 5e-5
SCHEDULER_WARMUP_STEPS = 100

# ---------------------------------------------------------------------------
# WandB
# ---------------------------------------------------------------------------
WANDB_ENABLE = True
WANDB_PROJECT = "molmoact2-record-test"
WANDB_ENTITY = None  # set to your wandb username/team if needed
