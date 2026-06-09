"""Environment setup shared by training scripts."""

from __future__ import annotations

import os
from pathlib import Path

# MolmoAct2 checkpoints are ~22 GB. Use /tmp when home disk is tight.
HF_CACHE = Path(os.environ.get("HF_HOME", "/tmp/huggingface"))
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE / "hub")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
