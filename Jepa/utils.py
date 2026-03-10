# utils.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
import random
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import yaml


# -------------------------
# Config
# -------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

def _cast_value(v: str):
    s = v.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.lower() in ("none", "null"):
        return None
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_cast_value(x.strip()) for x in inner.split(",")]
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s

def set_by_dotted_key(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Bad override '{ov}', expected key=value")
        k, v = ov.split("=", 1)
        set_by_dotted_key(cfg, k.strip(), _cast_value(v.strip()))
    return cfg

def save_resolved_config(cfg: Dict[str, Any], out_dir: str, filename: str = "resolved_config.json") -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


# -------------------------
# DDP helpers
# -------------------------
def is_ddp_env() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ

def rank() -> int:
    return int(os.environ.get("RANK", "0"))

def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))

def is_main() -> bool:
    return rank() == 0

def ddp_setup(backend: str = "nccl") -> None:
    dist.init_process_group(backend=backend)
    torch.cuda.set_device(local_rank())

def ddp_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()

def ddp_all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


# -------------------------
# Reproducibility
# -------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# EMA / model helpers
# -------------------------
@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, tau: float) -> None:
    for p_t, p_s in zip(teacher.parameters(), student.parameters()):
        p_t.data.mul_(tau).add_(p_s.data, alpha=1.0 - tau)

def unwrap_ddp(m: nn.Module) -> nn.Module:
    return m.module if hasattr(m, "module") else m

def set_requires_grad(m: nn.Module, requires_grad: bool) -> None:
    for p in m.parameters():
        p.requires_grad = requires_grad


# -------------------------
# Simple meters
# -------------------------
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += float(val) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)