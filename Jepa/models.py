# models.py
# -*- coding: utf-8 -*-
"""
Models module for JEPA-style training on code pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel


# -------------------------
# Helpers
# -------------------------
def infer_emb_dim(model_name: str) -> int:
    m = AutoModel.from_pretrained(model_name)
    return int(getattr(m.config, "hidden_size"))


def _freeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


def _unfreeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = True


# -------------------------
# Encoder
# -------------------------
class HFMeanPoolEncoder(nn.Module):
    """
    Encoder wrapper:
    - backbone: HF AutoModel
    - output: [B, D] mean-pooled embedding (mask-aware)
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

    @property
    def emb_dim(self) -> int:
        return int(getattr(self.backbone.config, "hidden_size"))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state  # [B, L, D]
        mask = attention_mask.unsqueeze(-1).to(h.dtype)  # [B, L, 1]
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled  # [B, D]


def _apply_lora(backbone: nn.Module, lora_cfg: Dict[str, Any]) -> nn.Module:
    """
    Apply PEFT LoRA to a HF model.
    lora_cfg expected keys (examples):
      enabled: bool
      r: int
      alpha: int
      dropout: float
      target_modules: list[str]
    """
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PEFT (peft) is required for LoRA mode but is not available in this environment. "
            "Install it via `pip install peft`."
        ) from e

    if not lora_cfg.get("target_modules"):
        raise ValueError(
            "LoRA requires `encoder.lora.target_modules` to be a non-empty list. "
            "Please set it in your config."
        )

    bias = lora_cfg.get("bias", "none")
    # For encoder-style models, PEFT task_type is not strictly required for plain LoRA.
    lcfg = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg["target_modules"]),
        bias=bias,
    )
    return get_peft_model(backbone, lcfg)


def build_encoder(cfg: Dict[str, Any], device: torch.device) -> Tuple[Optional[nn.Module], int]:

    enc_cfg = cfg.get("encoder", {})
    model_name = enc_cfg.get("name", None)

    if not model_name:
        emb_dim = enc_cfg.get("emb_dim")
        if emb_dim is None:
            raise ValueError("encoder.name is None but encoder.emb_dim is not set.")
        return None, int(emb_dim)

    encoder = HFMeanPoolEncoder(model_name=model_name).to(device)
    emb_dim = encoder.emb_dim

    train_mode = (enc_cfg.get("train_mode") or "frozen").lower()
    if train_mode not in ("frozen", "full", "lora"):
        raise ValueError(f"Unknown encoder.train_mode: {train_mode}")

    if train_mode == "frozen":
        _freeze_module(encoder)
        encoder.eval()  # usually frozen encoder in eval mode
    elif train_mode == "full":
        _unfreeze_module(encoder)
        encoder.train()
    else:  # lora
        lora_cfg = enc_cfg.get("lora", {})
        if not lora_cfg.get("enabled", True):
            raise ValueError("encoder.train_mode is 'lora' but encoder.lora.enabled is false.")
        # Apply LoRA to the backbone only (keeps wrapper intact)
        encoder.backbone = _apply_lora(encoder.backbone, lora_cfg)
        _unfreeze_module(encoder)  # PEFT will set only LoRA params trainable; others frozen.
        encoder.train()

    return encoder, emb_dim


# -------------------------
# Predictors
# -------------------------
class ViTPredictor1D(nn.Module):
    """
    1D "ViT-like" predictor operating on pooled embeddings [B, D].

    It reshapes embedding into tokens: [B, T, patch], projects to d_model,
    runs TransformerEncoder, projects back to patch and flattens.
    """

    def __init__(
        self,
        dim: int,
        patch: int = 32,
        layers: int = 4,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim % patch != 0:
            raise ValueError(f"emb_dim={dim} must be divisible by patch={patch}")
        self.dim = dim
        self.patch = patch
        self.num_tokens = dim // patch

        self.token_proj = nn.Linear(patch, dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.out = nn.Linear(dim, patch)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, D]
        b, d = z.shape
        x = z.view(b, self.num_tokens, self.patch)  # [B, T, patch]
        x = self.token_proj(x)                      # [B, T, D]
        x = self.encoder(x)                         # [B, T, D]
        x = self.out(x)                             # [B, T, patch]
        return x.reshape(b, d)                      # [B, D]

def build_predictor(cfg: Dict[str, Any], emb_dim: int, device: torch.device) -> nn.Module:
    p_cfg = cfg.get("predictor", {})
    name = (p_cfg.get("name") or "vit1d").lower()

    if name == "vit1d":
        v = p_cfg.get("vit", {})
        m = ViTPredictor1D(
            dim=emb_dim,
            patch=int(v.get("patch", 32)),
            layers=int(v.get("layers", 4)),
            heads=int(v.get("heads", 8)),
            mlp_ratio=float(v.get("mlp_ratio", 4.0)),
            dropout=float(v.get("dropout", 0.0)),
        ).to(device)
        return m

    raise ValueError(f"Unknown predictor.name: {name}")