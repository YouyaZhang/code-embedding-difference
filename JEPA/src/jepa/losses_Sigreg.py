# losses_r2.py
from __future__ import annotations
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import lejepa


class CosineAlignLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(self, z_pred: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
        z_pred = F.normalize(z_pred, dim=-1, eps=self.eps)
        z_tgt = F.normalize(z_tgt, dim=-1, eps=self.eps)
        return 1.0 - (z_pred * z_tgt).sum(dim=-1).mean()


class VarianceLoss(nn.Module):
    def __init__(self, target_std: float = 1.0, eps: float = 1e-4):
        super().__init__()
        self.target_std = float(target_std)
        self.eps = float(eps)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(z.var(dim=0, unbiased=False) + self.eps)
        return F.relu(self.target_std - std).mean()


class SIGRegLoss(nn.Module):
    def __init__(self, num_points: int = 17, num_slices: int = 256):
        super().__init__()
        univariate_test = lejepa.univariate.EppsPulley(
            n_points=num_points
        )
        self.loss_fn = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=univariate_test,
            num_slices=num_slices,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(z)


class EmaPredictiveLossRoute2(nn.Module):
    """
    Route 2:
      L = w_align * cosine_align(z_pred, z_tgt)
        + w_var   * (var(z_ctx) + var(z_tgt))
        + w_mse   * mse(z_pred, z_tgt)
        + w_sig   * sigreg(...)
    """

    def __init__(
        self,
        w_align: float = 1.0,
        w_var: float = 1.0,
        w_mse: float = 0.0,
        w_sig: float = 0.0,
        sig_num_points: int = 17,
        sig_num_slices: int = 256,
        sig_on: str = "pred",   # pred | ctx | tgt | pred_tgt | ctx_pred
        align_eps: float = 1e-8,
        var_target_std: float = 1.0,
        var_eps: float = 1e-4,
    ):
        super().__init__()
        self.w_align = float(w_align)
        self.w_var = float(w_var)
        self.w_mse = float(w_mse)
        self.w_sig = float(w_sig)
        self.sig_on = str(sig_on)

        self.align = CosineAlignLoss(eps=align_eps)
        self.var = VarianceLoss(target_std=var_target_std, eps=var_eps)
        self.mse = nn.MSELoss()

        self.sig = None
        if self.w_sig > 0:
            self.sig = SIGRegLoss(
                num_points=sig_num_points,
                num_slices=sig_num_slices,
            )

    def forward(self, z_ctx, z_pred, z_tgt):
        l_align = self.align(z_pred, z_tgt)
        l_var = self.var(z_ctx) + self.var(z_tgt)
        l_mse = self.mse(z_pred, z_tgt)

        l_sig = z_pred.new_zeros(())
        if self.sig is not None:
            if self.sig_on == "pred":
                l_sig = self.sig(z_pred)
            elif self.sig_on == "ctx":
                l_sig = self.sig(z_ctx)
            elif self.sig_on == "tgt":
                l_sig = self.sig(z_tgt)
            elif self.sig_on == "pred_tgt":
                l_sig = 0.5 * (self.sig(z_pred) + self.sig(z_tgt))
            elif self.sig_on == "ctx_pred":
                l_sig = 0.5 * (self.sig(z_ctx) + self.sig(z_pred))
            else:
                raise ValueError(f"Unknown sig_on: {self.sig_on}")

        loss = (
            self.w_align * l_align
            + self.w_var * l_var
            + self.w_mse * l_mse
            + self.w_sig * l_sig
        )

        return {
            "loss": loss,
            "align": l_align,
            "var": l_var,
            "mse": l_mse,
            "sig": l_sig,
        }


def build_loss(cfg: Dict[str, Any]) -> EmaPredictiveLossRoute2:
    lcfg = cfg.get("loss", {})
    return EmaPredictiveLossRoute2(
        w_align=float(lcfg.get("w_align", 1.0)),
        w_var=float(lcfg.get("w_var", 1.0)),
        w_mse=float(lcfg.get("w_mse", 0.0)),
        w_sig=float(lcfg.get("w_sig", 0.0)),
        sig_num_points=int(lcfg.get("sig_num_points", 17)),
        sig_num_slices=int(lcfg.get("sig_num_slices", 256)),
        sig_on=str(lcfg.get("sig_on", "pred")),
        align_eps=float(lcfg.get("align_eps", 1e-8)),
        var_target_std=float(lcfg.get("var_target_std", 1.0)),
        var_eps=float(lcfg.get("var_eps", 1e-4)),
    )


@torch.no_grad()
def retrieval_top1_acc(z_pred: torch.Tensor, z_tgt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    z_pred = z_pred.float()
    z_tgt = z_tgt.float()

    z_pred = F.normalize(z_pred, dim=-1, eps=eps)
    z_tgt = F.normalize(z_tgt, dim=-1, eps=eps)

    sim = z_pred @ z_tgt.t()
    pred_idx = sim.argmax(dim=1)
    gt_idx = torch.arange(sim.size(0), device=sim.device)
    return (pred_idx == gt_idx).float().mean()


@torch.no_grad()
def emb_std_mean(z: torch.Tensor) -> torch.Tensor:
    z = z.float()
    return z.std(dim=0, unbiased=False).mean()