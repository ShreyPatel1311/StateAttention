"""Trajectory forecasting: first S cycles -> SOH at every cycle to end of life.

Reuses the cycle-life encoder unchanged. Only the head differs:

    cycle life   CyclePatch -> blocks -> pool -> 1 scalar
    trajectory   CyclePatch -> blocks -> pool -> T_max SOH values

Direct multi-horizon rather than autoregressive, matching BatteryMFormer's
feed-forward TrajectoryDecoder. Autoregressive rollout over 5000 steps would
compound its own errors, and there is no teacher signal past EOL.

The head predicts at `stride` resolution and linearly interpolates back to every
cycle: a Linear(d -> 5000) head would be 480k parameters at d=96, dominating the
model and measuring head capacity rather than the encoder. Degradation
trajectories are smooth, so a coarse grid loses little.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .baselines import _PosEnc, masked_mean
from .embeds import build_embed
from .socview import SOCView
from .fused import FusedBlock, MeanVarPool
from .mambaformer import AttnBlock, SSMBlock


class TrajForecaster(nn.Module):
    """[B, S, L, F] -> [B, t_max] SOH trajectory."""

    def __init__(self, cfg: ModelConfig, max_cycles: int = 100, curve_len: int = 300,
                 in_ch: int = 3, t_max: int = 5000, stride: int = 25,
                 backbone: str = "fused"):
        super().__init__()
        self.t_max, self.stride, self.backbone = t_max, stride, backbone
        self.n_knots = (t_max + stride - 1) // stride
        d = cfg.d_model

        self.embed = build_embed(
            getattr(cfg, "embed_type", "flatten"), in_ch, d, curve_len,
            cfg.d_ff, cfg.dropout, patch=getattr(cfg, "embed_patch", 25),
            n_heads=cfg.n_heads)
        self.pos = _PosEnc(d, max(4096, max_cycles + 1))

        if backbone == "stateattention":
            self.blocks = nn.ModuleList(
                AttnBlock(cfg) if (i + 1) % max(1, cfg.attn_every) == 0 else SSMBlock(cfg)
                for i in range(cfg.n_layers)
            )
            self.pool, pool_dim = None, d      # masked mean
        elif backbone == "fused":
            self.blocks = nn.ModuleList(
                FusedBlock(cfg, cfg.attn_on, bool(getattr(cfg, "delta_values", False)))
                for _ in range(cfg.n_layers)
            )
            self.pool, pool_dim = MeanVarPool(), 2 * d
        else:
            raise ValueError(f"backbone must be 'fused' or 'mambaformer', got {backbone!r}")

        # SOC view: one token per SOC interval, summarising its evolution across
        # cycles. The cycle path pools *over* SOC positions, so on its own it
        # cannot isolate a drift confined to one voltage plateau. This is
        # BatteryMFormer's largest single ablation contributor (+0.478 MAPE on
        # Li-ion when removed) and the trajectory model has never used it.
        self.soc = None
        if getattr(cfg, "soc_view", False):
            patch = getattr(cfg, "soc_patch", 25)
            self.soc = SOCView(in_ch, d, curve_len, patch, cfg.dropout)
            self.soc_proj = nn.Sequential(
                nn.Flatten(1),
                nn.LayerNorm(self.soc.n_intervals * d),
                nn.Linear(self.soc.n_intervals * d, d),
            )
            pool_dim += d

        self.head = nn.Sequential(
            nn.LayerNorm(pool_dim), nn.Linear(pool_dim, d), nn.GELU(),
            nn.Linear(d, self.n_knots),
        )
        # Start the trajectory near SOH = 1.0. With default init the head emits
        # ~0, softplus(0) = 0.69, so the model would begin at SOH ~ 0.31 and
        # spend its early epochs climbing to the right decade instead of
        # learning degradation. softplus(-4) = 0.018 -> SOH ~ 0.982.
        self.head_type = getattr(cfg, "head_type", "free")
        if self.head_type == "rateshape":
            # A separate scalar for log-lifetime. Every other variant tried so
            # far couples "how fast this cell dies" to "what shape the decay
            # has", so a fold whose test cells die 3x sooner than training has
            # to relearn the shape too. Splitting them means only the scalar has
            # to extrapolate; the shape prior transfers unchanged.
            self.life_head = nn.Sequential(
                nn.LayerNorm(pool_dim), nn.Linear(pool_dim, d), nn.GELU(),
                nn.Linear(d, 1))
            nn.init.zeros_(self.life_head[-1].weight)
            nn.init.constant_(self.life_head[-1].bias, math.log(500.0))
        nn.init.zeros_(self.head[-1].weight)
        if self.head_type == "monotone":
            # Decrements are divided by the knot count, so the bias sets TOTAL
            # fade across the horizon rather than fade per knot. Without this the
            # 120 knots compound and the untrained model starts at SOH 0.19.
            # softplus(-4) = 0.0181 total -> ends at 0.982, matching the free head.
            self.mono_scale = 1.0 / self.n_knots
            nn.init.constant_(self.head[-1].bias, -4.0)
        else:
            self.mono_scale = 1.0
            nn.init.constant_(self.head[-1].bias, -4.0)

    def forward(self, x, mask):
        z = self.pos(self.embed(x)) * mask.unsqueeze(-1)
        for blk in self.blocks:
            z = blk(z, mask)
        feats = self.pool(z, mask) if self.pool is not None else masked_mean(z, mask)
        if self.soc is not None:
            feats = torch.cat([feats, self.soc_proj(self.soc(x, mask))], dim=-1)
        knots = self.head(feats)                                  # [B, n_knots]

        if self.head_type == "rateshape":
            # shape(u) over normalised life u in [0,1], monotone by construction
            step = F.softplus(knots) * (1.0 / self.n_knots)
            shape = 1.0 - torch.cumsum(step, dim=-1)          # [B, K]
            life = torch.exp(self.life_head(feats).squeeze(-1)).clamp(10.0, 1e5)
            t = torch.arange(self.t_max, device=knots.device, dtype=knots.dtype)
            u = (t[None, :] / life[:, None]).clamp(0.0, 1.0)   # [B, T]
            # grid_sample expects [-1, 1]; sample the shape curve at u
            grid = torch.stack([u * 2.0 - 1.0, torch.zeros_like(u)], dim=-1)
            out = F.grid_sample(shape[:, None, None, :], grid[:, None],
                                mode="bilinear", align_corners=True)
            return out[:, 0, 0]

        if self.head_type == "monotone":
            # Capacity fade is cumulative: a cell does not recover. The free head
            # can emit any curve, so on a fold whose test cells die three times
            # sooner than anything in training it is free to predict a trajectory
            # that flattens or rises. Emitting non-negative per-knot decrements
            # and integrating makes SOH non-increasing by construction, which is
            # a constraint that holds regardless of how fast the cell dies --
            # the kind of prior that should survive a lifetime shift.
            step = F.softplus(knots) * self.mono_scale
            fade = torch.cumsum(step, dim=-1)
            traj_knots = 1.0 - fade
            return F.interpolate(traj_knots.unsqueeze(1), size=self.t_max,
                                 mode="linear", align_corners=True).squeeze(1)

        traj = F.interpolate(knots.unsqueeze(1), size=self.t_max,
                             mode="linear", align_corners=True).squeeze(1)
        # SOH starts near 1 and decreases; predicting the residual from 1.0 puts
        # the model's job at "how much has it faded", not "what is SOH".
        return 1.0 - F.softplus(traj)


def masked_traj_loss(pred, target, mask):
    """MSE over cycles the cell actually reached."""
    m = mask.float()
    return ((pred - target) ** 2 * m).sum() / m.sum().clamp(min=1.0)


def traj_metrics(pred, target, mask):
    """MAPE (%) and MAE (1e-2) on SOH -- BatteryMFormer's metrics."""
    import numpy as np

    p, t, m = (np.asarray(a).ravel() for a in (pred, target, mask))
    keep = (m > 0) & (np.abs(t) > 1e-6)
    if not keep.any():
        return {"mape_pct": float("nan"), "mae_e2": float("nan")}
    p, t = np.clip(p[keep], 0.0, 1.5), t[keep]
    return {"mape_pct": float(np.mean(np.abs(p - t) / np.abs(t)) * 100),
            "mae_e2": float(np.mean(np.abs(p - t)) * 100)}
