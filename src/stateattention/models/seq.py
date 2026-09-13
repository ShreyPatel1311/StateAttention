"""Per-timestep sequence regression, for SOC estimation.

Reuses the cycle-life blocks unchanged -- FusedBlock, SSMBlock and AttnBlock all
have signature forward(z, mask) -> z and are agnostic to what the sequence axis
means. Only the two ends differ:

    cycle life   CyclePatchEmbed([B,C,L,F]) -> blocks -> pool -> ONE scalar
    SOC          Linear([B,T,F])            -> blocks -> per-step head -> [B,T]

Everything is causal: the SSM scan is causal by construction and StateAttention
carries a causal mask, so an estimate at t never sees t+1. That matters here in
a way it did not for cycle life -- an SOC estimator that peeks at the future is
useless on a real vehicle.
"""

from __future__ import annotations

import torch.nn as nn

from ..config import ModelConfig
from .baselines import _PosEnc
from .fused import FusedBlock
from .mambaformer import AttnBlock, SSMBlock


class SeqSOC(nn.Module):
    """[B, T, in_ch] -> [B, T].  backbone: 'fused' (uses attn_on) | 'mambaformer'."""

    def __init__(self, cfg: ModelConfig, in_ch: int = 2, backbone: str = "fused",
                 max_len: int = 4096):
        super().__init__()
        self.backbone = backbone
        d = cfg.d_model
        self.inp = nn.Sequential(nn.Linear(in_ch, d), nn.LayerNorm(d))
        self.pos = _PosEnc(d, max_len)

        if backbone == "stateattention":
            # same interleave as the cycle-life model: attention every k-th layer
            self.blocks = nn.ModuleList(
                AttnBlock(cfg) if (i + 1) % max(1, cfg.attn_every) == 0 else SSMBlock(cfg)
                for i in range(cfg.n_layers)
            )
        elif backbone == "fused":
            self.blocks = nn.ModuleList(
                FusedBlock(cfg, cfg.attn_on, bool(getattr(cfg, "delta_values", False)))
                for _ in range(cfg.n_layers)
            )
        else:
            raise ValueError(f"backbone must be 'fused' or 'mambaformer', got {backbone!r}")

        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1)
        )

    def forward(self, x, mask):
        z = self.pos(self.inp(x)) * mask.unsqueeze(-1)
        for blk in self.blocks:
            z = blk(z, mask)
        return self.head(z).squeeze(-1)  # [B, T], one SOC estimate per step
