"""Intra-cycle encoders: one cycle's curve -> one token.

The default `flatten` encoder is 32-45% of the trajectory model's parameters and
throws away all structure along the SOC axis -- it reshapes 300x4 points into one
1200-vector and applies a single Linear, so the model sees an unordered bag of
numbers and has to relearn that neighbouring points are related.

These variants keep the axis. They are drop-in: same [B, C, L, F] -> [B, C, D]
contract as CyclePatchEmbed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .baselines import CyclePatchEmbed

EMBEDS = ("flatten", "patch", "patchattn", "conv", "delta")


class PatchEmbed(nn.Module):
    """Strided conv along the SOC axis, then pool the patches.

    kernel = stride = P, so each patch sees one SOC window and nothing else.
    This is BatteryMFormer's SOCPatchEncoder geometry, reduced to a single token
    per cycle so it can stand in for the flatten encoder.
    """

    def __init__(self, in_ch: int, d_model: int, curve_len: int, patch: int,
                 dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, d_model, kernel_size=patch, stride=patch)
        self.n_patch = curve_len // patch
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model))

    def forward(self, x):  # [B, C, L, F]
        b, c, ln, f = x.shape
        z = self.conv(x.reshape(b * c, ln, f).transpose(1, 2))   # [B*C, D, M]
        z = z.mean(-1)                                            # pool patches
        z = z.reshape(b, c, -1)
        return self.norm(z + self.ff(z))


class PatchAttnEmbed(nn.Module):
    """Same patching, but a learned query attends over the SOC patches.

    Mean-pooling the patches assumes every SOC window matters equally, which is
    the opposite of the premise that degradation is SOC-localised. A query token
    lets the model choose which windows to read.
    """

    def __init__(self, in_ch: int, d_model: int, curve_len: int, patch: int,
                 dropout: float, n_heads: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, d_model, kernel_size=patch, stride=patch)
        self.n_patch = curve_len // patch
        self.pos = nn.Parameter(torch.zeros(1, self.n_patch, d_model))
        self.q = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.q, std=0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        b, c, ln, f = x.shape
        z = self.conv(x.reshape(b * c, ln, f).transpose(1, 2)).transpose(1, 2)
        z = z + self.pos                                          # [B*C, M, D]
        out, _ = self.attn(self.q.expand(z.shape[0], -1, -1), z, z)
        return self.norm(out.squeeze(1)).reshape(b, c, -1)


class ConvEmbed(nn.Module):
    """Small strided CNN over the SOC axis, then global pool.

    Overlapping receptive fields, unlike the non-overlapping patches, so a
    feature straddling a patch boundary is not split.
    """

    def __init__(self, in_ch: int, d_model: int, curve_len: int, dropout: float):
        super().__init__()
        h = max(16, d_model // 2)
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, h, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(h, h, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(h, d_model, 5, stride=2, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1))
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        b, c, ln, f = x.shape
        z = self.net(x.reshape(b * c, ln, f).transpose(1, 2)).squeeze(-1)
        return self.norm(self.drop(z)).reshape(b, c, -1)


class DeltaPatchEmbed(nn.Module):
    """Patch embedding of the curve AND of its difference from cycle 1.

    The strongest hand-crafted predictor on this benchmark is log var dQ(V) -- a
    statistic of the difference between a late and an early cycle. The flatten
    encoder can only reach that through the downstream blocks; here the first
    cycle is subtracted before patching, so the difference is available directly.
    """

    def __init__(self, in_ch: int, d_model: int, curve_len: int, patch: int,
                 dropout: float):
        super().__init__()
        self.raw = PatchEmbed(in_ch, d_model, curve_len, patch, dropout)
        self.dif = PatchEmbed(in_ch, d_model, curve_len, patch, dropout)
        self.mix = nn.Sequential(nn.LayerNorm(2 * d_model),
                                 nn.Linear(2 * d_model, d_model))

    def forward(self, x):
        d = x - x[:, :1]                       # difference from the first cycle
        return self.mix(torch.cat([self.raw(x), self.dif(d)], dim=-1))


def build_embed(kind: str, in_ch: int, d_model: int, curve_len: int, d_ff: int,
                dropout: float, patch: int = 25, n_heads: int = 4):
    if kind not in EMBEDS:
        raise ValueError(f"embed must be one of {EMBEDS}, got {kind!r}")
    if kind == "flatten":
        return CyclePatchEmbed(curve_len * in_ch, d_model, d_ff, 2, dropout)
    if curve_len % patch:
        raise ValueError(f"curve_len {curve_len} not divisible by patch {patch}")
    if kind == "patch":
        return PatchEmbed(in_ch, d_model, curve_len, patch, dropout)
    if kind == "patchattn":
        return PatchAttnEmbed(in_ch, d_model, curve_len, patch, dropout, n_heads)
    if kind == "conv":
        return ConvEmbed(in_ch, d_model, curve_len, dropout)
    return DeltaPatchEmbed(in_ch, d_model, curve_len, patch, dropout)
