"""SOC view: one token per SOC interval, summarising its evolution across cycles.

Adapted from BatteryMFormer (Tan et al., KDD 2026), whose ablation shows this
is their largest single contributor -- removing it costs +0.478 MAPE on Li-ion,
roughly twice any other component.

The motivation is that degradation signatures are localised to specific state-of-
charge intervals: a particular voltage plateau drifts while the rest of the curve
barely moves. The default path here cannot express that. CyclePatchEmbed
flattens a cycle's 300x3 points into one vector and projects it in a single
Linear, so position within the cycle carries no structure -- the model sees an
unordered bag of 900 numbers.

This module transposes the axes:

    cycle view (default)  one token per CYCLE, pooling over SOC positions
    SOC view   (here)     one token per SOC INTERVAL, pooling over CYCLES

which also matters for a second reason. The strongest hand-crafted predictor on
this benchmark is log var dQ(V) -- the variance, taken along the SOC axis, of a
cycle-to-cycle difference (Spearman -0.75 against log life, measured on 180
cells). MeanVarPool takes its variance along the *cycle* axis, which is a
different quantity. Per-interval statistics make the Severson-style feature
expressible.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SOCView(nn.Module):
    """[B, S, L, C] -> [B, M, d], one token per SOC interval.

    A strided Conv1d patches each cycle's curve along the SOC axis into M
    intervals, so an interval token is built only from points in that SOC
    window. Each interval's trajectory across cycles is then summarised by
    three masked statistics:

        mean   the interval's typical value
        logvar how much it moves across cycles
        delta  late cycles minus early cycles -- the dQ analogue

    `delta` is what makes this more than a pooling layer: it is a learned,
    per-interval generalisation of Q_late(V) - Q_early(V).
    """

    def __init__(self, in_ch: int, d_model: int, curve_len: int = 300,
                 patch: int = 25, dropout: float = 0.0, early_frac: float = 0.25):
        super().__init__()
        if curve_len % patch:
            raise ValueError(f"curve_len {curve_len} not divisible by patch {patch}")
        self.patch = patch
        self.n_intervals = curve_len // patch
        self.early_frac = early_frac
        self.conv = nn.Conv1d(in_ch, d_model, kernel_size=patch, stride=patch)
        self.proj = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, s, ln, c = x.shape
        # patch along the SOC axis, independently per cycle
        z = self.conv(x.reshape(b * s, ln, c).transpose(1, 2))  # [B*S, d, M]
        z = z.reshape(b, s, -1, self.n_intervals).permute(0, 3, 1, 2)  # [B, M, S, d]

        m = mask[:, None, :, None]  # [B, 1, S, 1]
        n = m.sum(2).clamp(min=1.0)
        mu = (z * m).sum(2) / n
        var = ((z - mu.unsqueeze(2)) ** 2 * m).sum(2) / n

        # delta: mean over late observed cycles minus mean over early ones.
        # Built from the mask so it follows the prefix length rather than a
        # fixed cycle index -- a sample with 10 visible cycles and one with 100
        # both get a meaningful early/late contrast.
        cum = mask.cumsum(1)
        n_vis = mask.sum(1, keepdim=True).clamp(min=1.0)
        early = ((cum <= n_vis * self.early_frac) & mask.bool()).float()[:, None, :, None]
        late = ((cum > n_vis * (1 - self.early_frac)) & mask.bool()).float()[:, None, :, None]
        e = (z * early).sum(2) / early.sum(2).clamp(min=1.0)
        lt = (z * late).sum(2) / late.sum(2).clamp(min=1.0)

        feats = torch.cat([mu, torch.log(var + 1e-6), lt - e], dim=-1)  # [B, M, 3d]
        return self.proj(feats)  # [B, M, d]
