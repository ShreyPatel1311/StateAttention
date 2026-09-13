"""Feed-forward variants for the block's y-path: y <- y + FFN(LN(y)).

Three options, selected by `ffn_type`:

    mlp   Linear -> GELU -> Linear                      (the default)
    moe   sparse mixture of experts, noisy top-k gating
    kan   Kolmogorov-Arnold layers: learnable B-spline
          activations on edges instead of fixed ones on nodes

Parameter cost per block at d_model=96, d_ff=192 differs sharply -- see
`build_ffn` -- so any comparison has to be width-matched.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class MoEFFN(nn.Module):
    """Sparse MoE with noisy top-k gating.

    Every expert is evaluated and then masked, so this is exact top-k but saves
    no compute -- fine for an ablation measuring quality, not throughput. There
    is no load-balancing auxiliary loss; instead the router logits get Gaussian
    noise during training, which discourages collapse onto one expert without
    changing the loss signature. Expert usage is exposed as `last_usage` so a
    collapsed router is visible rather than silent.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0,
                 n_experts: int = 4, top_k: int = 1, noise_std: float = 1.0):
        super().__init__()
        self.n_experts, self.top_k, self.noise_std = n_experts, top_k, noise_std
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(d_ff, d_model))
            for _ in range(n_experts)
        )
        self.register_buffer("last_usage", torch.zeros(n_experts), persistent=False)

    def forward(self, x):
        logits = self.router(x)  # [B, T, E]
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        w = logits.softmax(-1)

        topv, topi = w.topk(self.top_k, dim=-1)
        gate = torch.zeros_like(w).scatter(-1, topi, topv)  # zero for unselected
        gate = gate / gate.sum(-1, keepdim=True).clamp(min=1e-9)

        with torch.no_grad():
            self.last_usage = gate.gt(0).float().mean(dim=(0, 1)).detach()

        out = 0.0
        for e, expert in enumerate(self.experts):
            g = gate[..., e : e + 1]
            out = out + g * expert(x)
        return out


class KANLayer(nn.Module):
    """One KAN layer: y = W_base @ silu(x) + W_spline @ Bspline(x).

    Learnable univariate functions on edges (B-splines) rather than a fixed
    activation on nodes. Each edge carries grid_size + spline_order
    coefficients, so parameter count is (grid+order)x a Linear of the same
    shape -- the reason the KAN arm needs a much narrower hidden width to fit
    the same budget.
    """

    def __init__(self, n_in: int, n_out: int, grid_size: int = 3,
                 spline_order: int = 3, rng: tuple[float, float] = (-2.0, 2.0)):
        super().__init__()
        self.n_in, self.n_out = n_in, n_out
        self.grid_size, self.k = grid_size, spline_order
        h = (rng[1] - rng[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1) * h + rng[0]
        self.register_buffer("grid", grid.expand(n_in, -1).contiguous())
        self.base_weight = nn.Parameter(torch.empty(n_out, n_in))
        self.spline_weight = nn.Parameter(torch.empty(n_out, n_in * (grid_size + spline_order)))
        nn.init.kaiming_uniform_(self.base_weight, a=5**0.5)
        nn.init.normal_(self.spline_weight, std=0.1 / (n_in**0.5))

    def b_splines(self, x):
        """Cox-de Boor recursion. x: [..., n_in] -> [..., n_in, G+k]."""
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for p in range(1, self.k + 1):
            left = (x - grid[:, : -(p + 1)]) / (grid[:, p:-1] - grid[:, : -(p + 1)])
            right = (grid[:, p + 1 :] - x) / (grid[:, p + 1 :] - grid[:, 1:-p])
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        return bases

    def forward(self, x):
        lead, n_in = x.shape[:-1], x.shape[-1]
        flat = x.reshape(-1, n_in)
        base = F.linear(F.silu(flat), self.base_weight)
        spline = F.linear(self.b_splines(flat).flatten(1), self.spline_weight)
        return (base + spline).reshape(*lead, self.n_out)


class KANFFN(nn.Module):
    """Two KAN layers. Hidden width is deliberately narrow.

    A KAN edge costs (grid + order) parameters where a Linear edge costs one,
    so matching an MLP's d_ff would blow the budget ~6x. KANs are meant to need
    fewer nodes for the same expressivity, so the hidden width is d_model // 2
    by default rather than 2 * d_model.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0,
                 hidden: int | None = None, grid_size: int = 3):
        super().__init__()
        hidden = hidden or max(8, d_model // 2)
        self.l1 = KANLayer(d_model, hidden, grid_size=grid_size)
        self.drop = nn.Dropout(dropout)
        self.l2 = KANLayer(hidden, d_model, grid_size=grid_size)

    def forward(self, x):
        return self.l2(self.drop(self.l1(x)))


FFN_TYPES = ("mlp", "moe", "kan")


def build_ffn(kind: str, d_model: int, d_ff: int, dropout: float = 0.0, **kw) -> nn.Module:
    kind = kind.lower()
    if kind == "mlp":
        return MLPFFN(d_model, d_ff, dropout)
    if kind == "moe":
        return MoEFFN(d_model, d_ff, dropout, **kw)
    if kind == "kan":
        return KANFFN(d_model, d_ff, dropout, **kw)
    raise ValueError(f"ffn_type must be one of {FFN_TYPES}, got {kind!r}")
