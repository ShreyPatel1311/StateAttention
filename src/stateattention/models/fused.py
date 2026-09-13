"""Fused SSM + attention, with the attention placement as an ablation axis.

The starting point was

    h_t = softmax(A h_{t-1} (B x_t)^T / sqrt(d)) (B x_t)
    y_t = C h_t + D x_t

which does not work: at time t there is exactly one key, so the softmax has
nothing to normalise over. Empirically the state stops depending on its own
history within ~20 steps (wiping h at t=50 changes h_99 by 0.0), because a
softmax output is a convex combination and cannot accumulate. It also makes the
recurrence non-linear, so the parallel scan -- and the fused CUDA kernel -- are
gone.

The fix keeps the recurrence linear and moves the softmax to where it has a set
of keys to normalise over. The original spec has two distinct properties --
queries come from the state while keys/values come from the *input*, and the
attention output becomes the *state* -- so the arms separate them:

    h_t = a_t * h_{t-1} + b_t                              (linear, associative)

    "hx"   h'_t = h_t + Attn(q=W_q h_{t-1}, KV=z_{1:t})    <- faithful to the spec:
           y_t  = C h'_t + D z_t                              previous state queries
                                                              input, result modifies
                                                              the state
    "h"    h'_t = h_t + Attn(q=W_q h_t, KV=h_{1:t})        state queries state
           y_t  = C h'_t + D z_t
    "y"    y_t  = C h_t + D z_t + Attn(q=W_q h_t, KV=z_{1:t})
                                                           state queries input, but the
                                                           result never reaches the state
    "both" h and y together
    "none" pure SSM control

"hx" is the arm that matches the original equation. "h" changes the key/value
source; "y" changes where the output lands. Separating them says which of the
two properties is doing the work.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .baselines import CyclePatchEmbed, _Base, _PosEnc
from .ffn import build_ffn
from .scan import parallel_scan  # re-exported: fused.parallel_scan is a public name
from .socview import SOCView

ARMS_OK = ("none", "h", "hx", "y", "both")


class DiagonalSSM(nn.Module):
    """Selective diagonal SSM over cycle tokens: h_t = a_t * h_{t-1} + b_t.

    a_t and b_t are functions of the input, which is Mamba's selectivity. The
    recurrence is linear in h_{t-1}, so it is associative and the parallel scan
    above applies -- as would mamba-ssm's fused kernel on GPU.
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.to_dt = nn.Linear(d_model, d_model)
        self.to_b = nn.Linear(d_model, d_model)
        self.to_gate = nn.Linear(d_model, d_model)
        # A is negative so exp(dt * A) lands in (0, 1) and the state decays
        self.A_log = nn.Parameter(torch.zeros(d_model))
        self.D = nn.Parameter(torch.ones(d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        u = self.norm(z)
        dt = nn.functional.softplus(self.to_dt(u))  # [B, T, D], strictly positive
        a = torch.exp(-dt * torch.exp(self.A_log))  # decay per step, in (0, 1)
        b = self.to_b(u) * dt

        # Padded cycles must be inert: decay 1 and input 0 leave h unchanged.
        m = mask.unsqueeze(-1)
        a = a * m + (1.0 - m)
        b = b * m

        h = parallel_scan(a, b)

        gate = torch.sigmoid(self.to_gate(u))
        return z + self.drop(gate * (h + self.D * u))


class StateAttention(nn.Module):
    """Causal attention whose query comes from the SSM state.

    Keys and values come from `kv` -- the states themselves for the "h" arm, the
    cycle-token inputs for the "y" arm.
    """

    def __init__(self, cfg: ModelConfig, delta_values: bool = False):
        super().__init__()
        d = cfg.d_model
        self.h = cfg.n_heads
        self.dh = d // cfg.n_heads
        self.delta_values = delta_values
        self.nq, self.nkv = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def _split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.h, self.dh).transpose(1, 2)  # [B, H, T, dh]

    def forward(self, state: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor):
        b, t, _ = state.shape
        q = self._split(self.q(self.nq(state)))
        kv_n = self.nkv(kv)
        k, v = self._split(self.k(kv_n)), self._split(self.v(kv_n))

        if self.delta_values:
            # Values become differences v_j - v_t, making the operation a
            # learned generalisation of Severson's dQ(V) rather than a
            # weighted average of raw cycle content.
            v = v.unsqueeze(2) - v.unsqueeze(3)  # [B, H, T_q, T_k, dh]

        scores = (q @ k.transpose(-2, -1)) / (self.dh**0.5)
        causal = torch.ones(t, t, dtype=torch.bool, device=state.device).triu(1)
        scores = scores.masked_fill(causal, float("-inf"))
        scores = scores.masked_fill(~mask.bool()[:, None, None, :], float("-inf"))
        # A padded query row is entirely -inf; softmax would emit NaN there.
        scores = torch.where(
            scores.isinf().all(-1, keepdim=True), torch.zeros_like(scores), scores
        )
        w = self.drop(scores.softmax(-1))

        out = (w.unsqueeze(-1) * v).sum(-2) if self.delta_values else w @ v
        out = out.transpose(1, 2).reshape(b, t, self.h * self.dh)
        return self.o(out) * mask.unsqueeze(-1)


class FusedBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, attn_on: str, delta_values: bool):
        super().__init__()
        self.attn_on = attn_on
        # Recurrent core. "diagonal" is the 1-state selective recurrence this
        # model shipped with; "s6" is the real Mamba block (d_state=16, depthwise
        # conv, 2x expansion). hx wins the Zn-ion hard fold with the weaker core,
        # so whether the attention coupling or the small capacity is doing the
        # work is only separable by swapping it.
        core = getattr(cfg, "ssm_core", "diagonal")
        if core == "s6":
            from .mambaformer import SelectiveScanFallback
            self.ssm = SelectiveScanFallback(
                cfg.d_model, d_state=cfg.d_state, d_conv=cfg.d_conv,
                expand=cfg.expand)
            self.ssm_pre = nn.LayerNorm(cfg.d_model)
        elif core == "diagonal":
            self.ssm = DiagonalSSM(cfg.d_model, cfg.dropout)
            self.ssm_pre = None
        else:
            raise ValueError(f"ssm_core must be 'diagonal' or 's6', got {core!r}")
        self.attn_h = (
            StateAttention(cfg, delta_values) if attn_on in ("h", "both") else None
        )
        # "hx": query from the state, keys/values from the input cycle tokens,
        # and the result folded back into the state -- the original formulation.
        self.attn_hx = StateAttention(cfg, delta_values) if attn_on == "hx" else None
        self.attn_y = (
            StateAttention(cfg, delta_values) if attn_on in ("y", "both") else None
        )
        self.C = nn.Linear(cfg.d_model, cfg.d_model)
        self.Dp = nn.Linear(cfg.d_model, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)
        kind = getattr(cfg, "ffn_type", "mlp")
        extra: dict = {}
        if kind == "moe":
            extra = {"n_experts": getattr(cfg, "moe_experts", 4),
                     "top_k": getattr(cfg, "moe_top_k", 1)}
        elif kind == "kan":
            extra = {"grid_size": getattr(cfg, "kan_grid", 3)}
        self.ff = build_ffn(kind, cfg.d_model, cfg.d_ff, cfg.dropout, **extra)

    def forward(self, z, mask):
        if self.ssm_pre is not None:
            # S6 takes [B, T, D] and no mask; keep padded cycles inert and
            # residual-connect so the two cores are interchangeable here.
            h = z + self.ssm(self.ssm_pre(z)) * mask.unsqueeze(-1)
        else:
            h = self.ssm(z, mask)
        if self.attn_h is not None:
            h = h + self.attn_h(h, h, mask)  # q=state, kv=state -> state
        if self.attn_hx is not None:
            # Query is h_{t-1}, not h_t. The spec is
            #     h_t = softmax(A h_{t-1} (B x_t)^T / sqrt d) (B x_t)
            # so the query is the state *before* this step's input is absorbed.
            # Querying with h_t instead would let the query see z_t through the
            # scan -- selecting over something it already contains.
            h_prev = nn.functional.pad(h[:, :-1], (0, 0, 1, 0))  # h_{-1} = 0
            h = h + self.attn_hx(h_prev, z, mask)  # q=h_{t-1}, kv=input -> state
        y = self.C(h) + self.Dp(z)  # C h_t + D x_t
        if self.attn_y is not None:
            y = y + self.attn_y(h, z, mask)  # state queries the input cycles
        return y + self.ff(self.norm(y))


class MeanVarPool(nn.Module):
    """Pool cycle tokens with both moments.

    The single strongest hand-crafted predictor on this benchmark is
    log var(dQ(V)) -- a *second* moment. Attention and mean-pooling produce
    weighted first moments, so variance has to be learned through a squaring
    nonlinearity. Supplying it directly is a few lines and costs nothing.
    """

    def forward(self, y, mask):
        m = mask.unsqueeze(-1)
        n = m.sum(1).clamp(min=1.0)
        mu = (y * m).sum(1) / n
        var = ((y - mu.unsqueeze(1)) ** 2 * m).sum(1) / n
        return torch.cat([mu, torch.log(var + 1e-6)], dim=-1)


class FusedSSMAttention(_Base):
    """CyclePatch -> [SSM (+attention)] x L -> mean/var pool -> scalar."""

    def __init__(self, cfg: ModelConfig, max_cycles: int, curve_len: int, in_ch: int = 3):
        super().__init__(cfg, max_cycles, curve_len, in_ch)
        attn_on = getattr(cfg, "attn_on", "y")
        if attn_on not in ARMS_OK:
            raise ValueError(f"attn_on must be one of {ARMS_OK}, got {attn_on!r}")
        self.attn_on = attn_on
        delta = bool(getattr(cfg, "delta_values", False))

        self.embed = CyclePatchEmbed(self.cycle_dim, cfg.d_model, cfg.d_ff, 2, cfg.dropout)
        self.pos = _PosEnc(cfg.d_model, max(4096, max_cycles + 1))
        self.blocks = nn.ModuleList(
            FusedBlock(cfg, attn_on, delta) for _ in range(cfg.n_layers)
        )
        self.pool = MeanVarPool()

        # Optional second view over the SOC axis. The cycle path pools *over*
        # SOC positions, so on its own the model cannot isolate a drift confined
        # to one voltage plateau.
        self.soc = None
        head_in = 2 * cfg.d_model
        if getattr(cfg, "soc_view", False):
            patch = getattr(cfg, "soc_patch", 25)
            self.soc = SOCView(in_ch, cfg.d_model, curve_len, patch, cfg.dropout)
            # flatten rather than pool: interval identity is the point, and the
            # intervals are ordered, so which one moved has to survive.
            self.soc_proj = nn.Sequential(
                nn.Flatten(1),
                nn.LayerNorm(self.soc.n_intervals * cfg.d_model),
                nn.Linear(self.soc.n_intervals * cfg.d_model, cfg.d_model),
            )
            head_in += cfg.d_model

        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(self, x, mask):
        z = self.pos(self.embed(x)) * mask.unsqueeze(-1)
        for blk in self.blocks:
            z = blk(z, mask)
        feats = self.pool(z, mask)
        if self.soc is not None:
            feats = torch.cat([feats, self.soc_proj(self.soc(x, mask))], dim=-1)
        return self.head(feats).squeeze(-1)
