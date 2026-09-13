"""MambaFormer -- your model goes here.

This is a working starting point, not the finished idea: CyclePatch tokens fed
through an interleaved stack of selective-SSM blocks and attention blocks. The
motivation is that degradation is a long, smooth, causal process (which an SSM
scans cheaply) punctuated by protocol-specific events (which attention can
address directly). Swap the block contents freely -- `forward(x, mask) -> [B]`
is the only contract the trainer depends on.

If `mamba-ssm` is installed (the [gpu] extra, CUDA only) we use the fused kernel.
Otherwise a pure-PyTorch selective scan runs instead: same maths, much slower,
but it means the model is debuggable on a laptop before you rent a GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .baselines import CyclePatchEmbed, _Base, _PosEnc, masked_mean
from .scan import parallel_scan

try:  # fused CUDA kernel when mamba-ssm is available
    from mamba_ssm import Mamba as _FusedMamba

    HAS_MAMBA_SSM = True
except ImportError:  # pragma: no cover - depends on install extras
    _FusedMamba = None
    HAS_MAMBA_SSM = False


class SelectiveScanFallback(nn.Module):
    """Pure-PyTorch S6 block, via a parallel prefix scan.

    Numerically equivalent to mamba-ssm's fused kernel but written in plain
    PyTorch, so it runs anywhere. Slower than the CUDA kernel; not the O(L)
    Python loop it replaced.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner, self.d_state = d_inner, d_state

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # A is kept in log space and negated so the recurrence stays stable.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, u: torch.Tensor) -> torch.Tensor:  # [B, L, D]
        b, ln, _ = u.shape
        xz = self.in_proj(u)
        x, z = xz.chunk(2, dim=-1)

        x = self.conv(x.transpose(1, 2))[..., :ln].transpose(1, 2)
        x = F.silu(x)

        dbc = self.x_proj(x)
        dt, B_, C_ = torch.split(dbc, [1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # [B, L, d_inner]

        A = -torch.exp(self.A_log)  # [d_inner, d_state]
        dA = torch.exp(dt.unsqueeze(-1) * A)  # [B, L, d_inner, d_state]
        dBx = dt.unsqueeze(-1) * B_.unsqueeze(2) * x.unsqueeze(-1)

        # S6 is h_t = dA_t * h_{t-1} + dBx_t, elementwise over (d_inner,
        # d_state) -- the same linear recurrence the cycle-level SSM uses, so
        # the same prefix scan applies. Which implementation wins depends on
        # the device, measured on an A10 at d_model=192, B=32:
        #
        #   GPU: scan 54.8ms / 2.13GB   vs  loop 143.9ms / 0.55GB
        #   CPU: scan  373ms            vs  loop   98ms
        #
        # The scan does ~log2(T) passes over the whole [B,L,d_inner,d_state]
        # tensor -- more total work than the loop's T passes over a slice of it
        # -- so it only pays where that work runs in parallel. On CPU it loses;
        # on GPU it wins 2.6x because 100 tiny kernel launches dominate there.
        # Both paths are numerically identical to 6e-17.
        if u.is_cuda:
            h = parallel_scan(dA, dBx)  # [B, L, d_inner, d_state]
            y = torch.einsum("bldn,bln->bld", h, C_) + self.D * x
        else:
            h = x.new_zeros(b, self.d_inner, self.d_state)
            ys = []
            for t in range(ln):
                h = dA[:, t] * h + dBx[:, t]
                ys.append(torch.einsum("bdn,bn->bd", h, C_[:, t]))
            y = torch.stack(ys, dim=1) + self.D * x
        return self.out_proj(y * F.silu(z))


def make_ssm(d_model: int, cfg: ModelConfig) -> nn.Module:
    if HAS_MAMBA_SSM and torch.cuda.is_available():
        return _FusedMamba(
            d_model=d_model, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand
        )
    return SelectiveScanFallback(d_model, cfg.d_state, cfg.d_conv, cfg.expand)


class SSMBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.d_model)
        self.ssm = make_ssm(cfg.d_model, cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, z, mask):
        # padded cycles must not leak into the causal scan
        z = z * mask.unsqueeze(-1)
        return z + self.drop(self.ssm(self.norm(z)))


class AttnBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, z, mask):
        h = self.norm1(z)
        a, _ = self.attn(h, h, h, key_padding_mask=(mask == 0), need_weights=False)
        z = z + a
        return z + self.ff(self.norm2(z))


class MambaFormer(_Base):
    def __init__(self, cfg: ModelConfig, max_cycles: int, curve_len: int, in_ch: int = 3):
        super().__init__(cfg, max_cycles, curve_len, in_ch)
        self.embed = CyclePatchEmbed(curve_len * in_ch, cfg.d_model, cfg.d_ff, 2, cfg.dropout)
        self.pos = _PosEnc(cfg.d_model, max(4096, max_cycles + 1))

        # every attn_every-th layer is attention, the rest are SSM scans
        self.blocks = nn.ModuleList(
            AttnBlock(cfg) if (i + 1) % max(1, cfg.attn_every) == 0 else SSMBlock(cfg)
            for i in range(cfg.n_layers)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
        )

    def forward(self, x, mask):
        z = self.pos(self.embed(x))
        for blk in self.blocks:
            z = blk(z, mask)
        return self.head(masked_mean(z, mask)).squeeze(-1)
