"""Neural baselines, all sharing one interface.

    forward(x: [B, C, L, F], mask: [B, C]) -> [B]  (predicted life, in model space)

C = max_cycles, L = curve_len, F = 3 channels. `mask` is 1 for observed cycles.

The CP* family reimplements BatteryLife's CyclePatch: embed each *cycle* as a
token via an intra-cycle encoder, then run an inter-cycle encoder over the token
sequence. Their ablation shows the intra-cycle stage is where most of the gain
comes from, and that RevIN-style normalisation actively hurts here -- so there
is deliberately none.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..config import ModelConfig


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the cycle axis, ignoring padded cycles. h: [B, C, D]."""
    m = mask.unsqueeze(-1)
    return (h * m).sum(1) / m.sum(1).clamp(min=1.0)


class _Base(nn.Module):
    def __init__(self, cfg: ModelConfig, max_cycles: int, curve_len: int, in_ch: int):
        super().__init__()
        self.cfg = cfg
        self.max_cycles = max_cycles
        self.curve_len = curve_len
        self.in_ch = in_ch
        self.cycle_dim = curve_len * in_ch

    def init_head_bias(self, value: float) -> None:
        """Start the output at the mean training label (in log space).

        Without this the first epochs are spent walking the bias up from zero,
        which at the paper's 5e-5 learning rate takes long enough to distort
        early-stopping. Models with an unusual head override this.
        """
        with torch.no_grad():
            for mod in reversed(list(self.modules())):
                if isinstance(mod, nn.Linear) and mod.out_features == 1 and mod.bias is not None:
                    nn.init.zeros_(mod.weight)
                    nn.init.constant_(mod.bias, value)
                    return
        raise RuntimeError(
            f"{type(self).__name__} has no scalar Linear head; override init_head_bias"
        )


class Dummy(_Base):
    """Predicts a constant -- the training-label mean, learned as a bias.

    BatteryLife reports this as `Dummy`, and it is the floor any real model must
    clear. On Li-ion that floor is MAPE 0.831 / 15%-Acc 0.296.
    """

    def __init__(self, cfg, max_cycles, curve_len, in_ch):
        super().__init__(cfg, max_cycles, curve_len, in_ch)
        self.bias = nn.Parameter(torch.zeros(1))

    def init_head_bias(self, value: float) -> None:
        # No Linear head to find -- the single parameter *is* the prediction.
        with torch.no_grad():
            self.bias.fill_(value)

    def forward(self, x, mask):
        return self.bias.expand(x.shape[0])


class CyclePatchEmbed(nn.Module):
    """Intra-cycle encoder: one cycle's full curve -> one token."""

    def __init__(self, cycle_dim: int, d_model: int, d_ff: int, n_layers: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(cycle_dim, d_model)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )
            for _ in range(n_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))

    def forward(self, x):  # x: [B, C, L, F]
        b, c = x.shape[:2]
        z = self.proj(x.reshape(b, c, -1))
        for blk, ln in zip(self.blocks, self.norms, strict=True):
            z = ln(z + blk(z))
        return z  # [B, C, D]


class _CPModel(_Base):
    """CyclePatch embed -> pluggable inter-cycle encoder -> masked-mean -> scalar."""

    def __init__(self, cfg, max_cycles, curve_len, in_ch, intra_layers: int = 2):
        super().__init__(cfg, max_cycles, curve_len, in_ch)
        self.embed = CyclePatchEmbed(
            self.cycle_dim, cfg.d_model, cfg.d_ff, intra_layers, cfg.dropout
        )
        self.inter = self._build_inter(cfg)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(), nn.Linear(cfg.d_model // 2, 1),
        )

    def _build_inter(self, cfg) -> nn.Module:
        raise NotImplementedError

    def _run_inter(self, z, mask):
        raise NotImplementedError

    def forward(self, x, mask):
        z = self.embed(x)
        z = self._run_inter(z, mask)
        return self.head(masked_mean(z, mask)).squeeze(-1)


class CPMLP(_CPModel):
    """Best overall baseline in the paper: Li-ion MAPE 0.179, 15%-Acc 0.620."""

    def _build_inter(self, cfg):
        return nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(cfg.d_model),
                nn.Linear(cfg.d_model, cfg.d_ff),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_ff, cfg.d_model),
            )
            for _ in range(cfg.n_layers)
        )

    def _run_inter(self, z, mask):
        for blk in self.inter:
            z = z + blk(z)
        return z


class _PosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, z):
        return z + self.pe[:, : z.shape[1]]


class CPTransformer(_CPModel):
    """Li-ion MAPE 0.184; the strongest on Zn-ion (0.515) and Na-ion (0.255)."""

    def _build_inter(self, cfg):
        self.pos = _PosEnc(cfg.d_model, 4096)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        # norm_first disables the nested-tensor fast path anyway; saying so
        # explicitly keeps the warning out of every run's logs.
        return nn.TransformerEncoder(layer, num_layers=cfg.n_layers, enable_nested_tensor=False)

    def _run_inter(self, z, mask):
        return self.inter(self.pos(z), src_key_padding_mask=(mask == 0))


class _CPRecurrent(_CPModel):
    rnn_cls: type[nn.RNNBase] = nn.GRU

    def _build_inter(self, cfg):
        return self.rnn_cls(
            cfg.d_model,
            cfg.d_model,
            num_layers=max(1, cfg.n_layers // 2),
            batch_first=True,
            dropout=cfg.dropout if cfg.n_layers > 2 else 0.0,
        )

    def _run_inter(self, z, mask):
        out, _ = self.inter(z)
        return out


class CPGRU(_CPRecurrent):
    """Li-ion MAPE 0.189."""

    rnn_cls = nn.GRU


class CPLSTM(_CPRecurrent):
    """Li-ion MAPE 0.196."""

    rnn_cls = nn.LSTM


class MLP(_Base):
    """No cycle structure: flatten everything and regress. Li-ion MAPE 0.233."""

    def __init__(self, cfg, max_cycles, curve_len, in_ch):
        super().__init__(cfg, max_cycles, curve_len, in_ch)
        self.proj = nn.Linear(self.cycle_dim, cfg.d_model)
        layers: list[nn.Module] = []
        for _ in range(cfg.n_layers):
            layers += [
                nn.Linear(cfg.d_model, cfg.d_ff),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_ff, cfg.d_model),
            ]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(cfg.d_model, 1)

    def forward(self, x, mask):
        b, c = x.shape[:2]
        z = self.proj(x.reshape(b, c, -1))
        z = masked_mean(z, mask)
        return self.head(self.body(z)).squeeze(-1)


class CNN(_Base):
    """Cycles as image rows, channels as image channels. Li-ion MAPE 0.337."""

    def __init__(self, cfg, max_cycles, curve_len, in_ch):
        super().__init__(cfg, max_cycles, curve_len, in_ch)
        ch = cfg.d_model // 4
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(ch, ch * 2, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(ch * 2, ch * 4, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(ch * 4, 1)

    def forward(self, x, mask):
        # zero out padded cycles so the conv never reads stale rows
        x = x * mask[:, :, None, None]
        z = self.net(x.permute(0, 3, 1, 2)).flatten(1)
        return self.head(z).squeeze(-1)


class PatchTransformer(_Base):
    """PatchTST-style: linear patch embedding, no intra-cycle encoder.

    Included to reproduce the paper's ablation -- it is what CPTransformer
    becomes if you delete the intra-cycle stage. Li-ion MAPE 0.288.
    """

    def __init__(self, cfg, max_cycles, curve_len, in_ch):
        super().__init__(cfg, max_cycles, curve_len, in_ch)
        self.proj = nn.Linear(self.cycle_dim, cfg.d_model)
        self.pos = _PosEnc(cfg.d_model, 4096)
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.enc = nn.TransformerEncoder(
            layer, num_layers=cfg.n_layers, enable_nested_tensor=False
        )
        self.head = nn.Linear(cfg.d_model, 1)

    def forward(self, x, mask):
        b, c = x.shape[:2]
        z = self.pos(self.proj(x.reshape(b, c, -1)))
        z = self.enc(z, src_key_padding_mask=(mask == 0))
        return self.head(masked_mean(z, mask)).squeeze(-1)
