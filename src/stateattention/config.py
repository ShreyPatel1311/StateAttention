"""Config loading. Plain YAML -> dataclasses, with CLI dot-path overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataConfig:
    name: str = "batterylife"  # "batterylife" | "nasa"
    root: str = "data"
    # BatteryLife: which domain split to train/eval on
    domain: str = "li-ion"  # li-ion | zn-ion | na-ion | calb
    # Task framing, shared by both datasets
    max_cycles: int = 100  # S <= 100 early cycles are visible to the model
    min_cycles: int = 1  # shortest prefix turned into a training sample
    curve_len: int = 300  # each cycle resampled to this many points
    channels: tuple[str, ...] = ("voltage", "current", "capacity")
    # Divide the capacity channel by each cell's nominal capacity, turning raw
    # Ah into dimensionless SOC. Off means Li-ion and Zn-ion differ by ~2400x on
    # that channel, which makes cross-chemistry training or transfer hopeless.
    normalize_capacity: bool = True
    # NASA only
    eol_soh: float = 0.70  # NASA convention: EOL at 30% fade from 2.0 Ah rated
    nasa_cells: tuple[str, ...] = ()  # empty = all cells found on disk
    cv_folds: int = 0  # >0 enables leave-N-battery-out CV (NASA)
    # Subsampling, for fast ablations. 0 = use everything.
    max_cells: int = 0  # cap cells per split
    sample_stride: int = 1  # keep every k-th prefix length per cell


@dataclass
class ModelConfig:
    name: str = "cpmlp"
    d_model: int = 128
    d_ff: int = 256
    n_layers: int = 6
    n_heads: int = 4
    dropout: float = 0.0
    # mambaformer only
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    attn_every: int = 2  # interleave: attention block every k-th layer
    # fused model: where the attention sits relative to the linear recurrence.
    # "h" = state attends to state, "y" = state queries the input cycles.
    attn_on: str = "y"  # none | h | hx | y | both
    delta_values: bool = False  # attend over (v_j - v_t) instead of v_j
    # feed-forward in the block's y-path: y <- y + FFN(LN(y))
    ffn_type: str = "mlp"  # mlp | moe | kan
    # intra-cycle encoder. "flatten" reshapes the whole curve through one Linear
    # and is 32-45% of the trajectory model's parameters; the others keep the
    # SOC axis. See models/embeds.py.
    embed_type: str = "flatten"  # flatten | patch | patchattn | conv | delta
    embed_patch: int = 25
    # recurrent core inside FusedBlock: the shipped 1-state diagonal SSM, or the
    # real Mamba S6 used by the mambaformer backbone.
    ssm_core: str = "diagonal"  # diagonal | s6
    # trajectory head. "free" emits any curve; "monotone" emits non-negative
    # per-knot decrements and integrates, so SOH cannot rise.
    head_type: str = "free"  # free | monotone | rateshape
    # SOC view: an extra token per SOC interval, summarising its evolution
    # across cycles. Adapted from BatteryMFormer, whose ablation makes it
    # their largest single contributor.
    soc_view: bool = False
    soc_patch: int = 25  # points per SOC interval; curve_len must divide by it
    moe_experts: int = 4
    moe_top_k: int = 1
    kan_grid: int = 3


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 5e-5
    weight_decay: float = 0.0
    patience: int = 5  # early-stopping patience on val MAPE
    loss: str = "mse"  # mse | mae | huber | log_mse
    seeds: tuple[int, ...] = (2024, 2025, 2026)
    num_workers: int = 4
    amp: bool = True
    grad_clip: float = 1.0
    device: str = "auto"


@dataclass
class HubConfig:
    data_repo: str = ""
    ckpt_repo: str = ""
    push_checkpoints: bool = False


@dataclass
class Config:
    experiment: str = "default"
    out_dir: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    hub: HubConfig = field(default_factory=HubConfig)


def _coerce(cls, value: Any):
    """Build a dataclass from a dict, keeping tuple fields as tuples."""
    if not is_dataclass(cls) or not isinstance(value, dict):
        return value
    kwargs = {}
    for f in fields(cls):
        if f.name not in value:
            continue
        v = value[f.name]
        if is_dataclass(f.type) if isinstance(f.type, type) else False:
            v = _coerce(f.type, v)
        elif isinstance(v, list):
            v = tuple(v)
        kwargs[f.name] = v
    return cls(**kwargs)


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load YAML, apply `a.b=c` overrides, expand ${ENV_VAR} references."""
    raw = yaml.safe_load(Path(path).read_text()) or {}

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must look like key.path=value, got {ov!r}")
        key, val = ov.split("=", 1)
        node = raw
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)

    raw = _expand_env(raw)

    cfg = Config(
        experiment=raw.get("experiment", "default"),
        out_dir=raw.get("out_dir", "runs"),
        data=_coerce(DataConfig, raw.get("data", {})),
        model=_coerce(ModelConfig, raw.get("model", {})),
        train=_coerce(TrainConfig, raw.get("train", {})),
        hub=_coerce(HubConfig, raw.get("hub", {})),
    )
    return cfg


def _expand_env(node):
    if isinstance(node, dict):
        return {k: _expand_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_env(v) for v in node]
    if isinstance(node, str):
        return os.path.expandvars(node)
    return node


def resolve_path(p: str | Path) -> Path:
    """Relative paths in configs are relative to the repo root, not the cwd."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p
