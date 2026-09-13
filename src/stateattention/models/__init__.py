"""Model registry. `build_model(cfg)` is the only entry point trainers use."""

from __future__ import annotations

from ..config import Config, ModelConfig
from .baselines import CNN, CPGRU, CPLSTM, CPMLP, MLP, CPTransformer, Dummy, PatchTransformer
from .fused import FusedSSMAttention
from .mambaformer import MambaFormer

REGISTRY = {
    "fused": FusedSSMAttention,
    "dummy": Dummy,
    "mlp": MLP,
    "cnn": CNN,
    "patchtransformer": PatchTransformer,
    "cpmlp": CPMLP,
    "cptransformer": CPTransformer,
    "cpgru": CPGRU,
    "cplstm": CPLSTM,
    "stateattention": MambaFormer,
}

# Classical (sklearn) baselines live outside the torch path -- see classical.py.
CLASSICAL = ("ridge", "gbdt", "rf", "mean")


def build_model(model_cfg: ModelConfig, max_cycles: int, curve_len: int, in_ch: int = 3):
    name = model_cfg.name.lower()
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {sorted(REGISTRY) + list(CLASSICAL)}")
    return REGISTRY[name](model_cfg, max_cycles=max_cycles, curve_len=curve_len, in_ch=in_ch)


__all__ = ["build_model", "REGISTRY", "CLASSICAL", "Config"]
