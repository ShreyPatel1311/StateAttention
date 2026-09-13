"""Tests for the SOC view -- one token per SOC interval, pooled over cycles."""

from __future__ import annotations

import pytest
import torch

from stateattention.config import ModelConfig
from stateattention.models import build_model
from stateattention.models.socview import SOCView


def test_shape_and_interval_count():
    v = SOCView(3, 32, curve_len=300, patch=25)
    assert v.n_intervals == 12
    out = v(torch.randn(2, 40, 300, 3), torch.ones(2, 40))
    assert out.shape == (2, 12, 32)


def test_indivisible_patch_rejected():
    with pytest.raises(ValueError, match="divisible"):
        SOCView(3, 16, curve_len=300, patch=7)


def test_intervals_are_spatially_local():
    """An interval token must depend only on its own SOC window.

    This is the whole point of the view: if perturbing one region of the curve
    moved every token, it would be no better than the flattening it replaces.
    """
    torch.manual_seed(0)
    v = SOCView(3, 16, curve_len=100, patch=25).eval()  # 4 intervals of 25 pts
    x = torch.randn(1, 6, 100, 3)
    with torch.no_grad():
        base = v.conv(x.reshape(6, 100, 3).transpose(1, 2))  # [6, d, M]
        x2 = x.clone()
        x2[:, :, 0:25] += 10.0  # perturb ONLY the first SOC window
        pert = v.conv(x2.reshape(6, 100, 3).transpose(1, 2))
    moved = (base - pert).abs().amax(dim=(0, 1))  # per interval
    assert moved[0] > 1e-3, "the perturbed interval should move"
    assert moved[1:].max() < 1e-6, "other intervals must not see it"


def test_padding_invariance():
    torch.manual_seed(0)
    v = SOCView(3, 16, curve_len=100, patch=25).eval()
    x = torch.randn(2, 10, 100, 3)
    mask = torch.ones(2, 10)
    mask[:, 6:] = 0
    dirty = x.clone()
    dirty[:, 6:] = 99.0
    with torch.no_grad():
        a = v(x * mask[..., None, None], mask)
        b = v(dirty * mask[..., None, None], mask)
    assert torch.allclose(a, b, atol=1e-5)


def test_delta_follows_prefix_length_not_a_fixed_index():
    """early/late windows are derived from the mask, so a 10-cycle prefix and a
    100-cycle prefix both get a meaningful contrast."""
    torch.manual_seed(0)
    v = SOCView(3, 16, curve_len=100, patch=25).eval()
    x = torch.randn(1, 20, 100, 3)
    short, long = torch.zeros(1, 20), torch.ones(1, 20)
    short[:, :5] = 1
    with torch.no_grad():
        a = v(x * short[..., None, None], short)
        b = v(x * long[..., None, None], long)
    assert not torch.allclose(a, b, atol=1e-4), "prefix length must change the token"


@pytest.mark.parametrize("soc", [False, True])
def test_model_forward_with_and_without(soc):
    cfg = ModelConfig(name="fused", d_model=32, d_ff=64, n_layers=2, n_heads=4,
                      attn_on="hx", soc_view=soc, soc_patch=15)
    m = build_model(cfg, 12, 60)
    x = torch.randn(3, 12, 60, 3)
    mask = torch.ones(3, 12)
    mask[:, 8:] = 0
    out = m(x * mask[..., None, None], mask)
    assert out.shape == (3,) and torch.isfinite(out).all()
