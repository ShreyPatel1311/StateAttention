"""Tests for the block y-path FFN variants: y <- y + FFN(LN(y))."""

from __future__ import annotations

import pytest
import torch

from stateattention.models.ffn import FFN_TYPES, KANLayer, MoEFFN, build_ffn


@pytest.mark.parametrize("kind", FFN_TYPES)
def test_shape_and_grad(kind):
    f = build_ffn(kind, 32, 64, dropout=0.0)
    x = torch.randn(2, 7, 32, requires_grad=True)
    out = f(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in f.parameters())


def test_moe_gating_is_sparse_and_normalised():
    """top_k=1 must route each token to exactly one expert, with weight 1."""
    torch.manual_seed(0)
    moe = MoEFFN(16, 32, n_experts=4, top_k=1, noise_std=0.0).eval()
    x = torch.randn(3, 5, 16)
    logits = moe.router(x)
    w = logits.softmax(-1)
    topv, topi = w.topk(1, dim=-1)
    gate = torch.zeros_like(w).scatter(-1, topi, topv)
    gate = gate / gate.sum(-1, keepdim=True).clamp(min=1e-9)
    assert torch.allclose(gate.sum(-1), torch.ones(3, 5), atol=1e-6)
    assert (gate > 0).sum(-1).max().item() == 1, "top_k=1 selected more than one expert"

    moe(x)
    assert moe.last_usage.shape == (4,)
    assert abs(moe.last_usage.sum().item() - 1.0) < 1e-5


def test_moe_top2_selects_two():
    torch.manual_seed(0)
    moe = MoEFFN(16, 32, n_experts=4, top_k=2, noise_std=0.0).eval()
    moe(torch.randn(2, 4, 16))
    assert abs(moe.last_usage.sum().item() - 2.0) < 1e-5


def test_kan_splines_form_a_partition_of_unity():
    """B-spline bases must sum to 1 inside the grid range -- else the layer is
    silently attenuating its input wherever the grid does not cover it."""
    layer = KANLayer(4, 8, grid_size=3, spline_order=3, rng=(-2.0, 2.0))
    x = torch.linspace(-1.5, 1.5, 40).unsqueeze(-1).expand(-1, 4)
    bases = layer.b_splines(x)  # [40, 4, G+k]
    assert torch.allclose(bases.sum(-1), torch.ones(40, 4), atol=1e-4)


def test_kan_is_bounded_outside_the_grid():
    """Spline support is compact, which is why the KAN head did not blow up
    where an MLP extrapolated to exp() overflow."""
    torch.manual_seed(0)
    layer = KANLayer(4, 8, grid_size=3, rng=(-2.0, 2.0)).eval()
    with torch.no_grad():
        far = layer(torch.full((1, 4), 500.0))
    assert torch.isfinite(far).all()


def test_unknown_ffn_rejected():
    with pytest.raises(ValueError, match="ffn_type"):
        build_ffn("transformer", 16, 32)
