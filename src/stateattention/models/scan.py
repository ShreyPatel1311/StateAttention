"""Parallel prefix scan for first-order linear recurrences.

Shared by every SSM in this repo. Both the diagonal SSM over cycle tokens and
Mamba's S6 block reduce to

    h_t = a_t * h_{t-1} + b_t

with a_t and b_t elementwise, which is associative under

    (a2, b2) o (a1, b1) = (a2 a1, a2 b1 + b2)

and therefore admits a prefix scan. That associativity is exactly what a
softmax inside the recurrence would destroy.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def parallel_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Solve h_t = a_t * h_{t-1} + b_t for all t, with h_{-1} = 0.

    Hillis-Steele scan: ceil(log2 T) sequential steps instead of T. Any trailing
    dimensions are treated elementwise, so this handles both [B, T, D] cycle
    states and Mamba's [B, T, d_inner, d_state] matrix states.
    """
    t = a.shape[1]
    d = 1
    while d < t:
        # shift right along time by d; identity elements are a=1, b=0
        pad = [0, 0] * (a.dim() - 2) + [d, 0]
        a_sh = F.pad(a[:, :-d], pad, value=1.0)
        b_sh = F.pad(b[:, :-d], pad, value=0.0)
        b = b + a * b_sh
        a = a * a_sh
        d *= 2
    return b
