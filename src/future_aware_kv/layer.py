"""Minimal multihead layer: five bias-free projections and one output map."""

from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from .attention import attention
from .decode import DecodeState, decode_step


def apply_rope(x: Tensor, *, offset=0, base=10000.0):
    """Adjacent-pair RoPE, positive rotation, same convention as reversal."""
    d = x.shape[-1]
    if d % 2:
        raise ValueError("RoPE requires even head width")
    pos = torch.arange(offset, offset + x.shape[1], device=x.device, dtype=torch.float32)
    freq = torch.exp(-math.log(base) * torch.arange(0, d, 2, device=x.device) / d)
    angle = pos[:, None] * freq[None, :]
    cos = angle.cos().to(x.dtype)[None, :, None]
    sin = angle.sin().to(x.dtype)[None, :, None]
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


class FutureAwareKVAttention(nn.Module):
    """Drop-in [B,T,width] attention mixer; no residual/MLP/normalization.

    Default initialization is nn.Linear's initialization. Reproducing reversal
    or nanochat needs their archived full-model initialization, not defaults.
    Parameters: 6*width**2; head width=width/heads. Explicit fixed gains.
    """

    def __init__(
        self,
        width: int,
        heads: int,
        *,
        rope=False,
        rope_base=10000.0,
        backend="auto",
        schedule="eager",
        inner_gain=1.0,
        outer_gain=1.0,
        kv_gain=1.0,
    ):
        super().__init__()
        if min(width, heads) < 1 or width % heads:
            raise ValueError("width must divide into positive heads")
        if rope and width // heads % 2:
            raise ValueError("RoPE requires even head width")
        if not math.isfinite(rope_base) or rope_base <= 0:
            raise ValueError("RoPE base must be positive")
        if any(not math.isfinite(g) for g in (inner_gain, outer_gain, kv_gain)):
            raise ValueError("gains must be finite")
        if backend not in ("auto", "reference", "triton") or schedule not in ("eager", "graph"):
            raise ValueError("invalid backend or schedule")
        self.width, self.heads, self.head_width = width, heads, width // heads
        self.rope, self.rope_base = rope, rope_base
        self.backend, self.schedule = backend, schedule
        self.inner_gain, self.outer_gain, self.kv_gain = inner_gain, outer_gain, kv_gain
        # Keep separate parameters: Muon orthogonalizes each projection matrix
        # independently, as in the nanochat training integration.
        self.qf_proj = nn.Linear(width, width, bias=False)
        self.kf_proj = nn.Linear(width, width, bias=False)
        self.qc_proj = nn.Linear(width, width, bias=False)
        self.kc_proj = nn.Linear(width, width, bias=False)
        self.vc_proj = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def project(self, x: Tensor, *, offset=0):
        if x.ndim != 3 or x.shape[-1] != self.width:
            raise ValueError("x must be [B,T,width]")
        shape = (*x.shape[:2], self.heads, self.head_width)
        qf = self.qf_proj(x).reshape(shape)
        kf = self.kf_proj(x).reshape(shape)
        qc = self.qc_proj(x).reshape(shape)
        kc = self.kc_proj(x).reshape(shape)
        vc = self.vc_proj(x).reshape(shape)
        if self.rope:
            qf, kf, qc, kc = (
                apply_rope(p, offset=offset, base=self.rope_base) for p in (qf, kf, qc, kc)
            )
        return (
            (qf * self.inner_gain).contiguous(),
            kf.contiguous(),
            (qc * self.outer_gain).contiguous(),
            (kc * self.kv_gain).contiguous(),
            (vc * self.kv_gain).contiguous(),
        )

    def forward(self, x: Tensor):
        y = attention(*self.project(x), backend=self.backend, schedule=self.schedule)
        return self.output(y.reshape_as(x))

    @torch.no_grad()
    def decode(self, x: Tensor, state: DecodeState | None = None):
        y, state = decode_step(
            *self.project(x, offset=0 if state is None else state.length), state=state
        )
        return self.output(y.reshape_as(x)), state
