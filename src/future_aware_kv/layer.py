"""A small trainable future-aware KV attention layer."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .attention import PROJECTION_NAMES, _future_aware_kv_queries, future_aware_kv_reference


def _gate_parameter(value: float, heads: int) -> nn.Parameter:
    if not 0 < value < 1:
        raise ValueError("trainable gate initialization must be between 0 and 1")
    return nn.Parameter(torch.full((heads,), math.log(-math.log(value))))


def _threshold_parameter(
    fraction: float, reference_length: int, heads: int
) -> nn.Parameter:
    if not math.isfinite(fraction):
        raise ValueError("threshold initialization fraction must be finite")
    desired = fraction * (reference_length - 1)
    desired = min(reference_length + 9.999, max(-9.999, desired))
    unit = (desired + 10) / (reference_length + 20)
    return nn.Parameter(torch.full((heads,), math.log(unit / (1 - unit))))


def _rope_table(length: int, dim: int, device: torch.device, dtype: torch.dtype,
                base: float) -> tuple[Tensor, Tensor]:
    frequency = torch.exp(
        -math.log(base)
        * torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        / dim
    )
    angle = torch.arange(length, device=device, dtype=torch.float32)[:, None] * frequency[None, :]
    return angle.cos().to(dtype)[None, :, None, :], angle.sin().to(dtype)[None, :, None, :]


def _rotate(x: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    ).flatten(-2)


def apply_rope(x: Tensor, positions: Tensor, base: float = 10_000.0) -> Tensor:
    """Apply adjacent-pair RoPE to [batch,time,heads,head_dim]."""
    dim = x.shape[-1]
    if dim % 2:
        raise ValueError("RoPE requires an even head dimension")
    if positions.shape != (x.shape[1],):
        raise ValueError("positions must contain one value per token")
    cosine, sine = _rope_table(
        int(positions.max()) + 1, dim, x.device, x.dtype, base
    )
    positions = positions.to(x.device)
    return _rotate(x, cosine.index_select(1, positions), sine.index_select(1, positions))


class RotaryEmbedding(nn.Module):
    """RoPE with non-persistent cosine/sine tables cached by length and dtype."""

    def __init__(self, dim: int, base: float = 10_000.0):
        super().__init__()
        if dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        self.dim, self.base = dim, base
        self.register_buffer("_cosine", torch.empty(0), persistent=False)
        self.register_buffer("_sine", torch.empty(0), persistent=False)

    def prepare(self, length: int, reference: Tensor) -> None:
        current_length = self._cosine.shape[1] if self._cosine.ndim == 4 else 0
        if (
            current_length >= length
            and self._cosine.device == reference.device
            and self._cosine.dtype == reference.dtype
        ):
            return
        self._cosine, self._sine = _rope_table(
            length, self.dim, reference.device, reference.dtype, self.base
        )

    def forward(self, *tensors: Tensor, position: int | None = None) -> tuple[Tensor, ...]:
        if not tensors:
            return ()
        if any(x.shape != tensors[0].shape for x in tensors):
            raise ValueError("all RoPE tensors must have the same shape")
        length = tensors[0].shape[1] if position is None else position + 1
        self.prepare(length, tensors[0])
        if position is None:
            cosine = self._cosine[:, : tensors[0].shape[1]]
            sine = self._sine[:, : tensors[0].shape[1]]
        else:
            cosine = self._cosine[:, position : position + 1]
            sine = self._sine[:, position : position + 1]
        return tuple(_rotate(x, cosine, sine) for x in tensors)


class FutureAwareKVAttention(nn.Module):
    """Nine-projection hybrid soft-window FA-KV attention mixer.

    F uses RoPE and a learned soft past/future window. C uses ALiBi. Heads
    remain independent inside F and C; only ``output`` mixes heads.
    """

    def __init__(
        self,
        width: int,
        heads: int,
        head_dim: int | None = None,
        *,
        gate_init: float = 0.5,
        rope_base: float = 10_000.0,
        alibi_scale: float = 1 / 64,
        threshold_past_init_fraction: float = -0.05,
        threshold_future_init_fraction: float = 0.25,
        threshold_reference_length: int = 128,
        threshold_temperature: float = 4.0,
    ):
        super().__init__()
        if min(width, heads) < 1:
            raise ValueError("width and heads must be positive")
        if head_dim is None:
            if width % heads:
                raise ValueError("width must be divisible by heads when head_dim is omitted")
            head_dim = width // heads
        if head_dim < 1:
            raise ValueError("head_dim must be positive")
        if head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        if not math.isfinite(rope_base) or rope_base <= 0:
            raise ValueError("RoPE base must be positive")
        if not math.isfinite(alibi_scale) or alibi_scale < 0:
            raise ValueError("ALiBi scale must be nonnegative")
        if threshold_reference_length < 2:
            raise ValueError("threshold reference length must be at least two")
        if not math.isfinite(threshold_temperature) or threshold_temperature <= 0:
            raise ValueError("threshold temperature must be positive")
        self.width, self.heads, self.head_dim = width, heads, head_dim
        self.rope_base = rope_base
        self.alibi_scale = alibi_scale
        self.threshold_temperature = threshold_temperature
        self.hard_window = False
        self.rotary = RotaryEmbedding(head_dim, rope_base)
        slopes = torch.pow(2.0, -8 * torch.arange(1, heads + 1) / heads)
        self.register_buffer("_alibi_slopes", slopes, persistent=False)
        inner = heads * head_dim
        for name in PROJECTION_NAMES:
            layer = nn.Linear(width, inner, bias=False)
            nn.init.xavier_normal_(layer.weight)
            setattr(self, name, layer)
        self.output = nn.Linear(inner, width, bias=False)
        nn.init.xavier_normal_(self.output.weight)
        self.alpha_k = _gate_parameter(gate_init, heads)
        self.alpha_v = _gate_parameter(gate_init, heads)
        self.route_past_threshold_raw = _threshold_parameter(
            threshold_past_init_fraction, threshold_reference_length, heads
        )
        self.route_future_threshold_raw = _threshold_parameter(
            threshold_future_init_fraction, threshold_reference_length, heads
        )

    @property
    def gate_k(self) -> Tensor:
        return torch.exp(-torch.exp(self.alpha_k))

    @property
    def gate_v(self) -> Tensor:
        return torch.exp(-torch.exp(self.alpha_v))

    def window_thresholds(self, length: int) -> tuple[Tensor, Tensor]:
        """Return learned past and future distances for a sequence length."""
        scale = length + 20
        past = -10 + scale * torch.sigmoid(self.route_past_threshold_raw)
        future = -10 + scale * torch.sigmoid(self.route_future_threshold_raw)
        return past, future

    def project(self, x: Tensor, position: int | None = None) -> tuple[Tensor, ...]:
        shape = (*x.shape[:2], self.heads, self.head_dim)
        projected = tuple(
            getattr(self, name)(x).reshape(shape)
            for name in PROJECTION_NAMES
        )
        # Only F routing carries RoPE. KCF remains in a shared frame; C uses
        # ALiBi based on target and source positions.
        projected = (*self.rotary(*projected[:4], position=position), *projected[4:])
        return projected

    def forward(self, x: Tensor) -> Tensor:
        past_threshold, future_threshold = self.window_thresholds(x.shape[1])
        y = future_aware_kv_reference(
            *self.project(x),
            self.gate_k,
            self.gate_v,
            outer_alibi_slopes=self._alibi_slopes * self.alibi_scale,
            route_past_threshold=past_threshold,
            route_future_threshold=future_threshold,
            route_temperature=self.threshold_temperature,
            route_hard_window=self.hard_window,
        )
        return self.output(y.flatten(-2))

    def forward_queries(self, body: Tensor, queries: Tensor) -> Tensor:
        """Batch independent final queries for the many-to-one experiment."""
        if self.rotary is not None:
            self.rotary.prepare(body.shape[1] + 1, body)
        past_threshold, future_threshold = self.window_thresholds(body.shape[1] + 1)
        y = _future_aware_kv_queries(
            self.project(body),
            self.project(queries, position=body.shape[1]),
            self.gate_k,
            self.gate_v,
            outer_alibi_slopes=self._alibi_slopes * self.alibi_scale,
            route_past_threshold=past_threshold,
            route_future_threshold=future_threshold,
            route_temperature=self.threshold_temperature,
            route_hard_window=self.hard_window,
        )
        return self.output(y.flatten(-2))
