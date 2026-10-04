"""Readable definitions of future-aware KV attention."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


PROJECTION_NAMES = (
    "qfk", "kfk", "qfv", "kfv", "qc", "kc0", "kcf", "vc0", "vcf",
)


def _alibi_bias(
    slopes: Tensor,
    query_positions: Tensor,
    key_positions: Tensor,
    *,
    absolute: bool,
    dtype: torch.dtype,
) -> Tensor:
    """Return [1,heads,query,key] additive relative-position biases."""
    distance = query_positions[:, None] - key_positions[None, :]
    if absolute:
        distance = distance.abs()
    return -slopes.to(device=query_positions.device, dtype=dtype).reshape(1, -1, 1, 1) * distance.to(dtype)[None, None]


def _window_log_prior(
    past_threshold: Tensor,
    future_threshold: Tensor,
    source_positions: Tensor,
    input_positions: Tensor,
    temperature: float,
    *,
    hard: bool,
    dtype: torch.dtype,
) -> Tensor:
    """Return a stable log prior for a per-head past/future window."""
    if temperature <= 0:
        raise ValueError("window temperature must be positive")
    past_threshold = past_threshold.to(
        device=source_positions.device, dtype=dtype
    ).reshape(1, -1, 1, 1)
    future_threshold = future_threshold.to(
        device=source_positions.device, dtype=dtype
    ).reshape(1, -1, 1, 1)
    difference = input_positions[None, :] - source_positions[:, None]
    past = (difference < 0)[None, None]
    future = (difference > 0)[None, None]
    past_distance = (-difference).clamp_min(0).to(dtype)[None, None]
    future_distance = difference.clamp_min(0).to(dtype)[None, None]
    if hard:
        allowed = (
            (difference == 0)[None, None]
            | (past & (past_distance <= torch.floor(past_threshold)))
            | (future & (future_distance <= torch.floor(future_threshold)))
        )
        return torch.where(
            allowed,
            torch.zeros((), device=source_positions.device, dtype=dtype),
            torch.full((), -torch.inf, device=source_positions.device, dtype=dtype),
        )
    past_log = F.logsigmoid((past_threshold - past_distance) / temperature)
    future_log = F.logsigmoid((future_threshold - future_distance) / temperature)
    return torch.where(
        past,
        past_log,
        torch.where(future, future_log, torch.zeros_like(past_log)),
    )


def _check(projections: tuple[Tensor, ...]) -> tuple[int, int, int, int]:
    if len(projections) != 9:
        raise ValueError("expected QFK,KFK,QFV,KFV,QC,KC0,KCF,VC0,VCF")
    if any(x.ndim != 4 for x in projections):
        raise ValueError("every projection must have shape [batch,time,heads,dim]")
    if any(x.shape != projections[0].shape for x in projections[1:]):
        raise ValueError("all projections must have the same shape")
    return projections[0].shape


def future_aware_kv_reference(
    qfk: Tensor,
    kfk: Tensor,
    qfv: Tensor,
    kfv: Tensor,
    qc: Tensor,
    kc0: Tensor,
    kcf: Tensor,
    vc0: Tensor,
    vcf: Tensor,
    gate_k: Tensor,
    gate_v: Tensor,
    *,
    outer_alibi_slopes: Tensor | None = None,
    route_past_threshold: Tensor | None = None,
    route_future_threshold: Tensor | None = None,
    route_temperature: float = 1.0,
    route_hard_window: bool = False,
) -> Tensor:
    """Dense cubic matrix definition, with projections shaped [B,T,H,D].

    It computes every causal output. The large contractions are T-by-T matrix
    multiplications, so this is concise and often practical at experiment-sized
    T, but its arithmetic is O(T^3) and its scalar workspace is O(T^2).
    """
    projections = (qfk, kfk, qfv, kfv, qc, kc0, kcf, vc0, vcf)
    _, time, heads, dim = _check(projections)
    dtype = torch.float64 if qfk.device.type == "cpu" else torch.float32
    qfk, kfk, qfv, kfv, qc, kc0, kcf, vc0, vcf = (
        x.transpose(1, 2).to(dtype) for x in projections
    )
    gk = gate_k.to(dtype).reshape(1, heads, 1, 1)
    gv = gate_v.to(dtype).reshape(1, heads, 1, 1)
    causal = torch.ones(time, time, dtype=torch.bool, device=qc.device).tril()
    lower = causal.to(dtype)
    scale = dim**-0.5

    positions = torch.arange(time, device=qc.device)
    outer_bias = None if outer_alibi_slopes is None else _alibi_bias(
        outer_alibi_slopes, positions, positions, absolute=False, dtype=dtype
    )
    window_log_prior = None
    if route_past_threshold is not None or route_future_threshold is not None:
        if route_past_threshold is None or route_future_threshold is None:
            raise ValueError("both window thresholds are required")
        window_log_prior = _window_log_prior(
            route_past_threshold,
            route_future_threshold,
            positions,
            positions,
            route_temperature,
            hard=route_hard_window,
            dtype=dtype,
        )

    def unnormalized(
        q: Tensor,
        k: Tensor,
    ) -> tuple[Tensor, Tensor]:
        score = q @ k.transpose(-1, -2) * scale  # source,input
        if window_log_prior is not None:
            score = score + window_log_prior
        # A source first participates when target >= source. This fixed row
        # shift preserves the matrix factorization and is exact algebraically.
        shift = score.masked_fill(~causal, -torch.inf).amax(-1, keepdim=True).detach()
        e = torch.exp(score - shift)
        prefix_denominator = e.cumsum(-1).transpose(-1, -2)  # target,source
        return e, prefix_denominator

    ek, hk = unnormalized(qfk, kfk)
    ev, hv = unnormalized(qfv, kfv)

    base_logits = qc @ kc0.transpose(-1, -2) * scale
    payload_logits = qc @ kcf.transpose(-1, -2) * scale
    transported_logits = (payload_logits * lower) @ ek.transpose(-1, -2)
    # A hard past window can make pre-source prefix denominators exactly zero.
    # Those entries are outside the outer causal mask, but must still avoid
    # producing 0/0 before the mask is applied.
    safe_hk = hk.masked_fill(~causal, 1)
    safe_hv = hv.masked_fill(~causal, 1)
    logits = gk * base_logits + (1 - gk) * transported_logits / safe_hk
    if outer_bias is not None:
        logits = logits + outer_bias
    outer = logits.masked_fill(~causal, -torch.inf).softmax(-1)

    transported_value_weights = ((outer / safe_hv) @ ev) * lower
    output = gv * (outer @ vc0) + (1 - gv) * (transported_value_weights @ vcf)
    return output.transpose(1, 2).to(projections[0].dtype)


def _future_aware_kv_queries(
    body: tuple[Tensor, ...],
    queries: tuple[Tensor, ...],
    gate_k: Tensor,
    gate_v: Tensor,
    *,
    outer_alibi_slopes: Tensor | None = None,
    route_past_threshold: Tensor | None = None,
    route_future_threshold: Tensor | None = None,
    route_temperature: float = 1.0,
    route_hard_window: bool = False,
) -> Tensor:
    """Experiment helper for independent ``body + one query`` sequences.

    ``body`` is nine [B,T,H,D] tensors and ``queries`` is nine [B,Q,H,D]
    tensors. Each query is a separate virtual final token and cannot observe
    the other queries. This is exactly equivalent to Q calls to the cubic
    definition and returns only their final outputs, using O(T^2 D + Q T D)
    work instead of materializing unused body outputs Q times.
    """
    _, _, heads, dim = _check(body)
    _check(queries)
    if any(x.shape[0] != body[0].shape[0] or x.shape[2:] != body[0].shape[2:]
           for x in queries):
        raise ValueError("body and query batch/head dimensions must agree")
    original_dtype = body[0].dtype
    dtype = torch.float64 if body[0].device.type == "cpu" else torch.float32
    b = tuple(x.transpose(1, 2).to(dtype) for x in body)
    z = tuple(x.transpose(1, 2).to(dtype) for x in queries)
    bqk, bkk, bqv, bkv, _, bkc0, bkcf, bvc0, bvcf = b
    zqk, zkk, zqv, zkv, zqc, zkc0, zkcf, zvc0, zvcf = z
    scale = dim**-0.5
    gk = gate_k.to(dtype).reshape(1, heads, 1, 1)
    gv = gate_v.to(dtype).reshape(1, heads, 1, 1)
    time = body[0].shape[1]
    body_positions = torch.arange(time, device=body[0].device)
    query_position = torch.tensor([time], device=body[0].device)
    outer_final_bias = None if outer_alibi_slopes is None else _alibi_bias(
        outer_alibi_slopes,
        query_position,
        body_positions,
        absolute=False,
        dtype=dtype,
    )
    body_window_log_prior = None
    body_to_query_window_log_prior = None
    query_to_body_window_log_prior = None
    if route_past_threshold is not None or route_future_threshold is not None:
        if route_past_threshold is None or route_future_threshold is None:
            raise ValueError("both window thresholds are required")
        window_kwargs = {
            "temperature": route_temperature,
            "hard": route_hard_window,
            "dtype": dtype,
        }
        body_window_log_prior = _window_log_prior(
            route_past_threshold,
            route_future_threshold,
            body_positions,
            body_positions,
            **window_kwargs,
        )
        body_to_query_window_log_prior = _window_log_prior(
            route_past_threshold,
            route_future_threshold,
            body_positions,
            query_position,
            **window_kwargs,
        ).transpose(-1, -2)
        query_to_body_window_log_prior = _window_log_prior(
            route_past_threshold,
            route_future_threshold,
            query_position,
            body_positions,
            **window_kwargs,
        )

    def update_body(
        body_q: Tensor,
        body_k: Tensor,
        query_k: Tensor,
        payload: Tensor,
    ) -> tuple[Tensor, Tensor]:
        score = body_q @ body_k.transpose(-1, -2) * scale
        if body_window_log_prior is not None:
            score = score + body_window_log_prior
        maximum = score.amax(-1, keepdim=True).detach()
        weight = torch.exp(score - maximum)
        denominator = weight.sum(-1)
        context = (weight @ payload) / denominator.unsqueeze(-1)

        new_score = query_k @ body_q.transpose(-1, -2) * scale
        if body_to_query_window_log_prior is not None:
            new_score = new_score + body_to_query_window_log_prior
        next_maximum = torch.maximum(maximum.transpose(-1, -2), new_score)
        old_mass = denominator.unsqueeze(-2) * torch.exp(
            maximum.transpose(-1, -2) - next_maximum
        )
        new_mass = torch.exp(new_score - next_maximum)
        return context, new_mass / (old_mass + new_mass)

    def query_source_context(
        query_q: Tensor,
        body_k: Tensor,
        query_k: Tensor,
        body_payload: Tensor,
        query_payload: Tensor,
    ) -> Tensor:
        score = query_q @ body_k.transpose(-1, -2) * scale
        self_score = (query_q * query_k).sum(-1, keepdim=True) * scale
        if query_to_body_window_log_prior is not None:
            score = score + query_to_body_window_log_prior
        probability = torch.cat((score, self_score), -1).softmax(-1)
        return probability[..., :-1] @ body_payload + probability[..., -1:] * query_payload

    body_zk, fraction_k = update_body(bqk, bkk, zkk, bkcf)
    body_zv, fraction_v = update_body(bqv, bkv, zkv, bvcf)
    query_zk = query_source_context(zqk, bkk, zkk, bkcf, zkcf)
    query_zv = query_source_context(zqv, bkv, zkv, bvcf, zvcf)

    base_logits = zqc @ bkc0.transpose(-1, -2) * scale
    old_context_logits = zqc @ body_zk.transpose(-1, -2) * scale
    new_payload_logits = (zqc * zkcf).sum(-1, keepdim=True) * scale
    body_logits = gk * base_logits + (1 - gk) * (
        (1 - fraction_k) * old_context_logits + fraction_k * new_payload_logits
    )
    if outer_final_bias is not None:
        body_logits = body_logits + outer_final_bias
    query_key = gk * zkc0 + (1 - gk) * query_zk
    self_logit = (zqc * query_key).sum(-1, keepdim=True) * scale
    outer = torch.cat((body_logits, self_logit), -1).softmax(-1)
    body_weight, self_weight = outer[..., :-1], outer[..., -1:]

    base_value = body_weight @ bvc0 + self_weight * zvc0
    contextual_value = (body_weight * (1 - fraction_v)) @ body_zv
    contextual_value += (body_weight * fraction_v).sum(-1, keepdim=True) * zvcf
    contextual_value += self_weight * query_zv
    output = gv * base_value + (1 - gv) * contextual_value
    return output.transpose(1, 2).to(original_dtype)
