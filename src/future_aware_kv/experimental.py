"""Executable references for the matrix formulations in FUTURE_WORK.md."""

from __future__ import annotations

import torch
from torch import Tensor


def _prepare(projections: tuple[Tensor, ...]) -> tuple[tuple[Tensor, ...], int, int]:
    if len(projections) != 9 or any(x.shape != projections[0].shape for x in projections):
        raise ValueError("nine equally shaped projections are required")
    if projections[0].ndim != 4:
        raise ValueError("projections must have shape [batch,time,heads,dim]")
    _, time, _, dim = projections[0].shape
    return tuple(x.transpose(1, 2) for x in projections), time, dim


def _full_weights(query: Tensor, key: Tensor, causal: Tensor, scale: float) -> Tensor:
    score = query @ key.transpose(-1, -2) * scale
    shift = score.masked_fill(~causal, -torch.inf).amax(-1, keepdim=True).detach()
    return torch.exp(score - shift)


def materialized_prefix_attention(
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
) -> Tensor:
    """Quadratic-compute reference that stores every [source,target,dim] state."""
    original_dtype = qfk.dtype
    p, time, dim = _prepare((qfk, kfk, qfv, kfv, qc, kc0, kcf, vc0, vcf))
    qfk, kfk, qfv, kfv, qc, kc0, kcf, vc0, vcf = p
    causal = torch.ones(time, time, dtype=torch.bool, device=qfk.device).tril()
    scale = dim**-0.5
    key_weight = _full_weights(qfk, kfk, causal, scale)
    value_weight = _full_weights(qfv, kfv, causal, scale)

    key_denominator = key_weight.cumsum(-1)
    value_denominator = value_weight.cumsum(-1)
    key_numerator = (key_weight[..., None] * kcf[:, :, None, :, :]).cumsum(-2)
    value_numerator = (value_weight[..., None] * vcf[:, :, None, :, :]).cumsum(-2)

    gk = gate_k.reshape(1, -1, 1, 1, 1).to(qfk.dtype)
    gv = gate_v.reshape(1, -1, 1, 1, 1).to(qfk.dtype)
    key = gk * kc0[:, :, :, None, :] + (1 - gk) * (
        key_numerator / key_denominator[..., None]
    )
    value = gv * vc0[:, :, :, None, :] + (1 - gv) * (
        value_numerator / value_denominator[..., None]
    )

    logits = torch.einsum("bhtd,bhstd->bhts", qc, key) * scale
    probability = logits.masked_fill(~causal, -torch.inf).softmax(-1)
    output = torch.einsum("bhts,bhstd->bhtd", probability, value)
    return output.transpose(1, 2).to(original_dtype)


def _exclusive_cumsum(x: Tensor, dim: int) -> Tensor:
    return x.cumsum(dim) - x


def block_prefix_attention(
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
    block_size: int,
) -> Tensor:
    """Quadratic-compute reference with full-prefix states at block boundaries."""
    original_dtype = qfk.dtype
    p, time, dim = _prepare((qfk, kfk, qfv, kfv, qc, kc0, kcf, vc0, vcf))
    qfk, kfk, qfv, kfv, qc, kc0, kcf, vc0, vcf = p
    if block_size < 1 or time % block_size:
        raise ValueError("block_size must be positive and divide sequence length")
    blocks = time // block_size
    causal = torch.ones(time, time, dtype=torch.bool, device=qfk.device).tril()
    scale = dim**-0.5
    key_weight = _full_weights(qfk, kfk, causal, scale)
    value_weight = _full_weights(qfv, kfv, causal, scale)

    def into_blocks(weight: Tensor) -> Tensor:
        return weight.view(*weight.shape[:3], blocks, block_size).permute(0, 1, 3, 4, 2)

    key_weight_blocks = into_blocks(key_weight)       # [batch,head,block,input,source]
    value_weight_blocks = into_blocks(value_weight)
    query_blocks = qc.view(qc.shape[0], qc.shape[1], blocks, block_size, dim)
    key_payload_blocks = kcf.view(kcf.shape[0], kcf.shape[1], blocks, block_size, dim)
    value_payload_blocks = vcf.view(vcf.shape[0], vcf.shape[1], blocks, block_size, dim)

    key_block_denominator = key_weight_blocks.sum(3)
    value_block_denominator = value_weight_blocks.sum(3)
    key_block_numerator = torch.einsum(
        "bhnjs,bhnjd->bhnsd", key_weight_blocks, key_payload_blocks
    )
    value_block_numerator = torch.einsum(
        "bhnjs,bhnjd->bhnsd", value_weight_blocks, value_payload_blocks
    )
    key_denominator_before = _exclusive_cumsum(key_block_denominator, 2)
    value_denominator_before = _exclusive_cumsum(value_block_denominator, 2)
    key_numerator_before = _exclusive_cumsum(key_block_numerator, 2)
    value_numerator_before = _exclusive_cumsum(value_block_numerator, 2)

    key_denominator = key_denominator_before[:, :, :, None, :] + key_weight_blocks.cumsum(3)
    value_denominator = (
        value_denominator_before[:, :, :, None, :] + value_weight_blocks.cumsum(3)
    )
    key_logit_past = torch.einsum(
        "bhnrd,bhnsd->bhnrs", query_blocks, key_numerator_before
    ) * scale
    local_key_geometry = torch.einsum(
        "bhnrd,bhnjd->bhnrj", query_blocks, key_payload_blocks
    ) * scale
    local_causal = torch.ones(
        block_size, block_size, dtype=torch.bool, device=qfk.device
    ).tril()
    local_key_geometry = local_key_geometry.masked_fill(~local_causal, 0)
    key_logit_local = torch.einsum(
        "bhnrj,bhnjs->bhnrs", local_key_geometry, key_weight_blocks
    )
    shortcut_key_logit = torch.einsum("bhnrd,bhsd->bhnrs", query_blocks, kc0) * scale

    gk = gate_k.reshape(1, -1, 1, 1, 1).to(qfk.dtype)
    gv = gate_v.reshape(1, -1, 1, 1, 1).to(qfk.dtype)
    logits = gk * shortcut_key_logit + (1 - gk) * (
        key_logit_past + key_logit_local
    ) / key_denominator
    target_position = torch.arange(time, device=qfk.device).view(blocks, block_size)
    source_position = torch.arange(time, device=qfk.device)
    outer_causal = source_position[None, None, :] <= target_position[:, :, None]
    probability = logits.masked_fill(~outer_causal, -torch.inf).softmax(-1)

    value_source_weight = probability / value_denominator
    value_past = torch.einsum(
        "bhnrs,bhnsd->bhnrd", value_source_weight, value_numerator_before
    )
    value_local_weight = torch.einsum(
        "bhnrs,bhnjs->bhnrj", value_source_weight, value_weight_blocks
    ).masked_fill(~local_causal, 0)
    value_local = torch.einsum(
        "bhnrj,bhnjd->bhnrd", value_local_weight, value_payload_blocks
    )
    shortcut_value = torch.einsum("bhnrs,bhsd->bhnrd", probability, vc0)
    output = gv * shortcut_value + (1 - gv) * (value_past + value_local)
    return output.reshape(output.shape[0], output.shape[1], time, dim).transpose(1, 2).to(
        original_dtype
    )
