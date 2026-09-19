"""Shared softmax scans and BF16 high/residual tensor-core contractions."""

from __future__ import annotations
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = tl = None

if triton is not None:

    @triton.jit
    def _prefix_combine(ma, la, mb, lb):
        m = tl.maximum(ma, mb)
        safe_m = tl.where((la + lb) > 0, m, 0.0)
        l = tl.where(la > 0, la * tl.exp(ma - safe_m), 0.0)
        l += tl.where(lb > 0, lb * tl.exp(mb - safe_m), 0.0)
        return m, l

    @triton.jit
    def _repeat_rows(x, BT: tl.constexpr):
        # Structural repetition pads scalar group results only, never GEMMs.
        x = tl.trans(x)
        for step in tl.static_range(0, tl.constexpr((BT // x.shape[1]).bit_length() - 1)):
            x = tl.reshape(tl.permute(tl.join(x, x), (0, 2, 1)), (x.shape[0], 2 * x.shape[1]))
        return tl.trans(x)

    @triton.jit
    def _probability_contract(
        scores,
        prefix_m,
        prefix_inverse_l,
        factor,
        BS: tl.constexpr,
        BT: tl.constexpr,
        GROUP: tl.constexpr,
        VALUE: tl.constexpr,
    ):
        # Only scalar output tiles cross the group loop. Neither Z0 nor any
        # other D-wide tensor is an input to this helper.
        NC: tl.constexpr = BS if not VALUE else BT
        NF: tl.constexpr = BT if not VALUE else BS
        result = tl.full((BT, NC), 0.0, tl.float32)
        r = tl.arange(0, BT)
        g = tl.arange(0, GROUP)
        for base in range(0, BT, GROUP):
            target = base + g
            m = tl.trans(tl.gather(prefix_m, tl.broadcast_to(target[None, :], (BS, GROUP)), 1))
            inverse_l = tl.trans(
                tl.gather(prefix_inverse_l, tl.broadcast_to(target[None, :], (BS, GROUP)), 1)
            )
            factor_group = tl.gather(factor, tl.broadcast_to(target[:, None], (GROUP, NF)), 0)
            shifted = scores[None, :, :] - m[:, :, None]
            shifted = tl.where(r[None, None, :] <= target[:, None, None], shifted, -float("inf"))
            probability = tl.exp(shifted) * inverse_l[:, :, None]
            if VALUE:
                group_result = tl.sum(factor_group[:, :, None] * probability, 1)
            else:
                group_result = tl.sum(factor_group[:, None, :] * probability, 2)
            expanded = _repeat_rows(group_result, BT)
            result = tl.where((r[:, None] >= base) & (r[:, None] < base + GROUP), expanded, result)
        return result

    @triton.jit
    def _context_dot(a, b, FAST_DOT: tl.constexpr):
        if not FAST_DOT:
            return tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="tf32x3")
        # Original BF16 projections need no decomposition. FP32 operands use
        # BF16 high/residual components. Omit only the residual*residual term
        # when both operands are FP32 (a small, explicitly tested error).
        ah = a.to(tl.bfloat16)
        bh = b.to(tl.bfloat16)
        result = tl.dot(ah, bh)
        if a.dtype != tl.bfloat16:
            al = (a - ah.to(tl.float32)).to(tl.bfloat16)
            result += tl.dot(al, bh)
        if b.dtype != tl.bfloat16:
            bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
            result += tl.dot(ah, bl)
        return result

    @triton.jit
    def _score_adjoint_contract(
        scores, pm, il, db, c, qg, hv, mean, BS: tl.constexpr, BT: tl.constexpr, GROUP: tl.constexpr
    ):
        result = tl.full((BS, BT), 0.0, tl.float32)
        r = tl.arange(0, BT)
        for base in range(0, BT, GROUP):
            t = base + tl.arange(0, GROUP)
            ix = tl.broadcast_to(t[None, :], (BS, GROUP))
            m = tl.trans(tl.gather(pm, ix, 1))
            inverse = tl.trans(tl.gather(il, ix, 1))
            ds = tl.gather(db, tl.broadcast_to(t[:, None], (GROUP, BS)), 0)
            cs = tl.gather(c, tl.broadcast_to(t[:, None], (GROUP, BS)), 0)
            ms = tl.gather(mean, tl.broadcast_to(t[:, None], (GROUP, BS)), 0)
            qs = tl.gather(qg, tl.broadcast_to(t[:, None], (GROUP, BT)), 0)
            hs = tl.gather(hv, tl.broadcast_to(t[:, None], (GROUP, BT)), 0)
            p = (
                tl.exp(
                    tl.where(
                        r[None, None, :] <= t[:, None, None],
                        scores[None, :, :] - m[:, :, None],
                        -float("inf"),
                    )
                )
                * inverse[:, :, None]
            )
            term = (
                ds[:, :, None] * qs[:, None, :] + cs[:, :, None] * hs[:, None, :] - ms[:, :, None]
            )
            result += tl.sum(p * term, 0)
        return result

    @triton.jit
    def _sum_combine(a, b):
        return a + b


__all__ = [
    "triton",
    "tl",
    "_prefix_combine",
    "_repeat_rows",
    "_probability_contract",
    "_context_dot",
    "_score_adjoint_contract",
    "_sum_combine",
]


def check_inputs(inputs):
    if len(inputs) != 5 or inputs[0].ndim != 4 or min(inputs[0].shape) < 1:
        raise ValueError("five nonempty [B,T,H,D] projections required")
    x = inputs[0]
    if not x.is_floating_point() or any(
        y.shape != x.shape or y.device != x.device or y.dtype != x.dtype for y in inputs
    ):
        raise ValueError("projections must have the same floating dtype, shape and device")
    return x.shape


def check_long(inputs):
    b, t, h, d = check_inputs(inputs)
    if triton is None or not inputs[0].is_cuda:
        raise RuntimeError("long attention requires CUDA and Triton")
    if inputs[0].dtype != torch.bfloat16 or any(not x.is_contiguous() for x in inputs):
        raise ValueError("long Triton attention requires contiguous BF16 projections")
    if d > 128 or d % 2:
        raise ValueError("long Triton attention requires even head width <=128")
    return b, t, h, d


def allocate_state(b, h, t, d, device, levels=1):
    zk = torch.empty((levels, b * h, t, d), device=device, dtype=torch.float32)
    zv = torch.empty_like(zk)
    m = torch.empty((levels, b * h, t), device=device, dtype=torch.float32)
    return zk, zv, m, torch.empty_like(m)
