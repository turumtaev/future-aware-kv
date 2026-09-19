"""T<=128 scalar-factorized reversal backend (FP32 scalar math)."""

from __future__ import annotations
import math
import torch
from torch.autograd.function import once_differentiable
from ._common import triton, tl

if triton is not None:

    @triton.jit
    def _coefficients(A, G, ALPHA, C, M, IL, T, N: tl.constexpr):
        t, bh = tl.program_id(0), tl.program_id(1)
        s, j = tl.arange(0, N), tl.arange(0, N)
        base = bh * T * T
        a = tl.load(A + base + s[:, None] * T + j[None, :], (s[:, None] < T) & (j[None, :] <= t), 0)
        scores = tl.where(j[None, :] <= t, a, -float("inf"))
        maximum = tl.max(scores, 1)
        e = tl.exp(scores - maximum[:, None])
        inv_l = 1.0 / tl.sum(e, 1)
        p = e * inv_l[:, None]
        g = tl.load(G + base + t * T + j, j <= t, 0)
        logits = g + tl.sum(p * g[None, :], 1)
        logits = tl.where(s <= t, logits, -float("inf"))
        c = tl.exp(logits - tl.max(logits, 0))
        c = c / tl.sum(c, 0)
        alpha = c + tl.sum(c[:, None] * p, 0)
        out = base + t * T + s
        tl.store(ALPHA + out, alpha, s < T)
        tl.store(C + out, c, s < T)
        tl.store(M + out, maximum, s < T)
        tl.store(IL + out, inv_l, s < T)

    @triton.jit
    def _boundary_backward(A, G, C, M, IL, D_ALPHA, D_G, DB, MEAN, T, N: tl.constexpr):
        t, bh = tl.program_id(0), tl.program_id(1)
        s, j = tl.arange(0, N), tl.arange(0, N)
        base = bh * T * T
        row = base + t * T + s
        maximum = tl.load(M + row, s < T, 0)
        inv_l = tl.load(IL + row, s < T, 0)
        a = tl.load(A + base + s[:, None] * T + j[None, :], (s[:, None] < T) & (j[None, :] <= t), 0)
        valid = (s[:, None] < T) & (j[None, :] <= t)
        p = tl.exp(tl.where(valid, a - maximum[:, None], -float("inf"))) * inv_l[:, None]
        c = tl.load(C + row, s <= t, 0)
        da = tl.load(D_ALPHA + row, s <= t, 0)
        g = tl.load(G + row, s <= t, 0)
        dc = da + tl.sum(p * da[None, :], 1)
        db = c * (dc - tl.sum(c * dc, 0))
        dp = c[:, None] * da[None, :] + db[:, None] * g[None, :]
        mean = tl.sum(p * dp, 1)
        dg = db + tl.sum(db[:, None] * p, 0)
        tl.store(D_G + row, dg, s < T)
        tl.store(DB + row, db, s < T)
        tl.store(MEAN + row, mean, s < T)

    @triton.jit
    def _routing_backward(A, G, C, M, IL, D_ALPHA, DB, MEAN, D_A, T, N: tl.constexpr):
        s, bh = tl.program_id(0), tl.program_id(1)
        t, j = tl.arange(0, N), tl.arange(0, N)
        base = bh * T * T
        a = tl.load(A + base + s * T + j, j < T, 0)
        state = base + t * T + s
        maximum = tl.load(M + state, t < T, 0)
        inv_l = tl.load(IL + state, t < T, 0)
        c = tl.load(C + state, t < T, 0)
        db = tl.load(DB + state, t < T, 0)
        mean = tl.load(MEAN + state, t < T, 0)
        valid = (t[:, None] < T) & (t[:, None] >= s) & (j[None, :] <= t[:, None])
        p = tl.exp(tl.where(valid, a[None, :] - maximum[:, None], -float("inf"))) * inv_l[:, None]
        matrix = base + t[:, None] * T + j[None, :]
        da = tl.load(D_ALPHA + matrix, valid, 0)
        g = tl.load(G + matrix, valid, 0)
        contribution = p * (c[:, None] * da + db[:, None] * g - mean[:, None])
        gradient = tl.sum(contribution, 0)
        tl.store(D_A + base + s * T + j, gradient, j < T)


class _CompactCoefficients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, g):
        a, g = a.contiguous(), g.contiguous()
        time = a.shape[-1]
        alpha, c, m, inv_l = [torch.empty_like(a) for _ in range(4)]
        n = triton.next_power_of_2(time)
        warps = 8 if n == 128 else 4
        _coefficients[(time, a.numel() // (time * time))](
            a, g, alpha, c, m, inv_l, time, n, num_warps=warps
        )
        ctx.save_for_backward(a, g, c, m, inv_l)
        ctx.n, ctx.warps = n, warps
        return alpha

    @staticmethod
    @once_differentiable
    def backward(ctx, d_alpha):
        a, g, c, m, inv_l = ctx.saved_tensors
        d_alpha = d_alpha.contiguous()
        d_a, d_g, db, mean = [torch.empty_like(a) for _ in range(4)]
        time = a.shape[-1]
        grid = (time, a.numel() // (time * time))
        _boundary_backward[grid](
            a, g, c, m, inv_l, d_alpha, d_g, db, mean, time, ctx.n, num_warps=ctx.warps
        )
        _routing_backward[grid](
            a, g, c, m, inv_l, d_alpha, db, mean, d_a, time, ctx.n, num_warps=ctx.warps
        )
        return d_a, d_g


def run(inputs):
    qf, kf, qc, kc, vc = inputs
    with torch.autocast(device_type="cuda", enabled=False):
        qf, kf, qc, kc, values = (x.transpose(1, 2).float() for x in inputs)
        scale = vc.shape[-1] ** -0.5
        a = qf @ kf.transpose(-1, -2) * scale
        g = qc @ kc.transpose(-1, -2) * scale
        weights = _CompactCoefficients.apply(a, g)
        return (weights @ values).transpose(1, 2).to(vc.dtype)
