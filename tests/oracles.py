"""Independent test-only oracles, adapted from the original five-vector suite.

Materialized contextual KV checks the definition. Per-token normalized updates
check the long backward without duplicating the Triton block factorization.
"""

import torch

from future_aware_kv import apply_rope

PROJECTIONS = ("QF", "KF", "QC", "KC", "VC")


def make_inputs(shape, *, device="cpu", dtype=torch.float64, case="normal", seed=19):
    generator = torch.Generator(device=device).manual_seed(seed)
    xs = [
        torch.randn(shape, generator=generator, device=device, dtype=dtype)
        for _ in range(5)
    ]
    if case == "rope":
        xs = [apply_rope(x).contiguous() if i < 4 else x for i, x in enumerate(xs)]
    elif case != "normal":
        # D=64 has sqrt(D)=8 exactly, so single-jump guard tests have exact scores.
        d = shape[-1]
        xs[0].zero_()
        xs[1].zero_()
        xs[0][..., 0] = d**0.5
        j = torch.arange(shape[1], device=device)
        if case == "repeated_gap128":
            scores = 128 * (j // 16 + (j % 16 >= 8))
        else:
            gap = {"gap31": 31, "gap33": 33, "gap128": 128}[case]
            scores = gap * (j >= 8)
        xs[1][..., 0] = scores.to(dtype)[None, :, None]
    return tuple(xs)


def materialized_attention(inputs, targets=None):
    """Explicit prefix probabilities and contextual K/V; differentiable."""
    qf, kf, qc, kc, vc = (x.transpose(1, 2) for x in inputs)
    scale = qf.shape[-1] ** -0.5
    if targets is None:
        targets = range(qf.shape[2])
    outputs = []
    for t in targets:
        p = (
            qf[..., : t + 1, :] @ kf[..., : t + 1, :].transpose(-1, -2) * scale
        ).softmax(-1)
        key = kc[..., : t + 1, :] + p @ kc[..., : t + 1, :]
        value = vc[..., : t + 1, :] + p @ vc[..., : t + 1, :]
        c = (qc[..., t : t + 1, :] @ key.transpose(-1, -2) * scale).softmax(-1)
        outputs.append((c @ value).squeeze(-2))
    return torch.stack(outputs, 2).transpose(1, 2)


def dense_final_state(inputs):
    """All source queries against all inputs, including pre-aged source rows."""
    qf, kf, _, kc, vc = (x.transpose(1, 2) for x in inputs)
    scores = qf @ kf.transpose(-1, -2) * qf.shape[-1] ** -0.5
    maximum = scores.max(-1).values
    denominator = (scores - maximum.unsqueeze(-1)).exp().sum(-1)
    probability = scores.softmax(-1)
    return probability @ kc, probability @ vc, maximum, denominator


def streaming_attention(inputs, *, checkpoint=32):
    """Quadratic stable per-token autograd oracle, with checkpointed segments.

    This is the original independent streaming validation approach, not a
    production backend. Keep it in tests so no experiment harness is needed.
    """
    from torch.utils.checkpoint import checkpoint as recompute

    qf, kf, qc, kc, vc = (x.transpose(1, 2).contiguous() for x in inputs)
    b, h, t, d = qf.shape
    maximum = torch.full((b, h, t), -float("inf"), device=qf.device, dtype=qf.dtype)
    denominator = torch.zeros_like(maximum)
    zk, zv = torch.zeros_like(kc), torch.zeros_like(vc)
    positions = torch.arange(t, device=qf.device)
    scale = d**-0.5
    results = []
    for start in range(0, t, checkpoint):
        end = min(start + checkpoint, t)

        def segment(
            maximum, denominator, zk, zv, qf, kf, qc, kc, vc, start=start, end=end
        ):
            ys = []
            for j in range(start, end):
                score = (qf @ kf[..., j : j + 1, :].transpose(-1, -2)).squeeze(
                    -1
                ) * scale
                next_max = torch.maximum(maximum, score)
                old_mass = torch.where(
                    denominator > 0, denominator * (maximum - next_max).exp(), 0.0
                )
                beta = (score - next_max).exp()
                next_den = old_mass + beta
                p = (beta / next_den).unsqueeze(-1)
                zk = zk + p * (kc[..., j : j + 1, :] - zk)
                zv = zv + p * (vc[..., j : j + 1, :] - zv)
                maximum, denominator = next_max, next_den
                logits = (
                    (qc[..., j : j + 1, :] @ kc.transpose(-1, -2))
                    + (qc[..., j : j + 1, :] @ zk.transpose(-1, -2))
                ) * scale
                c = logits.masked_fill(positions > j, -float("inf")).softmax(-1)
                ys.append((c @ vc + c @ zv).squeeze(-2))
            return maximum, denominator, zk, zv, torch.stack(ys, 2)

        args = (maximum, denominator, zk, zv, qf, kf, qc, kc, vc)
        if torch.is_grad_enabled() and any(x.requires_grad for x in args):
            maximum, denominator, zk, zv, y = recompute(
                segment, *args, use_reentrant=False
            )
        else:
            maximum, denominator, zk, zv, y = segment(*args)
        results.append(y)
    return torch.cat(results, 2).transpose(1, 2)


def assert_gradients(actual, expected, *, rtol, atol):
    assert len(actual) == len(expected) == 5
    for name, got, want in zip(PROJECTIONS, actual, expected):
        assert torch.isfinite(got).all(), name
        torch.testing.assert_close(
            got.float(), want.float(), rtol=rtol, atol=atol, msg=name
        )
