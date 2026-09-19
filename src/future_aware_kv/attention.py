"""Five projections in, future-aware causal attention out."""

from __future__ import annotations
import torch
from torch import Tensor
from ._common import check_inputs


def reference_attention(qf: Tensor, kf: Tensor, qc: Tensor, kc: Tensor, vc: Tensor) -> Tensor:
    """Readable scalar-factorized oracle; expensive at long T, full autograd."""
    check_inputs((qf, kf, qc, kc, vc))
    dtype = torch.float64 if qf.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=qf.device.type, enabled=False):
        qf, kf, qc, kc, v = (x.transpose(1, 2).to(dtype) for x in (qf, kf, qc, kc, vc))
        scale = qf.shape[-1] ** -0.5
        a = qf @ kf.transpose(-1, -2) * scale
        g = qc @ kc.transpose(-1, -2) * scale
        ys = []
        for t in range(qf.shape[2]):
            p = a[..., : t + 1, : t + 1].softmax(-1)
            row = g[..., t, : t + 1]
            c = (row + (p @ row.unsqueeze(-1)).squeeze(-1)).softmax(-1).unsqueeze(-2)
            ys.append(((c + c @ p) @ v[..., : t + 1, :]).squeeze(-2))
        return torch.stack(ys, 2).transpose(1, 2).to(vc.dtype)


def attention(
    qf: Tensor, kf: Tensor, qc: Tensor, kc: Tensor, vc: Tensor, *, backend="auto", schedule="eager"
) -> Tensor:
    """[B,T,H,D] projections; rotate QF/KF/QC/KC before calling, never VC.

    auto: CPU reference, CUDA small kernel for T<=128, otherwise BF16 long.
    schedule=eager: bounded suffix backward (the measured LM deployment).
    schedule=graph: checkpointed backward (fastest measured CUDA-graph path).
    First derivatives only on Triton. No masks, dropout, GQA or ragged batches.
    """
    inputs = (qf, kf, qc, kc, vc)
    b, t, h, d = check_inputs(inputs)
    if backend not in ("auto", "reference", "triton"):
        raise ValueError("backend must be auto, reference or triton")
    if schedule not in ("eager", "graph"):
        raise ValueError("schedule must be eager or graph")
    if backend == "reference" or (backend == "auto" and not qf.is_cuda):
        return reference_attention(*inputs)
    from ._common import triton

    if triton is None or not qf.is_cuda:
        raise RuntimeError("Triton backend requires CUDA and Triton")
    if t <= 128:
        if qf.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("small Triton supports FP32, FP16, BF16")
        from ._small import run
    else:
        if qf.dtype != torch.bfloat16 or d > 128 or d % 2:
            raise ValueError("long Triton supports BF16 and even head width <=128")
        inputs = tuple(x.contiguous() for x in inputs)
        if not torch.is_grad_enabled() or not any(x.requires_grad for x in inputs):
            from ._forward import run as forward

            # Winner's sequential microbatching bounds inference scratch.
            if b == 1:
                return forward(inputs, training=False)[0]
            return torch.cat(
                [forward(tuple(x[i : i + 1] for x in inputs), training=False)[0] for i in range(b)],
                0,
            )
        if schedule == "graph":
            from ._checkpoint import run
        else:
            from ._suffix import run
    return run(inputs)
