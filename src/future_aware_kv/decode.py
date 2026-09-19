"""Inference-only online decode; PyTorch, not a benchmarked Triton decode."""

from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from ._common import check_inputs


@dataclass(frozen=True)
class DecodeState:
    qf: Tensor
    kf: Tensor
    kc: Tensor
    vc: Tensor
    maximum: Tensor
    denominator: Tensor
    zk: Tensor
    zv: Tensor

    @property
    def length(self):
        return self.qf.shape[1]


@torch.no_grad()
def decode_step(
    qf: Tensor, kf: Tensor, qc: Tensor, kc: Tensor, vc: Tensor, state: DecodeState | None = None
):
    """One [B,1,H,D] projected token -> (output, new immutable cache).

    New qF first catches up against earlier keys; all live queries then add
    the new input exactly once. Work/cache O(tD) per step, no pair histories.
    Q/K must already have their absolute-position RoPE and configured gains.
    """
    inputs = (qf, kf, qc, kc, vc)
    b, t, h, d = check_inputs(inputs)
    if t != 1:
        raise ValueError("decode_step accepts exactly one token")
    dtype = torch.float64 if qf.dtype == torch.float64 else torch.float32
    if state is not None and (
        state.qf.shape[0] != b
        or state.qf.shape[2:] != (h, d)
        or state.qf.device != qf.device
        or state.qf.dtype != dtype
    ):
        raise ValueError("DecodeState must match batch, heads, device and accumulation dtype")
    with torch.autocast(device_type=qf.device.type, enabled=False):
        qf, kf, qc, kc, vc = (x.detach().to(dtype) for x in inputs)
        scale = d**-0.5
        if state is None:
            queries, keys, kvalues, vvalues = qf, kf, kc, vc
            m = torch.full((b, 1, h), -float("inf"), device=qf.device, dtype=dtype)
            l = torch.zeros_like(m)
            zk = torch.zeros_like(kc)
            zv = torch.zeros_like(vc)
        else:
            # Initialize the newborn source against 0..t-1. Merely adding
            # token t to an empty new slot would miss its earlier context.
            scores = (qf * state.kf).sum(-1) * scale
            m0 = scores.max(1, keepdim=True).values
            e = (scores - m0).exp()
            l0 = e.sum(1, keepdim=True)
            zkn = (e.unsqueeze(-1) * state.kc).sum(1, keepdim=True) / l0.unsqueeze(-1)
            zvn = (e.unsqueeze(-1) * state.vc).sum(1, keepdim=True) / l0.unsqueeze(-1)
            queries = torch.cat((state.qf, qf), 1)
            keys = torch.cat((state.kf, kf), 1)
            kvalues = torch.cat((state.kc, kc), 1)
            vvalues = torch.cat((state.vc, vc), 1)
            m = torch.cat((state.maximum, m0), 1)
            l = torch.cat((state.denominator, l0), 1)
            zk = torch.cat((state.zk, zkn), 1)
            zv = torch.cat((state.zv, zvn), 1)
        scores = (queries * kf).sum(-1) * scale
        m1 = torch.maximum(m, scores)
        old = torch.where(l > 0, l * (m - m1).exp(), 0.0)
        beta = (scores - m1).exp()
        l1 = old + beta
        p = (beta / l1).unsqueeze(-1)
        zk = zk + p * (kc - zk)
        zv = zv + p * (vc - zv)
        c = ((kvalues + zk) * qc).sum(-1).mul(scale).softmax(1)
        y = (c.unsqueeze(-1) * (vvalues + zv)).sum(1, keepdim=True)
        return y.to(inputs[-1].dtype), DecodeState(queries, keys, kvalues, vvalues, m1, l1, zk, zv)
