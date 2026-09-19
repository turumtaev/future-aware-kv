"""Restore materialized/streaming oracle and gain regressions on CPU."""

import pytest
import torch

from future_aware_kv import reference_attention
from .oracles import make_inputs, materialized_attention, streaming_attention


@pytest.mark.parametrize("time", [1, 9, 17])
@pytest.mark.parametrize("gains", [(1.0, 1.0), (2.0, 4.0)])
def test_reference_outputs_and_all_gradients_with_query_gains(time, gains):
    leaves = make_inputs((2, time, 2, 4))
    leaves = tuple(x.requires_grad_() for x in leaves)
    qf, kf, qc, kc, vc = leaves
    projected = (qf * gains[0], kf, qc * gains[1], kc, vc)
    y = reference_attention(*projected)
    expected = materialized_attention(projected)
    torch.testing.assert_close(y, expected, rtol=1e-12, atol=1e-12)
    upstream = torch.randn_like(y)
    # Both oracles share the gain multiplications; keep those nodes for the
    # second backward while comparing gradients of the original projections.
    got = torch.autograd.grad(y, leaves, upstream, retain_graph=True)
    want = torch.autograd.grad(expected, leaves, upstream)
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("time,checkpoint", [(1, 1), (9, 4), (17, 8), (33, 16)])
@pytest.mark.parametrize("case", ["normal", "rope"])
def test_independent_streaming_oracle(time, checkpoint, case):
    inputs = tuple(x.requires_grad_() for x in make_inputs((2, time, 2, 8), case=case))
    y = streaming_attention(inputs, checkpoint=checkpoint)
    expected = materialized_attention(inputs)
    upstream = torch.randn_like(y)
    torch.testing.assert_close(y, expected, rtol=1e-10, atol=1e-10)
    got = torch.autograd.grad(y, inputs, upstream)
    want = torch.autograd.grad(expected, inputs, upstream)
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)
