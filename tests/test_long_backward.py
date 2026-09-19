"""Retained checkpoint256 and W512/NB4 suffix correctness, without old ablations."""

import copy
import pytest
import torch

from future_aware_kv import attention, FutureAwareKVAttention
from .oracles import make_inputs, streaming_attention, assert_gradients

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize("schedule", ["eager", "graph"])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 129, 2, 16),
        (2, 129, 3, 32),
        (1, 255, 2, 32),
        (1, 256, 2, 32),
        (1, 257, 2, 32),
        (1, 511, 1, 16),
        (1, 512, 1, 16),
        (1, 513, 1, 16),
    ],
)
def test_public_all_projection_gradients_at_boundaries(schedule, shape):
    inputs = tuple(
        x.requires_grad_()
        for x in make_inputs(shape, device="cuda", dtype=torch.bfloat16)
    )
    refs = tuple(x.detach().float().requires_grad_() for x in inputs)
    y = attention(*inputs, schedule=schedule)
    expected = streaming_attention(refs)
    upstream = torch.randn_like(y)
    got = torch.autograd.grad(y, inputs, upstream)
    want = torch.autograd.grad(expected, refs, upstream.float())
    torch.testing.assert_close(y.float(), expected, rtol=0.01, atol=0.005)
    assert_gradients(got, want, rtol=0.02, atol=0.005)


def raw_backward(inputs, upstream, schedule):
    """Inspect FP32 accumulators before public BF16 gradient rounding."""
    if schedule == "graph":
        from future_aware_kv import _checkpoint

        y, states, y32, lc = _checkpoint._forward_run(inputs)
        grads = _checkpoint._backward_run(inputs, states, y32, lc, upstream)
    else:
        from future_aware_kv import _forward, _suffix

        y, y32, lc = _forward.run(inputs)
        grads = _suffix.backward_run(inputs, y32, lc, upstream)
    return y32, grads


@pytest.mark.parametrize("schedule", ["eager", "graph"])
@pytest.mark.parametrize(
    "case", ["normal", "rope", "gap31", "gap33", "gap128", "repeated_gap128"]
)
def test_raw_fp32_gradients_and_query_shift_invariance(schedule, case):
    inputs = make_inputs(
        (1, 257, 2, 64), device="cuda", dtype=torch.bfloat16, case=case
    )
    upstream = torch.randn_like(inputs[0])
    with torch.no_grad():
        actual, got = raw_backward(inputs, upstream, schedule)
    refs = tuple(x.double().requires_grad_() for x in inputs)
    expected = streaming_attention(refs)
    want = torch.autograd.grad(expected, refs, upstream.double())
    torch.testing.assert_close(actual, expected.float(), rtol=0.003, atol=0.001)
    assert_gradients(got, want, rtol=0.003, atol=0.001)
    if case == "repeated_gap128":
        # Each jump leaves appreciable probability only on tied maximum-score
        # keys. Changing the query scale must not create an appreciable QF
        # gradient. This caught a prior cancellation/conservation bug.
        assert got[0].abs().max() < 0.001


@pytest.mark.parametrize("schedule", ["eager", "graph"])
@pytest.mark.parametrize("cutoff", [18, 257, 512])
def test_causal_gradients_stop_at_upstream_prefix(schedule, cutoff):
    inputs = tuple(
        x.requires_grad_()
        for x in make_inputs((1, 513, 1, 32), device="cuda", dtype=torch.bfloat16)
    )
    upstream = torch.randn_like(inputs[0])
    upstream[:, cutoff:] = 0
    y = attention(*inputs, schedule=schedule)
    gradients = torch.autograd.grad(y, inputs, upstream)
    for g in gradients:
        assert torch.count_nonzero(g[:, cutoff:]) == 0
        assert torch.count_nonzero(g[:, :cutoff]) > 0


@pytest.mark.parametrize("schedule", ["eager", "graph"])
@pytest.mark.parametrize("rope", [False, True])
def test_long_layer_input_and_parameter_chain(schedule, rope):
    torch.manual_seed(22)
    model = FutureAwareKVAttention(
        width=32, heads=2, rope=rope, inner_gain=0.7, outer_gain=0.35, schedule=schedule
    ).cuda()
    ref = copy.deepcopy(model)
    ref.backend = "reference"
    x = torch.randn(2, 129, 32, device="cuda", requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y, expected = model(x), ref(xr)
    upstream = torch.randn_like(y)
    got = torch.autograd.grad(y, (x, *model.parameters()), upstream)
    want = torch.autograd.grad(expected, (xr, *ref.parameters()), upstream)
    torch.testing.assert_close(y, expected, rtol=0.02, atol=0.005)
    for g, w in zip(got, want):
        assert torch.isfinite(g).all()
        torch.testing.assert_close(g, w, rtol=0.03, atol=0.01)


@pytest.mark.parametrize("schedule", ["eager", "graph"])
def test_noncontiguous_inputs_and_upstream(schedule):
    inputs = make_inputs((2, 129, 2, 48), device="cuda", dtype=torch.bfloat16)
    inputs = tuple(x[..., ::2].detach().requires_grad_() for x in inputs)
    assert all(not x.is_contiguous() for x in inputs)
    upstream = torch.randn(2, 129, 2, 48, device="cuda", dtype=torch.bfloat16)[..., ::2]
    refs = tuple(x.detach().float().requires_grad_() for x in inputs)
    y = attention(*inputs, schedule=schedule)
    expected = streaming_attention(refs)
    got = torch.autograd.grad(y, inputs, upstream)
    want = torch.autograd.grad(expected, refs, upstream.float())
    torch.testing.assert_close(y.float(), expected, rtol=0.01, atol=0.005)
    assert_gradients(got, want, rtol=0.02, atol=0.005)


@pytest.mark.slow
@pytest.mark.parametrize("schedule", ["eager", "graph"])
@pytest.mark.parametrize("case", ["normal", "rope", "repeated_gap128"])
def test_nanochat_shape_all_fp32_projection_gradients(schedule, case):
    # Restore the original independent T2048/H6/D128 validator as a pytest case.
    inputs = make_inputs(
        (1, 2048, 6, 128), device="cuda", dtype=torch.bfloat16, case=case, seed=67
    )
    upstream = torch.randn_like(inputs[0])
    with torch.no_grad():
        actual, got = raw_backward(inputs, upstream, schedule)
    refs = tuple(x.float().requires_grad_() for x in inputs)
    expected = streaming_attention(refs, checkpoint=32)
    want = torch.autograd.grad(expected, refs, upstream.float())
    torch.testing.assert_close(actual, expected, rtol=0.003, atol=0.001)
    assert_gradients(got, want, rtol=0.003, atol=0.001)
