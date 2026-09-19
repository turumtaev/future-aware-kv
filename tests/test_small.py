"""Small kernel coverage restored from the original five-vector regression tests."""

import copy
import pytest
import torch

from future_aware_kv import attention, FutureAwareKVAttention
from .oracles import make_inputs, materialized_attention

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize("time", [1, 9, 17, 31, 32, 33, 64, 127, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_small_outputs_all_gradients_and_causality(time, dtype):
    inputs = list(make_inputs((2, time, 2, 16), device="cuda", dtype=dtype))
    inputs[0] = (inputs[0] * 2).contiguous()
    inputs[2] = (inputs[2] * 4).contiguous()
    inputs = tuple(x.requires_grad_() for x in inputs)
    y = attention(*inputs, backend="triton")
    refs = tuple(x.detach().double().requires_grad_() for x in inputs)
    expected = materialized_attention(refs)
    upstream = torch.randn_like(y)
    tol = {
        torch.float32: (1e-3, 1e-4),
        torch.float16: (0.02, 0.005),
        torch.bfloat16: (0.03, 0.04),
    }[dtype]
    torch.testing.assert_close(y.float(), expected.float(), rtol=tol[0], atol=tol[1])
    got = torch.autograd.grad(y, inputs, upstream, retain_graph=True)
    want = torch.autograd.grad(expected, refs, upstream.double())
    for a, b in zip(got, want):
        torch.testing.assert_close(a.float(), b.float(), rtol=tol[0], atol=tol[1])
    cutoff = max(1, time // 2)
    grads = torch.autograd.grad(y[:, :cutoff].float().square().sum(), inputs)
    assert all(torch.count_nonzero(g[:, cutoff:]) == 0 for g in grads)
    if time == 1:
        torch.testing.assert_close(y, 2 * inputs[-1], rtol=0, atol=0)


@pytest.mark.parametrize("heads,dim", [(1, 4), (3, 24), (6, 128)])
def test_small_head_and_feature_shapes(heads, dim):
    inputs = make_inputs((2, 17, heads, dim), device="cuda", dtype=torch.bfloat16)
    y = attention(*inputs)
    expected = materialized_attention(tuple(x.float() for x in inputs))
    torch.testing.assert_close(y.float(), expected, rtol=0.01, atol=0.005)


@pytest.mark.parametrize("rope", [False, True])
def test_small_layer_parameter_chain(rope):
    torch.manual_seed(3)
    model = FutureAwareKVAttention(
        width=32, heads=2, rope=rope, inner_gain=0.7, outer_gain=0.35, backend="triton"
    ).cuda()
    ref = copy.deepcopy(model)
    ref.backend = "reference"
    x = torch.randn(2, 33, 32, device="cuda", requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    y, expected = model(x), ref(xr)
    upstream = torch.randn_like(y)
    torch.testing.assert_close(y, expected, rtol=1e-3, atol=1e-4)
    got = torch.autograd.grad(y, (x, *model.parameters()), upstream)
    want = torch.autograd.grad(expected, (xr, *ref.parameters()), upstream)
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)
