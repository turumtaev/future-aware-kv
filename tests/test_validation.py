"""Unsupported inputs must fail explicitly instead of changing semantics."""

import pytest
import torch

from future_aware_kv import attention, FutureAwareKVAttention, decode_step
from .oracles import make_inputs


@pytest.mark.parametrize(
    "bad", ["rank", "batch", "time", "heads", "features", "dtype", "integer"]
)
def test_projection_validation(bad):
    inputs = list(make_inputs((2, 5, 2, 4), dtype=torch.float32))
    if bad == "rank":
        inputs[0] = inputs[0][0]
    elif bad == "batch":
        inputs[1] = inputs[1][:1]
    elif bad == "time":
        inputs[1] = inputs[1][:, :-1]
    elif bad == "heads":
        inputs[1] = inputs[1][:, :, :1]
    elif bad == "features":
        inputs[1] = inputs[1][..., :-1]
    elif bad == "dtype":
        inputs[1] = inputs[1].double()
    else:
        inputs = [x.long() for x in inputs]
    with pytest.raises(ValueError):
        attention(*inputs)


@pytest.mark.parametrize("axis", range(4))
def test_empty_projection_axes(axis):
    shape = [1, 5, 2, 4]
    shape[axis] = 0
    inputs = tuple(torch.empty(shape) for _ in range(5))
    with pytest.raises(ValueError):
        attention(*inputs)


@pytest.mark.parametrize("keyword", ["dropout", "attn_mask", "key_padding_mask"])
def test_unsupported_attention_configs(keyword):
    with pytest.raises(TypeError):
        FutureAwareKVAttention(width=16, heads=2, **{keyword: None})


def test_invalid_schedule():
    with pytest.raises(ValueError):
        attention(*make_inputs((1, 5, 1, 4)), schedule="invalid")


@pytest.mark.parametrize(
    "dim,dtype",
    [
        (17, torch.bfloat16),
        (130, torch.bfloat16),
        (32, torch.float32),
        (32, torch.float16),
    ],
)
@pytest.mark.cuda
def test_unsupported_long_kernel_inputs(dim, dtype):
    inputs = make_inputs((1, 129, 1, dim), device="cuda", dtype=dtype)
    with pytest.raises(ValueError):
        attention(*inputs, backend="auto")


@pytest.mark.cuda
def test_mixed_devices():
    inputs = list(make_inputs((1, 5, 1, 4), device="cuda", dtype=torch.float32))
    inputs[1] = inputs[1].cpu()
    with pytest.raises(ValueError):
        attention(*inputs)
