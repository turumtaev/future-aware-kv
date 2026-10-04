import pytest
import torch

from future_aware_kv import future_aware_kv_reference
from future_aware_kv.experimental import block_prefix_attention, materialized_prefix_attention


@pytest.mark.parametrize("block_size", [1, 2, 3, 6])
def test_future_work_references_match_cubic_outputs_and_gradients(block_size):
    torch.manual_seed(21)
    projections = tuple(
        torch.randn(2, 6, 2, 4, dtype=torch.float64, requires_grad=True)
        for _ in range(9)
    )
    gate_k = torch.tensor([0.2, 0.8], dtype=torch.float64, requires_grad=True)
    gate_v = torch.tensor([0.7, 0.3], dtype=torch.float64, requires_grad=True)
    variables = (*projections, gate_k, gate_v)
    expected = future_aware_kv_reference(*variables)
    materialized = materialized_prefix_attention(*variables)
    blocked = block_prefix_attention(*variables, block_size=block_size)
    torch.testing.assert_close(materialized, expected, rtol=3e-11, atol=3e-11)
    torch.testing.assert_close(blocked, expected, rtol=3e-11, atol=3e-11)

    upstream = torch.randn_like(expected)
    expected_gradient = torch.autograd.grad(expected, variables, upstream, retain_graph=True)
    for output in (materialized, blocked):
        gradient = torch.autograd.grad(output, variables, upstream, retain_graph=True)
        for actual, reference in zip(gradient, expected_gradient):
            torch.testing.assert_close(actual, reference, rtol=3e-9, atol=3e-9)


def test_block_reference_validates_block_size():
    tensors = tuple(torch.randn(1, 5, 1, 2) for _ in range(9))
    gates = (torch.tensor([0.5]), torch.tensor([0.5]))
    with pytest.raises(ValueError, match="divide sequence length"):
        block_prefix_attention(*tensors, *gates, block_size=2)
