import pytest
import torch

from future_aware_kv import (
    FutureAwareKVAttention,
    RotaryEmbedding,
    apply_rope,
    future_aware_kv_reference,
)
from future_aware_kv.attention import _future_aware_kv_queries


def _literal_definition(projections, gate_k, gate_v):
    qfk, kfk, qfv, kfv, qc, kc0, kcf, vc0, vcf = (
        x.transpose(1, 2) for x in projections
    )
    scale = qfk.shape[-1] ** -0.5
    gk = gate_k[None, :, None, None]
    gv = gate_v[None, :, None, None]
    outputs = []
    for target in range(qfk.shape[2]):
        stop = target + 1
        pk = (qfk[:, :, :stop] @ kfk[:, :, :stop].transpose(-1, -2) * scale).softmax(-1)
        pv = (qfv[:, :, :stop] @ kfv[:, :, :stop].transpose(-1, -2) * scale).softmax(-1)
        key = gk * kc0[:, :, :stop] + (1 - gk) * (pk @ kcf[:, :, :stop])
        value = gv * vc0[:, :, :stop] + (1 - gv) * (pv @ vcf[:, :, :stop])
        outer = (qc[:, :, target : stop] @ key.transpose(-1, -2) * scale).softmax(-1)
        outputs.append((outer @ value).squeeze(-2))
    return torch.stack(outputs, 2).transpose(1, 2)


def test_cubic_matrix_form_matches_literal_definition():
    torch.manual_seed(1)
    projections = tuple(
        torch.randn(2, 5, 2, 3, dtype=torch.float64, requires_grad=True)
        for _ in range(9)
    )
    gate_k = torch.tensor([0.2, 0.8], dtype=torch.float64, requires_grad=True)
    gate_v = torch.tensor([0.7, 0.3], dtype=torch.float64, requires_grad=True)
    actual = future_aware_kv_reference(*projections, gate_k, gate_v)
    expected = _literal_definition(projections, gate_k, gate_v)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)
    upstream = torch.randn_like(actual)
    variables = (*projections, gate_k, gate_v)
    got = torch.autograd.grad(actual, variables, upstream, retain_graph=True)
    want = torch.autograd.grad(expected, variables, upstream)
    for left, right in zip(got, want):
        torch.testing.assert_close(left, right, rtol=2e-9, atol=2e-9)


@pytest.mark.parametrize("gate_k,gate_v", [(0.1, 0.9), (0.5, 0.5), (0.9, 0.1)])
def test_query_factorization_matches_cubic_outputs_and_gradients(gate_k, gate_v):
    torch.manual_seed(3)
    body = tuple(
        torch.randn(2, 4, 2, 3, dtype=torch.float64, requires_grad=True)
        for _ in range(9)
    )
    query = tuple(
        torch.randn(2, 3, 2, 3, dtype=torch.float64, requires_grad=True)
        for _ in range(9)
    )
    gk = torch.tensor([gate_k, min(1.0, gate_k + 0.05)], dtype=torch.float64,
                      requires_grad=True)
    gv = torch.tensor([gate_v, max(0.0, gate_v - 0.05)], dtype=torch.float64,
                      requires_grad=True)
    actual = _future_aware_kv_queries(body, query, gk, gv)
    expected = torch.stack([
        future_aware_kv_reference(
            *(torch.cat((b, z[:, index : index + 1]), 1)
              for b, z in zip(body, query)),
            gk,
            gv,
        )[:, -1]
        for index in range(query[0].shape[1])
    ], 1)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)

    upstream = torch.randn_like(actual)
    variables = (*body, *query, gk, gv)
    got = torch.autograd.grad(actual, variables, upstream, allow_unused=True)
    want = torch.autograd.grad(expected, variables, upstream, allow_unused=True)
    for variable, left, right in zip(variables, got, want):
        left = torch.zeros_like(variable) if left is None else left
        right = torch.zeros_like(variable) if right is None else right
        torch.testing.assert_close(left, right, rtol=2e-9, atol=2e-9)


def test_future_key_and_value_geometries_are_independent():
    torch.manual_seed(9)
    tensors = [torch.randn(1, 5, 1, 4, dtype=torch.float64) for _ in range(9)]
    gates = (torch.tensor([0.4]), torch.tensor([0.6]))
    baseline = future_aware_kv_reference(*tensors, *gates)
    changed_key = future_aware_kv_reference(
        tensors[0], tensors[1] * 1.7, *tensors[2:], *gates
    )
    changed_value = future_aware_kv_reference(
        *tensors[:3], tensors[3] * 1.7, *tensors[4:], *gates
    )
    assert not torch.allclose(baseline, changed_key)
    assert not torch.allclose(baseline, changed_value)
    assert not torch.allclose(changed_key, changed_value)


def test_layer_defaults_to_model_width_split_across_heads():
    torch.manual_seed(12)
    layer = FutureAwareKVAttention(width=8, heads=2)
    x = torch.randn(2, 4, 8, requires_grad=True)
    y = layer(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert x.grad is not None


@pytest.mark.parametrize("hard_window", [False, True])
def test_hybrid_window_layer_matches_independent_dense_queries(hard_window):
    torch.manual_seed(13)
    layer = FutureAwareKVAttention(
        width=8,
        heads=2,
        threshold_reference_length=5,
        threshold_temperature=0.7,
    ).double()
    layer.hard_window = hard_window
    body = torch.randn(2, 4, 8, dtype=torch.float64)
    queries = torch.randn(2, 3, 8, dtype=torch.float64)
    assert layer(body).shape == body.shape
    together = layer.forward_queries(body, queries)
    dense = torch.stack(
        [layer(torch.cat((body, queries[:, index : index + 1]), dim=1))[:, -1]
         for index in range(queries.shape[1])],
        dim=1,
    )
    torch.testing.assert_close(together, dense, rtol=2e-11, atol=2e-11)
    separate = torch.cat(
        [layer.forward_queries(body, queries[:, index : index + 1])
         for index in range(queries.shape[1])],
        dim=1,
    )
    torch.testing.assert_close(together, separate, rtol=2e-12, atol=2e-12)

    if not hard_window:
        together.square().mean().backward()
        assert torch.isfinite(layer.route_past_threshold_raw.grad).all()
        assert torch.isfinite(layer.route_future_threshold_raw.grad).all()

    projected = torch.randn(2, 3, 2, 4, dtype=torch.float64)
    shared = apply_rope(projected, torch.full((3,), 7))
    repeated = torch.cat(
        [apply_rope(projected[:, index : index + 1], torch.tensor([7]))
         for index in range(3)],
        dim=1,
    )
    torch.testing.assert_close(shared, repeated)


def test_rotary_embedding_reuses_nonpersistent_tables():
    rotary = RotaryEmbedding(4).double()
    x = torch.randn(2, 5, 2, 4, dtype=torch.float64)
    first, = rotary(x)
    pointer = rotary._cosine.data_ptr()
    second, = rotary(x)
    assert rotary._cosine.data_ptr() == pointer
    torch.testing.assert_close(first, second)
    assert not any("cosine" in name or "sine" in name for name in rotary.state_dict())
