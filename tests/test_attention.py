import pytest
import torch
from future_aware_kv import (
    attention,
    reference_attention,
    FutureAwareKVAttention,
    decode_step,
)


def materialized_oracle(xs):
    qf, kf, qc, kc, vc = (x.transpose(1, 2) for x in xs)
    scale = qf.shape[-1] ** -0.5
    ys = []
    for t in range(qf.shape[2]):
        p = (
            qf[..., : t + 1, :] @ kf[..., : t + 1, :].transpose(-1, -2) * scale
        ).softmax(-1)
        key = kc[..., : t + 1, :] + p @ kc[..., : t + 1, :]
        value = vc[..., : t + 1, :] + p @ vc[..., : t + 1, :]
        c = (qc[..., t : t + 1, :] @ key.transpose(-1, -2) * scale).softmax(-1)
        ys.append((c @ value).squeeze(-2))
    return torch.stack(ys, 2).transpose(1, 2)


@pytest.mark.parametrize("time", [1, 7])
def test_factorization_and_all_gradients(time):
    torch.manual_seed(3)
    xs = [
        torch.randn(2, time, 2, 4, dtype=torch.float64, requires_grad=True)
        for _ in range(5)
    ]
    y = reference_attention(*xs)
    expected = materialized_oracle(xs)
    gy = torch.randn_like(y)
    torch.testing.assert_close(y, expected, rtol=1e-12, atol=1e-12)
    for a, b in zip(
        torch.autograd.grad(y, xs, gy), torch.autograd.grad(expected, xs, gy)
    ):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-11)
    if time == 1:
        torch.testing.assert_close(y, 2 * xs[-1])


@pytest.mark.parametrize("rope", [False, True])
def test_layer_decode_and_future_causality(rope):
    torch.manual_seed(19)
    layer = FutureAwareKVAttention(
        16,
        2,
        rope=rope,
        backend="reference",
        inner_gain=2,
        outer_gain=4,
        kv_gain=2**-0.5,
    )
    x = torch.randn(2, 9, 16, requires_grad=True)
    y = layer(x)
    changed = x.detach().clone()
    changed[:, 5:] += 100
    torch.testing.assert_close(y[:, :5], layer(changed)[:, :5], rtol=0, atol=0)
    g = torch.autograd.grad(y[:, :5].square().sum(), x, retain_graph=True)[0]
    assert torch.count_nonzero(g[:, 5:]) == 0
    state = None
    decoded = []
    for token in x.detach().split(1, 1):
        z, state = layer.decode(token, state)
        decoded.append(z)
    torch.testing.assert_close(y, torch.cat(decoded, 1), rtol=2e-5, atol=2e-6)
    assert state.length == 9
    assert not state.zk.requires_grad
    assert sum(p.numel() for p in layer.parameters()) == 6 * 16 * 16
    # Muon must receive six independent square matrices, not one packed
    # rectangular projection parameter plus the output map.
    parameters = list(layer.parameters())
    assert len(parameters) == 6
    assert all(p.shape == (16, 16) for p in parameters)
    assert len({p.data_ptr() for p in parameters}) == 6
    y.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters()
    )


def test_decode_gap128_and_newborn_query():
    torch.manual_seed(29)
    xs = [torch.randn(1, 17, 2, 4, dtype=torch.float64) for _ in range(5)]
    xs[0].zero_()
    xs[0][..., 0] = 2
    xs[1].zero_()
    xs[1][:, 8:, :, 0] = 128
    state = None
    ys = []
    for t in range(17):
        y, state = decode_step(*(x[:, t : t + 1] for x in xs), state=state)
        ys.append(y)
    torch.testing.assert_close(
        torch.cat(ys, 1), reference_attention(*xs), rtol=1e-11, atol=1e-11
    )
    a = (xs[0].transpose(1, 2) @ xs[1].transpose(1, 2).transpose(-1, -2)) / 2
    p = a.softmax(-1)
    maximum = a.max(-1).values
    denominator = (a - maximum.unsqueeze(-1)).exp().sum(-1)
    torch.testing.assert_close(state.maximum.transpose(1, 2), maximum, rtol=0, atol=0)
    torch.testing.assert_close(
        state.denominator.transpose(1, 2), denominator, rtol=1e-11, atol=1e-11
    )
    torch.testing.assert_close(
        state.zk.transpose(1, 2), p @ xs[3].transpose(1, 2), rtol=1e-11, atol=1e-11
    )
    torch.testing.assert_close(
        state.zv.transpose(1, 2), p @ xs[4].transpose(1, 2), rtol=1e-11, atol=1e-11
    )


def test_errors_and_automatic_cpu_reference():
    xs = [torch.randn(1, 3, 2, 4) for _ in range(5)]
    torch.testing.assert_close(attention(*xs), reference_attention(*xs))
    with pytest.raises(ValueError):
        attention(*xs, backend="typo")
    with pytest.raises(RuntimeError):
        attention(*xs, backend="triton")
    with pytest.raises(ValueError):
        attention(xs[0][:, :0], *xs[1:])
    with pytest.raises(ValueError):
        FutureAwareKVAttention(15, 2)
    with pytest.raises(ValueError):
        decode_step(*xs)
