"""Restore long outputs, final states, guard thresholds and microbatch checks."""

import pytest
import torch

from future_aware_kv import attention
from .oracles import make_inputs, materialized_attention, dense_final_state

pytestmark = pytest.mark.cuda


def check_forward_and_state(monkeypatch, shape, *, case="normal"):
    from future_aware_kv import _forward

    inputs = make_inputs(shape, device="cuda", dtype=torch.bfloat16, case=case)
    allocated = []
    original = _forward.allocate_state

    def capture_state(*args, **kwargs):
        state = original(*args, **kwargs)
        allocated.append(state)
        return state

    monkeypatch.setattr(_forward, "allocate_state", capture_state)
    with torch.no_grad():
        y = _forward.run(inputs, training=False)[0]
        targets = sorted(
            {
                i
                for i in (
                    0,
                    7,
                    15,
                    16,
                    31,
                    32,
                    63,
                    64,
                    127,
                    128,
                    255,
                    256,
                    511,
                    512,
                    1023,
                    shape[1] - 1,
                )
                if i < shape[1]
            }
        )
        refs = tuple(x.float() for x in inputs)
        expected = materialized_attention(refs, targets)
        final = dense_final_state(refs)
    torch.testing.assert_close(y[:, targets].float(), expected, rtol=0.01, atol=0.005)
    assert len(allocated) == 1
    b, t, h, d = shape
    zk, zv, m, l = (x[0] for x in allocated[0])
    actual = (
        zk.reshape(b, h, t, d),
        zv.reshape(b, h, t, d),
        m.reshape(b, h, t),
        l.reshape(b, h, t),
    )
    for got, want in zip(actual, final):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, want, rtol=0.0005, atol=0.0005)


@pytest.mark.parametrize(
    "time,dim",
    [
        (1, 16),
        (17, 32),
        (65, 16),
        (128, 128),
        (129, 32),
        (255, 32),
        (256, 64),
        (257, 128),
        (511, 32),
        (512, 32),
        (513, 32),
    ],
)
def test_dense_outputs_and_final_state(monkeypatch, time, dim):
    # Direct retained long forward also checks the old small-tail kernel cases;
    # the public dispatcher normally selects the small backend for T<=128.
    check_forward_and_state(monkeypatch, (1, time, 2, dim))


@pytest.mark.slow
def test_nanochat_shape_outputs_and_final_state(monkeypatch):
    check_forward_and_state(monkeypatch, (1, 2048, 6, 128))


@pytest.mark.parametrize("time", [17, 129, 257])
@pytest.mark.parametrize("case", ["gap31", "gap33", "gap128", "repeated_gap128"])
def test_guard_thresholds_and_repeated_score_jumps(monkeypatch, time, case):
    check_forward_and_state(monkeypatch, (1, time, 2, 64), case=case)


@pytest.mark.parametrize("case", ["normal", "rope"])
def test_future_changes_do_not_affect_prefix_outputs(case):
    inputs = make_inputs(
        (2, 129, 3, 32), device="cuda", dtype=torch.bfloat16, case=case
    )
    changed = [x.clone() for x in inputs]
    for x in changed:
        x[:, 33:] = torch.randn_like(x[:, 33:]) * 4
    with torch.no_grad():
        y = attention(*inputs)
        modified = attention(*changed)
        expected = materialized_attention(
            tuple(x.float() for x in inputs), targets=[0, 32, 128]
        )
    torch.testing.assert_close(y[:, :33], modified[:, :33], rtol=0, atol=0)
    torch.testing.assert_close(
        y[:, [0, 32, 128]].float(), expected, rtol=0.01, atol=0.005
    )


def test_microbatch_inference_matches_batched_kernel():
    from future_aware_kv import _forward

    inputs = make_inputs((4, 129, 2, 32), device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        microbatched = attention(*inputs)
        batched = _forward.run(inputs, training=False)[0]
    torch.testing.assert_close(microbatched, batched, rtol=0, atol=0)
