"""Check actual retained suffix launches and saved activation ownership."""

import math
import pytest
import torch

from future_aware_kv import attention
from .oracles import make_inputs

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize("schedule,saved_count", [("eager", 7), ("graph", 11)])
def test_saved_tensors_match_backward_memory_contract(schedule, saved_count):
    inputs = tuple(
        x.requires_grad_()
        for x in make_inputs((1, 513, 1, 16), device="cuda", dtype=torch.bfloat16)
    )
    y = attention(*inputs, schedule=schedule)
    saved = y.grad_fn.saved_tensors
    assert len(saved) == saved_count
    for got, original in zip(saved[:5], inputs):
        assert got.data_ptr() == original.data_ptr()
    if schedule == "eager":
        assert saved[5].shape == inputs[0].shape  # FP32 Y, no vector checkpoints.
        assert saved[5].dtype == torch.float32
        assert saved[6].numel() == 513  # C log-sum-exp.
    else:
        for state in saved[5:9]:
            assert state.shape[0] == math.ceil(513 / 256) + 1
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(y.sum(), inputs))


class RecordingLaunch:
    """Record host launch arguments, then execute the real Triton kernel."""

    def __init__(self, kernel, phase, events):
        self.kernel, self.phase, self.events = kernel, phase, events

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def recorded(*args, **kwargs):
            bound = dict(zip(self.kernel.arg_names, args))
            scalars, norms = bound["SCALARS"], bound["NORMS"]
            self.events.append(
                (
                    self.phase,
                    bound["SOURCE_START"],
                    bound["INDEX"],
                    tuple(scalars.shape),
                    tuple(norms.shape),
                    tuple(bound.get("ZK", bound.get("RK")).shape),
                )
            )
            return launch(*args, **kwargs)

        return recorded


@pytest.mark.parametrize("time", [513, 1025])
def test_source_window_major_schedule_and_bounded_scratch(monkeypatch, time):
    from future_aware_kv import _forward, _suffix

    inputs = make_inputs((1, time, 1, 16), device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn_like(inputs[0])
    events = []
    for name, phase in (
        ("_prefix_superblock", "forward"),
        ("_suffix_superblock", "reverse"),
    ):
        monkeypatch.setattr(
            _suffix, name, RecordingLaunch(getattr(_suffix, name), phase, events)
        )
    with torch.no_grad():
        _, y32, lc = _forward.run(inputs)
        grads = _suffix.backward_run(inputs, y32, lc, upstream)
    assert all(torch.isfinite(g).all() for g in grads)
    w, bt, nb = 512, 16, 4
    blocks = math.ceil(time / bt)
    groups = list(range(0, blocks, nb))
    expected = []
    for source_start in range(0, time, w):
        expected.extend(("forward", source_start, i) for i in groups)
        expected.extend(("reverse", source_start, i) for i in reversed(groups))
    assert [e[:3] for e in events] == expected
    for _, _, _, scalar_shape, norm_shape, vector_shape in events:
        assert scalar_shape == (3, blocks, 1, w // 32, bt, 32)
        assert norm_shape == (2, blocks, 1, w)
        assert vector_shape == (1, w, 16)
    # Count every valid pair covered by each traversal's actual launch ranges.
    # Independent of grouping, coverage is exactly T*T, not T*T*ceil(T/W).
    pairs = {"forward": 0, "reverse": 0}
    for phase, source_start, i, *_ in events:
        source_count = min(w, time - source_start)
        target_count = min(nb * bt, time - i * bt)
        pairs[phase] += source_count * target_count
    assert pairs == {"forward": time * time, "reverse": time * time}
