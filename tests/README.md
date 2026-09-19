# Release regression tests

This suite adapts the original research tests to the retained package API. It
has **149 parametrized cases: 35 CPU and 114 CUDA**, including seven optional
full-length cases. The historical 206-test GPU result also covered discarded
kernel variants and tuning settings; it is distinct from the release suite's case count.

Current verification: **all 149 cases passed, with no failures or skips**, on
an RTX 4090 with Python 3.12.14, PyTorch 2.11.0+cu128 and Triton 3.6.0.
Installed-layer forward/backward also passed at B=1, T=2048, width=768,
six heads, BF16 autocast, both backward schedules and both RoPE/NoPE.
GPU reference decode matched prefill at B=2, T=17, width=32, two heads,
FP64 and both RoPE/NoPE at rtol=atol=1e-10.

## Coverage

| File | Cases | Coverage |
| --- | ---: | --- |
| `test_attention.py` | 6 CPU | Reference gradients, decode/prefill, newborn-query catch-up, causality, final decode state, errors |
| `test_reference.py` | 14 CPU | Independent materialized and checkpointed streaming oracles; all five gradients and query gains |
| `test_validation.py` | 15 CPU + 5 CUDA | Shape, dtype, device and unsupported-option rejection |
| `test_small.py` | 32 CUDA | T≤128 forward/backward, FP32/FP16/BF16, boundary lengths, heads and layer gradients |
| `test_long_forward.py` | 27 CUDA | Stable inference, final M/L/ZK/ZV, gaps of 31/33/128, repeated jumps, causality and batch scheduling |
| `test_long_backward.py` | 46 CUDA | Checkpoint and suffix backward, all projection gradients, layer gradients, prefix causality, noncontiguous inputs and adversarial jumps |
| `test_schedule.py` | 4 CUDA | Saved tensors, actual source-window-major launches, bounded scratch and quadratic pair coverage |

`oracles.py` contains independent PyTorch calculations. Small cases explicitly
construct contextual keys/values; long gradient cases use a stable per-token
recurrence with PyTorch checkpointing. References consume the same rounded
projections as Triton. FP32 accumulator checks run before public BF16 gradient
rounding; tolerances account for tensor-core arithmetic. CUDA reference matmuls
disable TF32 for these comparisons.

## Run

Use Python 3.10 or later in a virtual environment. For CPU tests:

```bash
python -m pip install -e '.[test]'
python -m pytest -q -m 'not cuda'
```

On Linux with a compatible CUDA PyTorch installation already installed:

```bash
python -m pip install -e '.[cuda,test]'
python -m pytest -q
```

The default run omits seven expensive T=2048 cases. For the final GPU check:

```bash
python -m pytest -q --run-slow
```

That includes B=1, T=2048, H=6, D=128 forward/state checks and both backward
schedules against independent gradients, with RoPE and repeated gap=128
inputs. First-use compilation and full-length reference backward can take
several minutes. Run sequentially; parallel pytest workers increase GPU memory
pressure and are unnecessary for correctness checks.

CUDA cases skip when CUDA or Triton is unavailable. Check the pytest skip
summary: a complete final GPU run must execute all 149 cases without skips.
These are correctness tests, not latency benchmarks or new training results.
