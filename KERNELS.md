# How the QK-F / QKV-C kernels work

The [README](README.md) explains the attention layer in token-by-token loops.
This guide explains how its long-sequence GPU implementation reuses prefix
summaries, replaces vector operations with matrix multiplications, and computes
gradients without saving every contextual key/value.

F means full-prefix attention; C means causal retrieval from its updated
memories. All F/C work stays within each head. The final W_OC projection mixes
heads afterward, in the PyTorch layer rather than the attention kernel.

## Notation and files

| Symbol | Meaning |
|---|---|
| B | Batch size |
| T | Sequence length |
| H | Number of attention heads |
| d | Channels per head; model width is H × d |
| t | Absolute position of the current target/output token |
| s | Absolute position of a source memory being updated |
| j | Absolute position of an input contributing to that memory |
| BS = 32 | Sources processed together by one Triton program |
| BT = 16 | Arriving input/target positions processed in one block |

A **Triton program** is one instance of a kernel assigned a piece of work. A
**tile** is the rectangular group of tensor elements it processes. A **kernel
launch** dispatches many programs to the GPU. NVIDIA **tensor cores** accelerate
matrix multiplications; the implementation tries to put its expensive vector
work into operations they can execute.

The functional API accepts five tensors shaped `[B, T, H, d]`. The files are:

| File | Purpose |
|---|---|
| [attention.py](src/future_aware_kv/attention.py) | Reference computation and backend selection |
| [layer.py](src/future_aware_kv/layer.py) | Five projections, optional RoPE and final W_OC |
| [decode.py](src/future_aware_kv/decode.py) | PyTorch reference decode and cache |
| [_common.py](src/future_aware_kv/_common.py) | Stable normalization and precision helpers |
| [_small.py](src/future_aware_kv/_small.py) | Separate T ≤ 128 implementation and gradients |
| [_forward.py](src/future_aware_kv/_forward.py) | Long forward, source partials and their reduction |
| [_checkpoint.py](src/future_aware_kv/_checkpoint.py) | Backward with saved prefix summaries |
| [_suffix.py](src/future_aware_kv/_suffix.py) | Backward with source-window replay and reverse accumulation |

## First, expand the attention definition

Consider one head at target `t`, with `n = t + 1` visible tokens. In this section
all source/input positions are in `0..t`. `softmax_rows` normalizes each row over
its input positions. Full attention produces scalar weights P:

```python
P = softmax_rows(QF[:n] @ KF[:n].T / sqrt(d))  # [sources, inputs]
contextual_keys = KC[:n] + P @ KC[:n]         # [sources, d]
contextual_values = VC[:n] + P @ VC[:n]       # [sources, d]
```

C normally takes a dot product between QC[t] and every contextual key. Expand
that dot product instead: a dot product with a weighted sum equals the weighted
sum of dot products.

```python
G = QC[t] @ KC[:n].T / sqrt(d)  # [inputs]: scores against original keys.
scores_c = G + P @ G            # [sources]: bypass + contextual contribution.
C = softmax(scores_c)           # Retrieval weights, normalized over sources.
```

The value calculation can be rearranged in the same way:

```python
Y = C @ (VC[:n] + P @ VC[:n])
# Compute effective scalar weights first, then multiply original values:
Y = (C + C @ P) @ VC[:n]
```

This algebra explains the factorization. Evaluating these whole-prefix matrices
separately for every `t` would still take cubic work. The long kernel instead
reuses the previous prefix and applies this expansion only to a small new block.

## Reuse the previous prefix

At block entry each source already has four quantities summarizing all earlier
inputs:

| State | Meaning |
|---|---|
| M0 | Maximum F score in the previous prefix |
| L0 | Sum of exp(score − M0) over that prefix |
| ZK0 | Normalized F-weighted average of previous KC inputs |
| ZV0 | Normalized F-weighted average of previous VC inputs |

Now process 16 arriving positions. Let `source_start` and `block_start` be
absolute positions. Local source `i` corresponds to `s = source_start + i`;
local target `r` corresponds to `t = block_start + r`; local input `k`
corresponds to `j = block_start + k`. The target sees new inputs `k <= r`.
The following indexed equations describe the tile; they are not a standalone
Python implementation.

```python
sources = range(source_start, source_start + BS)
new_block = range(block_start, block_start + BT)
S = QF[sources] @ KF[new_block].T / sqrt(d)  # [BS, BT]

M[i,r] = max(M0[i], max(S[i,k] for k in range(r + 1)))
old_mass[i,r] = L0[i] * exp(M0[i] - M[i,r])  # Zero for an empty old prefix.
L[i,r] = old_mass[i,r] + sum(exp(S[i,k] - M[i,r]) for k in range(r + 1))
old_fraction[i,r] = old_mass[i,r] / L[i,r]
new_fraction[i,r,k] = exp(S[i,k] - M[i,r]) / L[i,r]  # Only k <= r.
```

The contextual key is the original source plus an old-prefix contribution and
a new-block contribution:

```python
s = source_start + i
key[i,r] = KC[s] + old_fraction[i,r] * ZK0[i]
key[i,r] += sum(new_fraction[i,r,k] * KC[block_start + k]
                for k in range(r + 1))
# The contextual value has the same expansion with VC and ZV0.
```

Compute the dot products needed for C with three matrix multiplications:

```python
Q_source = QC[new_block] @ KC[sources].T / sqrt(d)  # [BT, BS]
Q_old = QC[new_block] @ ZK0.T / sqrt(d)             # [BT, BS]
Q_new = QC[new_block] @ KC[new_block].T / sqrt(d)    # [BT, BT]

score_c[r,i] = Q_source[r,i] + old_fraction[i,r] * Q_old[r,i]
score_c[r,i] += sum(new_fraction[i,r,k] * Q_new[r,k] for k in range(r + 1))
```

The implementation calls these matrices QU, QZ and QV respectively. QV here
contains dot products with new **keys**, despite its internal name. Keeping
these terms separate lets the kernel calculate scores without carrying a
new d-wide key/value summary for every intermediate target.

### Factor the value path too

For each target, mask sources with `s > t`. Within the source tile compute a
maximum C score and weights `W[r,i] = exp(score_c[r,i] - tile_max[r])` for valid
sources. These are unnormalized weights; other source tiles will contribute to
the same final softmax.

```python
old_coeff = W * old_fraction.T                 # [BT, BS]
new_coeff[r,k] = sum(W[r,i] * new_fraction[i,r,k] for i in range(BS))
# new_coeff[r,k] is zero when k > r.
partial_numerator = W @ VC[sources]
partial_numerator += old_coeff @ ZV0
partial_numerator += new_coeff @ VC[new_block]
partial_denominator = sum_over_sources(W)
```

The displayed `new_fraction` is mathematical notation. The kernel does not
save a full `[BT, BS, BT]` probability tensor. It uses a fast shared-scale path
or regenerates small groups of coefficients as needed. At block end it saves
only the final M/L/ZK/ZV per source for the next block.

## Parallel execution and output reduction

One program owns 32 source queries for one batch item and head. Source tiles,
heads and batch items run independently. All training/prefill source queries
are available, so even sources after the current target can accumulate private
F summaries. C excludes those sources until their positions become visible.

Each 16-target block has a source-update launch and an output-reduction launch.
Target blocks advance in order because the next block uses the previous block's
prefix summaries. The output reduction combines every source tile's maximum,
denominator and numerator by stable softmax rescaling, then writes completed
outputs. Context and partial buffers are reused for the next block.

At B = 1 and T = 2048, forward has 128 source-update launches and 128 output
reductions. CUDA graphs replay that launch sequence with less CPU overhead;
ordinary eager execution pays for dispatching each launch. Inference processes
batch items sequentially to reduce scratch memory, then concatenates outputs.
Training uses the full batch in the retained long paths.

Forward work is O(BHT²d), with O(BHTd) persistent summaries plus one block's
partials. Extra normalization and traffic remain inside that work bound;
[benchmarks](results/benchmarks.md) report the current implementation's costs.

## Numerical stability and precision

A fast path scales all F weights in a source row by one block-wide maximum.
If a late score is much larger than an early prefix's scores, early weights can
underflow before they are normalized. The **gap = 128** stress test deliberately
creates such a score jump.

The kernel uses the fast path only when the block-final maximum exceeds the
first local prefix's maximum by at most 32 for every valid source row. Otherwise
it computes each prefix's own maximum/denominator and regenerates coefficients
in groups of four targets. This stable fallback avoids keeping the complete
three-index coefficient tensor live and passes the adversarial gap test.

M/L/ZK/ZV and softmax accumulators use FP32. To use BF16 tensor-core operations
with an FP32 operand, the kernel splits it into a BF16 high part and a BF16
approximation to the remainder. For two such operands, it approximates:

```text
a @ b ≈ a_high @ b_high + a_high @ b_low + a_low @ b_high
```

It omits low × low. Original BF16 operands need no split. This is approximate
FP32 computation, checked with relative/absolute tolerances. Outputs and public
gradients are rounded to the projection dtype.

The retained long forward uses BS = 32, BT = 16, four warps (groups of 32 GPU
threads) and three compiler pipeline stages. Detailed compiler resources are
in [benchmarks](results/benchmarks.md).

## Two backward schedules

Both schedules compute gradients for QF, KF, QC, KC and VC. Some KC/VC gradients
come directly from the original-source bypass; others come from their role as
F payloads. Those contributions must be combined without double counting.
The final W_OC and input projections use ordinary PyTorch autograd.

### Checkpointed backward: `schedule="graph"`

Forward saves M/L/ZK/ZV every 256 targets, plus the FP32 output and C log-sum-exp
(the scalar normalization needed to reconstruct C probabilities). These saved
prefix summaries are **checkpoints**; they are not model-weight checkpoints.

Backward visits intervals in reverse. Within each interval it reruns forward
from the saved entry state, reconstructs summaries at 16-target boundaries,
then propagates gradients backward through them. Programs owning source tiles
accumulate source contributions; separate reductions combine contributions to
target projections. Splitting score-gradient and value-gradient work into
separate launches reduced the measured kernel pressure.

For checkpoint interval I, saved/replayed vector storage is approximately
O(BHTd × (T/I + I/16)), excluding other output/gradient buffers. With fixed I,
the saved-boundary component grows quadratically in T. This schedule has the
lowest measured CUDA-graph latency among the included implementations.

### Source-window backward: `schedule="eager"` (default)

Forward retains original projections, FP32 output and C log-sum-exp, with no
saved d-wide boundary summaries. Backward handles one fixed source window of
W = 512 positions at a time:

1. Replay the entire target range `0..T-1` once for those sources. Carry only
   window-local M/L/ZK/ZV and save scalar information needed for backward.
2. Traverse the entire target range `T-1..0` once for the same sources. Accumulate
   gradients using reverse/suffix sums—contributions from later targets to each
   earlier input—and combine source/target projection gradients.
3. Reuse the scratch for the next disjoint source window.

This is **source-window-major**: a source/target pair is recomputed only O(1)
times. It does not restart earlier prefixes for every target window. Work stays
O(BHT²d), with scalar history O(BHWT) and vector state O(BHWd), apart from
output/gradient buffers O(BHTd). There are no full-sequence T×T scalar histories
and no saved d-wide forward boundary summaries. When T ≤ W, one padded source
window naturally covers the whole small sequence.

To reduce launch overhead, each source program handles four consecutive
16-target phases per launch in both traversals (NB = 4 in the implementation).
The phases still use bounded global scratch, not exclusively GPU registers.
Separate reductions combine target-gradient partials after each group; this
adds workspace O(BH × NB × Wd), with NB fixed independently of T.

A **register spill** occurs when live data exceeds the allocated registers and
is placed in GPU local memory. This schedule has real spills, but fewer launches
make it the fastest measured eager option. Gradients accumulate in FP32 without
floating-point atomic additions. The timing/memory tradeoff is in
[benchmarks](results/benchmarks.md).

## Small sequences and decode

T ≤ 128 uses a separate implementation: it computes full scalar F scores and
QC/KC dot products, then derives C and effective value coefficients for each
target. Custom scalar kernels compute coefficient gradients; PyTorch handles
the projection dot-product gradients. It saves no cubic probability history,
but its boundary-by-boundary scalar work is cubic in T. This is the small path
used in reversal, not the quadratic long kernel described above.

Reference decode keeps original QF/KF/KC/VC and per-source normalization/context
state. The new source first reads earlier inputs; then every live source adds
the arriving token once, and C retrieves from the resulting memories. Work and
cache are O(td) per token. It is inference-only PyTorch code.

## Layer integration

`FutureAwareKVAttention(width=768, heads=6)` splits its projections into six
128-wide heads. There is no mixing between F and C; only final W_OC mixes heads.
QF/KF/QC/KC/VC use five separate `nn.Linear(width, width, bias=False)`
parameters, allowing Muon to orthogonalize each projection independently as
in the nanochat integration. The output projection is a sixth separate matrix.
The constructor rejects unsupported keywords with `TypeError`. Tensor shapes,
devices, dtypes, head widths and backend/schedule names are validated separately.
There are no attention masks, padding masks, attention-probability dropout or
grouped-query attention settings. External `nn.Dropout` is usable. Padding is
treated as input, so callers must avoid feeding padded rows as real sequences.

Layer RoPE uses adjacent pairs, positive rotation and base 10000, matching the
reversal convention. Nanochat used base 100000, negative split-half rotation,
Q/K root-mean-square normalization, value embeddings and its own initialization.
For that kind of integration, use `attention(qf, kf, qc, kc, vc)` with already
processed projections. The fixed `inner_gain`, `outer_gain` and `kv_gain` options
scale QF, QC and KC/VC respectively; default layer initialization is not the
full-model experiments' recipe.

`backend="reference"` forces the PyTorch reference. The `schedule` argument
selects backward's algorithm; choosing `"graph"` does not capture or replay a
CUDA graph for the caller. Triton supports first derivatives only.

## What has been verified

The original combined GPU suite passed 206 tests covering outputs, all five
projection gradients, causality, odd lengths, multiple heads/batches, RoPE,
no position encoding, and repeated score jumps. Retained kernel arithmetic and
launch configurations were checked against the measured implementation.

The restored [release suite](tests/README.md) has 149 parametrized cases: 35
CPU and 114 CUDA. It checks independent materialized/streaming references,
all five projection gradients, future-token isolation, decode/prefill
equivalence, final M/L/ZK/ZV, adversarial jumps, both backward schedules and
the actual bounded source-window-major launch schedule. All 149 passed on an
RTX 4090, including T=2048/H=6/D=128 outputs/state and gradients. Additional
installed-layer forward/backward checks passed at width 768 for both schedules,
with and without RoPE. GPU reference decode matched prefill; it has not been
latency-benchmarked or validated as a complete language-model generation
deployment. See the [packaged regression suite](tests/README.md).

Run `python -m pytest -q --run-slow` on a CUDA machine to include all cases.
Reported GPU timings are historical measurements, not fresh timings of the
packaged release.
