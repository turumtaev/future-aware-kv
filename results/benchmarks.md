# GPU latency and memory

The retained QK-F / QKV-C implementation takes **2.704 ms** for projected
forward at width 768 / 6 heads / 2048 tokens on an RTX 4090. Its two backward
schedules trade GPU latency against memory and CPU launch overhead. A separate
full nanochat experiment measured about 8× slower training updates than baseline.
These are different measurement scopes; the tables below keep them separate.

All measurements used an RTX 4090 (24 GiB), PyTorch 2.11.0+cu128 and Triton 3.6.0.
They describe the measured kernel implementation before package extraction;
there has been no new GPU benchmark of this release. Scripts and raw compiler
artifacts are not included.

## How to read the measurements

| Term | Meaning |
|---|---|
| B | Batch size |
| T | Tokens per sequence |
| H | Heads; here H = 6 for projected measurements |
| d | Channels per head; here d = 128, giving model width 768 |
| Graph ms | GPU time when replaying a captured CUDA launch sequence, reducing CPU dispatch overhead |
| Eager wall ms | Synchronized elapsed time for ordinary execution, including CPU dispatch |
| F+B | Forward and backward measured together |
| Extra peak MiB | Peak allocated tensor memory above the live inputs, including outputs and scratch |

MiB means 2²⁰ bytes. Extra peak is not total GPU memory or the allocator's
reserved-memory figure. Forward-plus-backward times were measured independently;
they are not sums of separate median timings.

## Projected forward: attention only

The five QF/KF/QC/KC/VC tensors are already projected and use BF16. This scope
excludes learned projection matrices, W_OC, the surrounding model and optimizer.
The long kernel processes 32 sources and 16 targets per block with four warps
and three pipeline stages; [KERNELS.md](../KERNELS.md) explains those settings.

Each variant received five warmups. Medians use 25 CUDA-graph samples with
variant order rotated/reversed, and five eager samples. Jobs ran sequentially
on an otherwise idle GPU. The direct control updates d-wide key/value summaries
for each target; the retained version factors that work into matrix products.

| B | T | Direct control: graph ms | QK-F / QKV-C: graph ms | QK-F / QKV-C: eager wall ms | Extra peak MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 1.039 | 0.346 | 1.539 | 4.54 |
| 1 | 1024 | 2.529 | 0.849 | 2.964 | 9.07 |
| 1 | 2048 | 8.227 | 2.704 | 5.887 | 18.14 |
| 4 | 2048 | 30.264 | 11.036 | 26.585 | 27.14 |

Inference handles batch items sequentially to reduce active scratch. At B = 1,
T = 2048, forward launches 128 source updates and 128 output reductions. At
B = 4 it executes four such sequences and concatenates outputs.

A matched B = 1, T = 2048 comparison measured:

| Operation | Graph ms |
|---|---:|
| One ordinary causal attention call, forced PyTorch Flash SDPA | 0.082 |
| Two causal attention calls, first output used as second query | 0.156 |
| Original F+C, direct online source updates | 5.900 |
| Original F+C, stable factorized version | 4.029 |
| QK-F / QKV-C, guarded factorized version | 2.700 |

SDPA is PyTorch's scaled dot-product attention operation; these controls force
its Flash backend. The two-call control is not a full two-layer Transformer:
it has no intermediate learned projections, MLP or residual block. Original
F+C shares one memory as causal key/value; QK-F / QKV-C keeps separate keys and
values. Precision choices also differ. These timings compare implementations,
not learned quality or intrinsic architectural cost.

In this scope the retained five-vector forward is about 33× slower than one
ordinary causal call. The 2.700 and 2.704 ms entries come from distinct variant
measurements in the same study; they are not a further optimization.

## Projected forward and backward

Shape: B = 1, T = 2048, H = 6, d = 128, BF16 projections. Backward computes all
five projection gradients, including their public BF16 casts. It excludes
parameter gradients for the learned projections and the rest of the model.
Five warmups preceded 15 alternating graph samples, five eager samples and
three profiler captures.

| Package setting | Backward graph ms | F+B graph ms | Backward eager wall ms | F+B eager wall ms | F+B extra peak MiB | Backward launches |
|---|---:|---:|---:|---:|---:|---:|
| `schedule="graph"`: save a prefix checkpoint every 256 targets | 14.525 | 17.773 | 33.57 | 50.21 | 391.016 | 1069 |
| `schedule="eager"`: source windows of 512, groups of four target blocks in both traversals | 18.949 | 21.728 | 21.12 | 27.14 | 136.352 | 554 |

The checkpoint schedule wins in CUDA-graph GPU time. The source-window schedule
wins in eager wall time and uses substantially less extra memory. It was used
for nanochat training. Selecting `schedule="graph"` chooses the algorithm;
the caller still has to capture/replay a CUDA graph to obtain graph timings.

The source-window schedule carries only a bounded group of sources while
replaying targets forward once and backward once. Grouping four 16-target
blocks per launch reduces dispatch overhead. It has more register spills, but
that GPU cost is outweighed by fewer launches in eager execution.

A separate scaling check held the source window at **256**, not the 512 above.
Doubling T from 1024 to 2048 produced approximately 4× backward time and 2×
scalar-history memory. Each source/target pair was counted once per traversal:
there was no extra replay factor from the number of target windows. That check
supports quadratic work; its timings should not be extrapolated as the W = 512
operating point.

## Compiler resources and correctness

A register holds thread-local data on the GPU. A spill moves excess live data
to local memory. Shared memory is storage within one thread block. Barrier and
shuffle sites are compiled synchronization/data-exchange instructions; a static
site may execute repeatedly in loops or never execute in a skipped branch.

| Selected kernel | Registers/thread | Compiler-reported spills/thread | Shared bytes/block | Local load/store sites | Barrier/shuffle sites |
|---|---:|---:|---:|---:|---:|
| Guarded forward source update | 240 | 0 | 20,480 | 0 / 0 | 149 / 236 |
| Source-window reverse kernel, four-block groups | 255 | 446 | 20,480 | 260 / 231 | 342 / 377 |

Local load/store counts are static sites in NVIDIA's compiled machine code,
not bytes transferred or executed instruction counts. Spill figures are compiler
resource reports, not a runtime count of spill events.

The guarded forward uses a fast block-wide softmax scale when safe, then falls
back to per-prefix scaling for large score jumps. Adversarial tests introduce
jumps around 128 and repeat them across blocks; outputs and final normalization,
key-summary and value-summary states agree with an independent reference within
tolerance. The fallback can cost more than the fast path.

FP32 contractions use BF16 high/remainder products and omit remainder ×
remainder. Long backward checks against an independent stable PyTorch reference
use relative tolerance 0.003 and absolute tolerance 0.001 for FP32 projection
gradients; BF16 public gradients add rounding. The historical combined GPU suite
passed 206 focused tests. After extraction, all 149 packaged regression cases
passed on an RTX 4090, including full-length checks. See the
[packaged regression suite](../tests/README.md); latency tables remain historical.

## Full nanochat training: model and optimizer

This scope uses a **six-layer GPT, width 384, 3 heads, head width 128**, sequence
length 2048. Each optimizer update processes eight microbatches of 16 sequences:
262,144 tokens/update. Model execution is eager, with nanochat's native
MuonAdamW optimizer. Timing includes the embeddings, attention, MLP, language-model
output head, parameter gradients, accumulated microbatches, data copies and
optimizer work.

| Model | Steady ms/update | Tokens/sec | Peak allocated MiB | Total wall min |
|---|---:|---:|---:|---:|
| Causal baseline | 1357.2 | 193153 | 20069.3 | 24.8 |
| Two causal attentions with head-isolated intermediate projections | 1563.4 | 167673 | 21249.1 | 28.5 |
| QK-F / QKV-C, source-window backward | 10854.6 | 24150 | 20964.5 | 194.9 |

Steady time is the median update time after excluding the first ten updates.
Peak allocated memory is the trainer's reported model-run peak, unlike the
attention-only extra-memory columns above. Total wall time
includes evaluation and checkpoint saving. Each model receives 278,396,928
training tokens. Architecture definitions, quality metrics and selection protocol
are in [nanochat results](nanochat.md).
