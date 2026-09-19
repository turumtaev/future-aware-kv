# Small language-model experiment with nanochat

A six-layer nanochat model using QK-F / QKV-C reached test **1.014182 bits per
byte**, versus **1.017221** for the causal baseline, after the same training-token
budget. Its updates took about 8× longer. This single-seed experiment demonstrates
that the layer can train inside a language model; it does not establish a reliable
quality improvement.

The runs completed on one RTX 4090 (24 GiB). The study replaced
attention while retaining nanochat's surrounding model and native optimizer.
The training harness and checkpoints are not included in this minimal release.

## What was compared

| Model | Attention inside each GPT block | Historical run name |
|---|---|---|
| Causal baseline | One ordinary causal attention with dense Q/K/V/output projections | `A` |
| Two causal attentions, head-isolated | First causal attention, per-head intermediate projection/normalization, per-head second Q/K/V projections, then second causal attention | `AA-head-isolated` |
| QK-F / QKV-C | Full-prefix updates of separate causal K/V, then causal retrieval, independently within each head | `two_geometry` |

The two-attention control does not mix heads between attentions: intermediate
projections and normalization operate within each head. Only its final dense
output projection mixes head outputs. It has no intermediate residual or MLP
between the two attentions. This makes head isolation explicit, but does not
match parameter counts exactly.

All three use nanochat's residual structure, MLPs, embeddings, value embeddings
and output head. QK-F / QKV-C uses native position encoding and Q/K normalization,
plus a fixed 1/√2 gain on KC/VC. Those model settings differ from constructing the
minimal layer with its default initialization; see
[layer integration](../KERNELS.md#layer-integration).

## Shared training and evaluation settings

| Setting | Value |
|---|---|
| Depth / model width | 6 GPT layers / 384 channels |
| Attention heads / head width | 3 / 128 |
| Sequence length | 2048 tokens |
| Precision | BF16 |
| Position encoding | Native RoPE; full-context attention in every layer |
| Seed | 17 |
| Microbatch / accumulation | 16 sequences per microbatch, 8 microbatches per update |
| Tokens per optimizer update | 262,144 |
| Optimizer updates / total training tokens | 1062 / 278,396,928 |
| Optimizer and schedule | Native MuonAdamW; the baseline's schedule shared by all models |
| Dataset | `karpathy/climbmix-400b-shuffle` |
| Tokenizer vocabulary | 32,768, shared across all models |
| Data layout | Native beginning-of-sequence best-fit packing into fixed rows |
| Validation / test evaluation tokens | 1,048,576 per split |

The models see identical packed training rows and evaluation rows. Validation
uses dataset shard 06542; the isolated test split uses shard 06541. Each
validation/test evaluation has 32 batches of 16 × 2048 tokens. A shared data
layout matters: changing context lengths or row boundaries can change loss
without changing the model.

Each run logged 1062 training updates and 13 validation evaluations. Select the
checkpoint with the lowest validation bits per byte, then evaluate the test
split once. All three selected checkpoints were at update 1062. Test metrics
were not used for checkpoint selection.

## Final results

**BPB** means bits per byte: the summed negative log-probability in bits divided
by the number of represented data bytes. It accounts for token lengths rather
than averaging per token. **Test loss** is mean next-token cross-entropy in nats
per token. Lower values are better for both. Zero-byte special tokens are excluded
from the byte-normalized numerator, following the evaluator's convention.

| Model | Val BPB ↓ | Test BPB ↓ | Test loss ↓ | Total min | Steady ms/update | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| Causal baseline | 0.995121 | 1.017221 | 3.312540 | 24.8 | 1357.2 | 73,531,538 |
| Two causal attentions, head-isolated | 1.010996 | 1.033615 | 3.365860 | 28.5 | 1563.4 | 74,711,186 |
| QK-F / QKV-C | 0.991707 | 1.014182 | 3.302501 | 194.9 | 10854.6 | 75,301,010 |

Steady update time is the median after excluding the first ten updates; total time
includes validation, test evaluation and checkpoint saving. These are **whole
model/optimizer timings**, not attention-kernel timings.
[Benchmarks](benchmarks.md) separates those scopes and reports memory.

QK-F / QKV-C's test BPB is lower by 0.003039 (0.299%) in this seed, with 2.41%
more model parameters. The difference is small. The models share native settings,
but this study has not performed balanced validation-only tuning of initialization
and hyperparameters, or repeated seeds. Neither the baseline nor the new operator
should be declared better from this run.

## What this experiment does not answer

This is an equal-token-budget comparison at one fixed depth. It is not an
equal-compute or equal-wall-time comparison, and it does not determine the best
model size/training horizon for a compute budget. Ordinary Transformer FLOP
estimates do not account for all the extra QK-F / QKV-C work.

The [nanochat miniseries discussion](https://github.com/karpathy/nanochat/discussions/420)
examines model size and training horizon under compute budgets. This experiment
does not reproduce that miniseries protocol. BPB numbers from other nanochat
runs require matching data, tokenizer, context/evaluation layout and training
settings before direct comparison.

The run also does not measure generation speed or validate the minimal layer's
reference decode in a complete language-model deployment.

## Run records

The QK-F / QKV-C run initially encountered GPU allocation fragmentation after
validation. Its successful implementation releases unused cached CUDA allocations
after evaluation; the model, data, optimizer and kernel settings were unchanged.
Training completed after that memory-management fix.

Scripts, raw curves, model/resume checkpoints and tokenizer files are not included
in this layer release. These numbers summarize past training experiments, rather
than a new quality evaluation of the packaged default layer.
