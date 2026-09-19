# Future-aware KV: full-prefix attention inside causal attention

**Old keys and values can incorporate later tokens without breaking causality.**
This repository explores a pattern: full-prefix attention updates old memories
from newly arrived tokens, then causal attention reads those updated memories.
At output position `t`, an old source `s` can incorporate inputs after `s`,
but only within the visible prefix `0..t`.

The useful observation is that this does not require rerunning full attention
from scratch on every prefix, which would cost O(T³d). With fixed source queries
and input projections, each source's full-prefix attention summary can be updated
online. Fold those updates into the causal attention computation: add each new
input once to each source's summary, then retrieve from the updated memories.
This keeps the following scaling for sequence length `T` and head width `d`:

| Operation | Attention cost per head |
| --- | --- |
| Training/prefill forward | O(T²d) work |
| Decode at position `t` | O(td) work per token |
| Forward running state / decode cache | O(Td) memory |

For fixed width and head count, these are **O(T²) prefill, O(T) decode per token,
and O(T) running memory**. Backward checkpoints and scratch have separate memory
costs, described in [KERNELS.md](KERNELS.md). Matching ordinary dense attention's
asymptotic scaling still leaves additional arithmetic, cache updates and GPU
scheduling costs.

The main subject here is this **full-prefix memory update + causal retrieval**
pattern. **QK-F / QKV-C** is one concrete realization, with a multihead PyTorch
layer, a readable reference, and Triton forward/backward kernels. Choices such
as sharing projections, using separate key/value summaries, and combining those
summaries with the original source are parts of the design space. The included
experiments measure this particular realization; they do not establish a quality
advantage for the broader family.

## Why let old KV change?

In a causal Transformer, a token's key/value at a given layer is computed when
that token is processed. Later queries reuse it, but it cannot incorporate tokens
that arrived afterward. For example, a source representing an ambiguous name
cannot revise its representation when a later sentence explains who it means.

What if we update old KV before retrieving from it? At current position `t`,
source position `s <= t` could summarize the whole **visible prefix `0..t`**,
including positions after `s`. Its KV is future-aware relative to `s`. Output
`t` still sees nothing after `t`, so next-token prediction remains causal.
Earlier outputs are not revised; only the memories used for subsequent outputs
change.

## The general schedule

There are two attentions:

- **F: full-prefix attention.** Source `s` reads every input `j` in the currently
  visible prefix, regardless of whether `j` is before or after `s`.
- **C: causal attention.** Current target `t` retrieves from updated sources
  `s <= t`.

An online attention update adds one input to a softmax-weighted summary without
recomputing its previous inputs. The reuse relies on fixed source queries and
fixed input keys/payloads within this layer: adding a token extends the same
attention distribution rather than changing scores for earlier inputs.
The following conceptual training/prefill schedule makes the dependency explicit;
the concrete functions are defined below.

```python
# T = number of input tokens; t = current output; s = memory/source position.
# Each F[s] holds a full-attention summary for source s.
# State types and update functions are defined in the online version below.
F = [empty_state(kind="full") for s in range(T)]
for t in range(T):
    C = empty_state(kind="causal")     # New retrieval accumulator for output t.
    for s in range(T):                # Source updates can run in parallel.
        update_full(F, input=t, source=s)
        if s <= t:                    # Never retrieve from a future source.
            C = update_causal(C, F, source=s, target=t)
    Y[t] = C.output                   # Already normalized by the online update.
```

During training/prefill all source queries are available. Even a source `s > t`
can accumulate private F state ahead of its first retrieval, because the C mask
prevents that state from affecting an earlier output. Each `(s,t)` pair receives
one F update; C reads each causally valid pair once. Work is O(T²d), rather than
recomputing a full-prefix contextualizer from scratch for every output.

This is different from two ordinary stacked causal layers: in those layers,
source `s`'s first-layer output uses only inputs `0..s`. Here its contextual
memory at target `t` uses inputs `0..t`, and continues changing as `t` grows.

## QK-F / QKV-C in plain loops

Here is the concrete variant implemented in this repository. Its five
projections and original-source bypass specify one way to realize the
general schedule above.

Project five vectors per token **per head**:

```python
QF = X @ W_QF  # F queries: which inputs should update each source's memory?
KF = X @ W_KF  # F keys: match arriving inputs against those source queries.
QC = X @ W_QC  # C queries: what does the current target want to retrieve?
KC = X @ W_KC  # Original causal key bases; also payloads transported by F.
VC = X @ W_VC  # Original causal value bases; also payloads transported by F.
```

The following pseudocode describes one head. `dot` is a vector dot product,
`zeros(d)` a zero vector, and `softmax` normalizes a list of scores. `d` is head
width. Batch and head loops are omitted; the same computation runs independently
for every batch item and head.

First, a naive O(T³d) implementation: recompute every source's full-prefix
summary separately for every target. This is the definition, not the fast path.

```python
for t in range(T):                            # Output/query position.
    keys, values = [], []
    for s in range(t + 1):                    # Sources C is allowed to read.
        scores_f = [dot(QF[s], KF[j]) / sqrt(d)
                    for j in range(t + 1)]   # F reads ALL visible inputs.
        weights_f = softmax(scores_f)
        context_k, context_v = zeros(d), zeros(d)
        for j in range(t + 1):
            context_k += weights_f[j] * KC[j]
            context_v += weights_f[j] * VC[j]
        keys.append(KC[s] + context_k)        # Original source + F context.
        values.append(VC[s] + context_v)

    scores_c = [dot(QC[t], keys[s]) / sqrt(d) for s in range(t + 1)]
    weights_c = softmax(scores_c)
    Y[t] = sum(weights_c[s] * values[s] for s in range(t + 1))
```

KC/VC are immutable bases. Only their attention summaries change. Adding the
original KC[s]/VC[s] provides an **original-source bypass**: retrieval keeps a
direct path to the source token alongside its summary of the visible prefix.
This was inspired by GoldFinch's second-value addition and use of original token
embeddings, rather than copying either mechanism exactly.

The same F weights update both K and V, but F and C use separate matching
projections. **Heads are not mixed between F and C.** C head `h` reads only the
contextual KV produced by F head `h`, with no intervening output projection.
Run the single-head computation above independently for each head, then:

```python
# Y_by_head[h][t] = Y[t] from the single-head loops, run for head h.
# Only this final projection mixes heads; there is no output projection for F.
for t in range(T):
    output[t] = concat(*(Y_by_head[h][t] for h in range(heads))) @ W_OC
```

The layer has six bias-free matrices: W_QF, W_KF, W_QC, W_KC, W_VC and W_OC,
for `6 * width²` parameters.

### Online version: reuse each source's prefix summary

Here is a stable online-softmax update, used by both F and C. Each state has
named fields: a scalar maximum score, a scalar rescaled weight sum, and a
normalized output vector. F's output holds a concatenated key/value summary;
C's output holds one retrieved value vector.

```python
class AttentionState:
    def __init__(self, *, maximum=-infinity, denominator=0, output):
        self.maximum = maximum
        self.denominator = denominator
        self.output = output


def empty_state(*, kind):
    # F carries a d-wide key AND d-wide value; C carries a d-wide value.
    payload_width = {"full": 2 * d, "causal": d}[kind]
    return AttentionState(output=zeros(payload_width))


def online_update(state, score, payload):
    M, L, Z = state.maximum, state.denominator, state.output
    next_M = max(M, score)
    old_weight = 0 if L == 0 else L * exp(M - next_M)
    new_weight = exp(score - next_M)
    next_L = old_weight + new_weight
    p = new_weight / next_L
    next_Z = Z + p * (payload - Z)  # Incorporate this input into the mean.
    return AttentionState(maximum=next_M, denominator=next_L, output=next_Z)


def update_full(F, input, source):
    score = dot(QF[source], KF[input]) / sqrt(d)
    payload = concat(KC[input], VC[input])
    F[source] = online_update(F[source], score, payload)


def update_causal(C, F, source, target):
    ZK, ZV = split_into_two_vectors(F[source].output)
    key = KC[source] + ZK
    value = VC[source] + ZV
    score = dot(QC[target], key) / sqrt(d)
    return online_update(C, score, value)


# Training/prefill for ONE head: O(T²d) work and O(Td) running summary state.
F = [empty_state(kind="full") for s in range(T)]
for t in range(T):
    C = empty_state(kind="causal")
    for s in range(T):              # Independent F states across sources.
        update_full(F, input=t, source=s)
        if s <= t:
            C = update_causal(C, F, source=s, target=t)
    Y[t] = C.output

# Repeat per head, then concatenate Y_by_head[h][t] and apply W_OC as above.
```

### Decode: initialize the new source, then update existing sources

At decode position `t`, earlier projections and their F states are cached.
For one head, project the arriving input X[t], initialize its new source state,
then update every live source:

```python
QF[t] = X[t] @ W_QF
KF[t] = X[t] @ W_KF
QC[t] = X[t] @ W_QC
KC[t] = X[t] @ W_KC
VC[t] = X[t] @ W_VC
# If using RoPE, rotate QF/KF/QC/KC at position t before the updates below.

# 1. QF[t] is a NEW query: it has never read earlier inputs.
#    Build its summary of inputs 0..t-1. Existing sources already have theirs.
F.append(empty_state(kind="full"))
for j in range(t):
    update_full(F, input=j, source=t)

# 2. Add the arriving input t to EVERY live source, including the new source.
#    After each update, that source summarizes exactly inputs 0..t.
C = empty_state(kind="causal")
for s in range(t + 1):
    update_full(F, input=t, source=s)
    C = update_causal(C, F, source=s, target=t)
Y[t] = C.output

# The new source's diagonal input t was added only in the second loop.
# Keep original projections and F summaries for the next decode step.
# Repeat for every head, then mix only the C outputs:
# output[t] = concat(*(Y_by_head[h][t] for h in range(heads))) @ W_OC
```

Both loops are linear in the prefix length. Updating old KV adds work but does
not change dense attention's O(td) per-token work or O(td) cache scaling.
Source F updates are independent; C retrieval is a parallelizable softmax
reduction. The included decode is a PyTorch reference, not a tuned Triton kernel.

## GPU implementation

The repository includes Triton kernels for forward and backward. We optimized
until the layer was usable for our small language-model experiments: full
nanochat training updates were about 8× slower than baseline. Kernel-only
comparisons measure different work and are reported separately in
[benchmarks](results/benchmarks.md).

[KERNELS.md](KERNELS.md) explains how the algebra becomes matrix multiplications,
what runs in parallel, and how the kernels handle stability and backward memory.

## Install and use

Python 3.10+. The reference works on CPU; Triton requires Linux/NVIDIA CUDA.
For GPU use, install CUDA-capable PyTorch first.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'       # CPU/reference
# On a CUDA machine instead:
python -m pip install -e '.[cuda,test]'
python -m pytest -q --run-slow
```

```python
import torch
from future_aware_kv import FutureAwareKVAttention

# 768 model channels, 6 independent heads, 128 channels per head.
layer = FutureAwareKVAttention(width=768, heads=6, rope=True).cuda()
x = torch.randn(1, 2048, 768, device="cuda")  # [batch, tokens, model width]

# Current long-sequence Triton kernels require BF16 projections.
# Autocast lets the linear projections produce BF16 from FP32 model inputs.
with torch.autocast("cuda", dtype=torch.bfloat16):
    y = layer(x)                           # Same shape as x.
    loss = y.float().square().mean()       # Example loss, not LM training.
loss.backward()
```

This is an attention mixer: add your model's residual connections, normalization
and MLP around it. RoPE is optional. The layer uses ordinary `nn.Linear`
initialization; the full-model experiments used their own initialization and
training settings.

CPU uses the reference automatically. CUDA uses Triton: sequences up to 128
support FP32/FP16/BF16, while longer sequences currently require BF16 and even
head width at most 128. Invalid shapes/dtypes raise errors. Unsupported arguments
such as `dropout=...`, `attn_mask=...` or `key_padding_mask=...` raise `TypeError`;
they are never silently ignored. Attention-probability dropout is not implemented,
but ordinary dropout around this layer is usable.

The layer assumes every row is a complete sequence; padding is treated as real
input unless you avoid it. Triton supports first derivatives only. This module
has its own API rather than `nn.MultiheadAttention`'s signature.

For a model with its own projections/position encoding, call
`attention(qf, kf, qc, kc, vc)` with five `[batch, tokens, heads, head_width]`
tensors. Apply your chosen RoPE to QF/KF/QC/KC before calling, never VC.
Additional integration details are in [KERNELS.md](KERNELS.md).

## Experiments

- [Benchmarks](results/benchmarks.md): RTX4090 forward/backward latency and
  memory, including width768 / 6 heads / head-width128 / sequence-length2048.
- [Sequence reversal](results/reversal.md): learn to generate an input sequence
  in reverse, then test both trained and unseen sequence lengths.
- [Nanochat](results/nanochat.md): small language-model training at equal token
  budgets. Test BPB was 1.0142 versus baseline 1.0172 in one seed; updates were
  about 8× slower. This does not establish a quality advantage.

These reports describe past experiments; their training scripts and checkpoints
are not included in this minimal layer release. All 149 packaged regression
cases passed on an RTX 4090, including T=2048 output/state and projection
gradient checks for both backward schedules, RoPE and adversarial score gaps.
Installed-layer training/prefill and GPU reference decode/prefill equivalence
also passed. See the [test suite and validation instructions](tests/README.md).
Reference decode has not been latency-benchmarked.

## How we arrived here

[GoldFinch](https://arxiv.org/abs/2407.12077) uses RWKV/Finch recurrence to
construct a reusable cache in linear time, then Transformer attention to read
from it during generation. This suggested separating KV construction from the
component that reads KV to predict the next token.

Our earlier experiment reversed the recurrent scan. At each decode position,
we scanned the visible prefix from right to left, constructing memories that
contained later-token information and immediately retrieving from them:

```python
def decode_reversed_rnn(t):
    # Rebuild right-to-left memories over the visible prefix for this output.
    state = initial_rnn_state()
    retrieval = initial_attention_state()
    for s in range(t, -1, -1):
        state = rnn_update(state, input_at_position=s)
        retrieval = attention_update(retrieval, memory=state, query_at_position=t)
    return retrieval.output
```

A linear scan plus retrieval costs O(td), matching dense attention decode's
sequence-length scaling. Reversal experiments motivated pursuing this direction,
but long recurrent paths were a concern for information retention. Kernel work
also exposed expensive recomputation and numerical instability when recovering
previous recurrent states during backward.

F+C means **full-prefix attention (F) followed by causal attention (C)**.
It replaced the reversed RNN with attention: each source had a fixed query and
an attention summary that grew with the visible prefix. The original F+C used
one contextual memory as both causal key and value. Its first GPU implementation
needed expensive operations that were difficult to express as fast matrix
multiplications, motivating variants with separate keys and values and two
independent query/key pairs.

The current QK-F / QKV-C variant uses the same full-attention weights to update
separate causal keys and values. Its computation can be rearranged into matrix
multiplications. It was practical enough to train a small nanochat model; its
full-model training updates were about 8× slower than baseline in that experiment.

## Related work

[GoldFinch](https://arxiv.org/abs/2407.12077) motivated the separation of KV
construction from retrieval. This implementation is not a reproduction of it.

[RetroAttention](https://arxiv.org/abs/2508.09001) also revises past attention
outputs and overwrites KV during generation. It uses a bounded retrospective
window to recover missed sparse-attention information; our operator instead
updates every source over the growing prefix and is trained as that operator.
These connections do not establish novelty of the exact formulation.
