# Future work: three matrix formulations

FA-KV means **future-aware key/value attention**. The [README](README.md)
defines its full-prefix-then-causal schedule. This note shows three matrix
formulations of the concrete layer included in this repository. They compute
the same result with different arithmetic and memory costs. The cubic
implementation is in [`attention.py`](src/future_aware_kv/attention.py).
Executable versions of the materialized-prefix and block-prefix formulations
are in [`experimental.py`](src/future_aware_kv/experimental.py); tests compare
their position-free core and every input gradient with the cubic reference.
The additive window and ALiBi terms shown below are omitted from those small
prototypes.

The pseudocode describes one attention head. Batch and head dimensions are
omitted. Numerical-stability details are also omitted so that the dataflow
remains visible.

## Names and dimensions

`Q`, `K`, and `V` mean query, key, and value. `F` means the inner full-prefix
attention, `C` means the outer causal attention, and `0` marks the unchanged
source shortcut. The final `K` or `V` in an `F` projection name says which
causal-attention object that full-prefix branch constructs.

| Name | Full meaning |
|---|---|
| `QFK` | Query for full-prefix attention that constructs contextual causal keys |
| `KFK` | Key for full-prefix attention that constructs contextual causal keys |
| `QFV` | Query for full-prefix attention that constructs contextual causal values |
| `KFV` | Key for full-prefix attention that constructs contextual causal values |
| `QC` | Query for the outer causal attention |
| `KC0` | Causal-attention key, unchanged shortcut part |
| `KCF` | Causal-attention key payload transported through full-prefix attention |
| `VC0` | Causal-attention value, unchanged shortcut part |
| `VCF` | Causal-attention value payload transported through full-prefix attention |
| `gK` | Causal-key gate: shortcut coefficient `gK`, full-prefix coefficient `1-gK` |
| `gV` | Causal-value gate: shortcut coefficient `gV`, full-prefix coefficient `1-gV` |
| `Y` | Output of the complete FA-KV layer before the head-mixing projection |

The dimensions used below are:

| Name | Full meaning |
|---|---|
| `T` | Sequence length |
| `d` | Width of one attention head |
| `B` | Number of tokens in one sequence block |
| `number_of_blocks` | `T // B`; the pseudocode assumes `T` is divisible by `B` |

Every projection has shape `[T, d]`. The public candidate applies RoPE to the
four F query/key projections and adds a learned window prior shared by its key
and value paths. During training the prior is the log of smooth sigmoid gates
for past and future distance. Hard inference replaces it with zero inside the
learned integer boundaries and `-inf` outside them:

```python
key_full_scores = rope(QFK) @ rope(KFK).T / sqrt(d) + window_log_prior
value_full_scores = rope(QFV) @ rope(KFV).T / sqrt(d) + window_log_prior
```

The outer C attention adds `causal_alibi[target, source]`. These position terms
do not change the tensor shapes or asymptotic costs below.

In `einsum` strings, `b` is block, `r` is target position within a block, `j`
is input position within a block, `s` is source position, `t` is global target
position, and `d` is the feature dimension. `tril` keeps the lower-triangular
part, `exp` is elementwise exponentiation, and `cumsum` is a cumulative sum.

## 1. Cubic matrix reference

The literal three-loop definition costs `O(T³d)`: for every target and source,
it sums a prefix of `d`-wide payload vectors. The included reference factors
the payload vectors out of that loop. It stores only `[T,T]` matrices but
performs two cubic `[T,T] @ [T,T]` multiplications.

```python
key_full_weight = exp(key_full_scores)      # [source, input]
value_full_weight = exp(value_full_scores)

key_prefix_denominator = cumsum(key_full_weight, dim=1)    # [source, target]
value_prefix_denominator = cumsum(value_full_weight, dim=1)

# Contextual-key contribution to every causal logit, without storing keys.
shortcut_key_logits = QC @ KC0.T / sqrt(d)    # [target, source]
payload_key_logits = QC @ KCF.T / sqrt(d)     # [target, input]
contextual_key_sum = tril(payload_key_logits) @ key_full_weight.T

causal_logits = gK * shortcut_key_logits
causal_logits += (1 - gK) * contextual_key_sum / key_prefix_denominator.T
causal_logits += causal_alibi

causal_mask = arange(T)[:, None] >= arange(T)[None, :]  # [target, source]
causal_logits = where(causal_mask, causal_logits, -inf)
causal_weight = softmax(causal_logits, dim=1)            # [target, source]

# Contextual-value contribution, without storing contextual values.
value_source_weight = causal_weight / value_prefix_denominator.T
contextual_value_weight = tril(value_source_weight @ value_full_weight)

Y = gV * (causal_weight @ VC0)
Y += (1 - gV) * (contextual_value_weight @ VCF)
```

The factored reference costs `O(T³ + T²d)` rather than the literal
`O(T³d)`, but both are cubic in sequence length:

```text
compute: O(T³ + T²d)
memory:  O(T²)
```

## 2. Materialize every prefix state

The cubic work disappears if the contextual key and value for every
`(source, target)` pair are computed once with cumulative sums.

Executable reference: `materialized_prefix_attention`.

```python
key_full_weight = exp(key_full_scores)      # [source, input]
value_full_weight = exp(value_full_scores)

key_prefix_denominator = cumsum(key_full_weight, dim=1)    # [source, target]
value_prefix_denominator = cumsum(value_full_weight, dim=1)

key_weighted_payload = key_full_weight[:, :, None] * KCF[None, :, :]
value_weighted_payload = value_full_weight[:, :, None] * VCF[None, :, :]

key_prefix_numerator = cumsum(key_weighted_payload, dim=1)  # [source,target,d]
value_prefix_numerator = cumsum(value_weighted_payload, dim=1)

contextual_key = gK * KC0[:, None, :] + (
    (1 - gK) * key_prefix_numerator / key_prefix_denominator[:, :, None]
)

contextual_value = gV * VC0[:, None, :] + (
    (1 - gV) * value_prefix_numerator / value_prefix_denominator[:, :, None]
)

# Contract every target with the contextual keys at its own prefix boundary.
causal_logits = einsum("td,std->ts", QC, contextual_key) / sqrt(d)
causal_logits += causal_alibi
causal_mask = arange(T)[:, None] >= arange(T)[None, :]  # [target, source]
causal_logits = where(causal_mask, causal_logits, -inf)
causal_weight = softmax(causal_logits, dim=1)

# Contract every target with contextual values at the same boundary.
Y = einsum("ts,std->td", causal_weight, contextual_value)
```

The two `[source,target,d]` prefix tensors dominate memory:

```text
compute: O(T²d)
memory:  O(T²d)
```

## 3. Store block-boundary prefixes and reconstruct the rest

The cubic reference avoids `[T,T,d]` tensors by doing cubic `[T,T]` products.
The full-prefix formulation removes the cubic work but stores those large
tensors. A block formulation stores prefix states only at block boundaries and
reconstructs every target inside a block with batched matrix multiplications.

Executable reference: `block_prefix_attention`.

First reshape the input axis into blocks. This creates all block views at once;
there is no Python loop over blocks.

```python
number_of_blocks = T // B

key_full_weight = exp(key_full_scores)
value_full_weight = exp(value_full_scores)

# [source, block, local_input] -> [block, local_input, source]
key_weight_blocks = key_full_weight.view(T, number_of_blocks, B).permute(1, 2, 0)
value_weight_blocks = value_full_weight.view(T, number_of_blocks, B).permute(1, 2, 0)

causal_query_blocks = QC.view(number_of_blocks, B, d)   # [block,local_target,d]
key_payload_blocks = KCF.view(number_of_blocks, B, d)   # [block,local_input,d]
value_payload_blocks = VCF.view(number_of_blocks, B, d) # [block,local_input,d]
```

The `block` axis has length `number_of_blocks = T // B`. The next axis has
length `B`: it is `local_target` when the original global target axis of `QC`
is blocked, and `local_input` when the input/payload axis is blocked.

Summarize every block independently, then take an exclusive cumulative sum over
the block dimension. `exclusive_cumsum` returns zero for block 0, so each entry
contains the state immediately before its block.

```python
block_key_denominator = key_weight_blocks.sum(dim=1)      # [block,source]
block_value_denominator = value_weight_blocks.sum(dim=1)

block_key_numerator = einsum(
    "bjs,bjd->bsd", key_weight_blocks, key_payload_blocks
)                                                             # [block,source,d]
block_value_numerator = einsum(
    "bjs,bjd->bsd", value_weight_blocks, value_payload_blocks
)

key_denominator_before = exclusive_cumsum(block_key_denominator, dim=0)
value_denominator_before = exclusive_cumsum(block_value_denominator, dim=0)
key_numerator_before = exclusive_cumsum(block_key_numerator, dim=0)
value_numerator_before = exclusive_cumsum(block_value_numerator, dim=0)
```

All target blocks are then processed in one batch. The exact state for each
target is its block-boundary state plus its prefix inside the current block.

```python
# Prefix denominators inside each block: [block,local_target,source].
key_denominator = (
    key_denominator_before[:, None, :] + cumsum(key_weight_blocks, dim=1)
)

value_denominator = (
    value_denominator_before[:, None, :] + cumsum(value_weight_blocks, dim=1)
)

# Causal query dotted with key numerators from earlier complete blocks.
key_logit_past = einsum(
    "brd,bsd->brs", causal_query_blocks, key_numerator_before
) / sqrt(d)

# Causal query dotted with key payloads inside the current block.
local_key_geometry = einsum(
    "brd,bjd->brj", causal_query_blocks, key_payload_blocks
) / sqrt(d)
local_causal_mask = arange(B)[:, None] >= arange(B)[None, :]
local_key_geometry = where(local_causal_mask, local_key_geometry, 0)
key_logit_local = einsum(
    "brj,bjs->brs", local_key_geometry, key_weight_blocks
)

shortcut_key_logits = einsum("brd,sd->brs", causal_query_blocks, KC0) / sqrt(d)
causal_logits = gK * shortcut_key_logits
causal_logits += (1 - gK) * (key_logit_past + key_logit_local) / key_denominator
causal_logits += causal_alibi.view(number_of_blocks, B, T)

target_position = arange(T).view(number_of_blocks, B)
source_position = arange(T)
causal_mask = source_position[None, None, :] <= target_position[:, :, None]
causal_logits = where(causal_mask, causal_logits, -inf)
causal_weight = softmax(causal_logits, dim=2)  # [block,local_target,source]

# Contextual values from earlier complete blocks.
value_source_weight = causal_weight / value_denominator
value_past = einsum(
    "brs,bsd->brd", value_source_weight, value_numerator_before
)

# Contextual values contributed inside the current block.
value_local_weight = einsum(
    "brs,bjs->brj", value_source_weight, value_weight_blocks
)
value_local_weight = where(local_causal_mask, value_local_weight, 0)
value_local = einsum(
    "brj,bjd->brd", value_local_weight, value_payload_blocks
)

shortcut_value = einsum("brs,sd->brd", causal_weight, VC0)
output_blocks = gV * shortcut_value + (1 - gV) * (value_past + value_local)
Y = output_blocks.reshape(T, d)
```

The block summaries and the contractions with block-boundary states cost
`O(T²d)`. The local block contractions cost `O(T²B)`. The numerator checkpoints
have shape `[T/B,T,d]`:

```text
compute: O(T²d + T²B)
memory:  O(T² + T²d/B)
```

Choosing `B` proportional to `d` gives:

```text
compute: O(T²d)
memory:  O(T²)
```

## Relation to the original online algorithm

`O(T²)` memory is not a lower bound for FA-KV. The online schedule in the
README keeps only the current denominator and `d`-wide numerator for every
source. Its forward pass has:

```text
compute:      O(T²d)
active state: O(Td)
```

The output itself has `T*d` elements, so a forward pass that returns the complete
sequence requires memory proportional to at least `T*d`. The `O(Td)` online
schedule meets that lower bound. Dense causal attention must also evaluate a
number of source-target pairs proportional to `T²`, so its dependence on
sequence length cannot be subquadratic. The online FA-KV schedule therefore has
optimal asymptotic scaling in `T`; the factor `d` is the vector work for each
pair.
