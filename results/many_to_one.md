# Many-to-one matching

This experiment asks whether a model can bind many earlier member tokens to a
label that appears later. It compares one public FA-KV candidate with a
two-layer causal-attention baseline.

## Task

Each example starts with four groups. A group contains `m` unique member tokens
followed by one label token. You can read members as lowercase letters and
labels as uppercase letters:

```text
a b c d A   e f g h B
```

The model receives every member as an independent final query and predicts the
next label to its right. Group boundaries are shifted independently for every
sequence so a fixed token position does not identify the answer. For example,
a shift of three plus a fresh final group gives:

```text
body:     d A e f g h B a b c C
queries:  a b c d   e f g h
targets:  C C C A   B B B B
```

All queries share the body computation but cannot see one another.

## Models

- **2A:** two ordinary causal-attention+MLP blocks with RoPE.
- **FA-KV:** one future-aware-attention+MLP block. F uses RoPE; C uses ALiBi.
  Each F head learns a soft past distance and future distance shared by its key
  and value paths. Training anneals the window temperature from `4.0` to
  `0.25`. Evaluation floors the learned distances and uses an exact hard
  window, so inputs outside it receive zero probability.

Both models use `d_model=64`, two heads of width 32, tied input/output token
embeddings, and no learned absolute-position table. The 2A model has 329,088
parameters; FA-KV has 320,456. Copying Q into K at initialization does not tie
the weights: the matrices train independently.

## Protocol

Hyperparameters were selected using validation-only runs. Each architecture
received 12 initial recipes at `m=32` for 600 steps; its best three were
promoted to `m=64` for 1,500 steps. The frozen recipes were then trained for
3,000 steps with fresh seeds 20, 21, and 22 at every `m`.

The selected 2A recipe uses AdamW with learning rate `0.002`, weight decay
`0.01`, and copied Q/K initialization. The selected FA-KV recipe uses learning
rate `0.008`, weight decay `0.01`, bypass-gate initialization `0.5`, outer
ALiBi scale `1/64`, past-window initialization `-0.05 × sequence length`, and
future-window initialization `0.25 × sequence length`.

Each training update contains 2,048 queried member targets. Reported numbers
are mean ± sample standard deviation over three held-out test sets, each made
from 64 fresh batches. FA-KV is always evaluated with its hard window.

<!-- generated-results:start -->
| m | 2A test accuracy | FA-KV hard-window test accuracy |
|---:|---:|---:|
| 2 | 99.87% ± 0.06% | 100.00% ± 0.00% |
| 4 | 99.83% ± 0.03% | 100.00% ± 0.00% |
| 8 | 96.88% ± 2.54% | 99.99% ± 0.01% |
| 16 | 99.87% ± 0.08% | 98.28% ± 2.49% |
| 32 | 86.65% ± 0.34% | 99.98% ± 0.01% |
| 64 | 84.17% ± 1.25% | 99.45% ± 0.51% |
| 128 | 24.90% ± 0.03% | 19.28% ± 0.62% |
<!-- generated-results:end -->

The central comparison is `m=64`: hard-window FA-KV reaches 99.45% while 2A
reaches 84.17%, despite FA-KV having fewer parameters and one attention+MLP
block instead of two. At `m=128`, neither model learns the task.

At `m=64`, every FA-KV seed learns a past boundary of `-10` for both heads,
which excludes past inputs at hard inference. Its future boundaries range from
57 to 102 tokens, roughly the distance needed to reach a later group label.

The curve is not monotonic. In particular, one FA-KV `m=16` seed converts its
soft window poorly and lowers the three-seed mean to 98.28%. The per-seed
values and learned integer windows are preserved in
[`many_to_one.csv`](many_to_one.csv).

This is a synthetic binding result, not evidence about language-model quality.
Three seeds are enough to expose the transition but not to estimate failure
rates precisely.

## Reproduce

From the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
./run_all.sh
```

`run_all.sh` runs the tests, all 42 frozen-recipe training runs, and the report
generator. Set `PYTHON_BIN`, `DEVICE`, or `RESULT_ROOT` to override their
defaults. Completed summaries are skipped, so interrupted grids can resume.
