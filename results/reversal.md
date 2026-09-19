# Sequence reversal: trained and unseen lengths

The model reads a random token sequence, then generates it backward. For example:

```text
Input prompt:    3 1 4 |       # | is a separator token.
Expected output: 4 1 3
```

This tests whether the model can retrieve the right token at each output
position, including at sequence lengths it was not trained on. The current
QK-F / QKV-C variant achieved 82.7% generated-token accuracy and 33.3% exact
sequence accuracy at the trained length, averaged across three seeds. It did
not reliably generalize to unseen lengths in this initial study.

## Models in this comparison

Seven models use the same width-64, four-head Transformer wrapper, with one
attention mixer, residual connections, normalization, an MLP and tied token
embedding/output weights. The attention variants differ as follows. F means
full-prefix attention, C means causal retrieval; projection names describe
one head. Every mixer also has a final output projection.

| Report name | What is different |
|---|---|
| C | Ordinary causal attention; the baseline |
| Original F+C | Five input projections QF/KF/VF/U/QC; one memory U[s] + weighted VF serves as both C key and value |
| F+C2 | Two projections R/V: QF, KF, QC and the causal key base are all R; V is the value base |
| F+C3 | Three projections R/Q/V: QF, KF and the causal key base share R; C has its own query Q |
| QKV-F+C | Three projections Q/K/V: F and C share Q/K, with separate value V |
| QKQV-F+C | Four projections QF/KF/QC/V: independent F/C queries, shared key base KF, separate value V |
| QK-F / QKV-C | Five independent projections QF/KF/QC/KC/VC; the implementation released here |

F+C2 and F+C3 are historical variant names, **not counts of stacked attention
layers**. The current variant was logged as `QFKF-QCKC-VC`; that name refers to
the same five-vector operator. Historical variants are described for context;
their implementations are not included in this minimal release.

## Training, selection and metrics

Each example is sampled from an alphabet of 32 tokens. Append the separator
and the reversed sequence, then train next-token prediction only on the reverse
portion. Source length n gives a complete example of 2n + 1 tokens; the shifted
model input has 2n tokens. Thus the trained source length 32 corresponds to
model input length 64.

| Setting | Value |
|---|---|
| Training source length | 32 |
| Model width / heads / head width | 64 / 4 / 16 |
| Full-run steps / batch size | 10,000 / 64 |
| Supervised reverse tokens per full run | 20,480,000 |
| Precision / position encoding | BF16 / RoPE without additive position embeddings |
| Hyperparameter screens | Four candidate settings per architecture, 2,000 steps each |
| Full-run seeds | 47635772, 47635773, 47635774 |
| Validation examples per evaluation | 64 |
| Final test examples per seed and length | 128 |
| Test source lengths | 8, 16, 24, 32, 40, 48, 64 |

The four short screens select each architecture's initialization, query-score
gains, learning rate and schedule using validation only. Selection ranks final
validation exact-sequence accuracy first, then generated-token accuracy,
teacher-forced token accuracy and loss. Each selected setting is trained afresh
for three seeds. The final 10,000-step checkpoint is tested; it is not selected
using the best test or intermediate validation result.

Training, validation and final tests use separate random generators. All 28
screens and 21 full runs completed, producing 147 seed/length test entries.
The compact variants use the small T ≤ 128 Triton path, rather than the long
kernel benchmarked at T = 2048.

The metrics measure different things:

- **AR token accuracy:** fraction of correct tokens when the model generates
  the entire reverse sequence greedily, using its own previous outputs. AR
  means autoregressive.
- **AR exact accuracy:** fraction of generated sequences with every token correct.
- **Test loss:** next-token cross-entropy in nats per reverse token, with the
  correct previous tokens supplied (teacher forcing).

High teacher-forced performance need not yield high generated-sequence accuracy:
an early generation error changes the context for later predictions.

## Accuracy at the trained length

These are mean test accuracies across the three full-run seeds, expressed as
percentages:

| Architecture | AR token | AR exact sequence |
|---|---:|---:|
| C | 38.1% | 0.0% |
| Original F+C | 97.4% | 35.2% |
| F+C2 | 13.9% | 0.0% |
| F+C3 | 64.5% | 0.0% |
| QKV-F+C | 76.9% | 1.8% |
| QKQV-F+C | 85.0% | 33.1% |
| QK-F / QKV-C | 82.7% | 33.3% |

The means hide substantial seed sensitivity:

| Architecture | Exact accuracy, each seed | Generated-token accuracy, each seed |
|---|---|---|
| Original F+C | 94.5%, 3.9%, 7.0% | See length-wise mean/variation below |
| QKQV-F+C | 0.0%, 96.1%, 3.1% | 64.6%, 99.9%, 90.6% |
| QK-F / QKV-C | 51.6%, 48.4%, 0.0% | 97.4%, 96.8%, 54.0% |

The current model matches Original F+C's full-model parameter count (43,136),
but this tuning has not established a reliable match in quality. At source
length 64 it averages 8.7% generated-token accuracy and no exact reversals.
All its other unseen test lengths also have zero exact reversals.

All seven architectures have zero exact accuracy at every tested source length
larger than 32. Shorter-length results vary by architecture and seed. This
initial tuning budget leaves the causal baseline and several variants underfit;
it does not establish an intrinsic architectural advantage or limitation.

## All tested lengths

The following table reports **mean ± sample standard deviation across three
seeds**, not confidence intervals. Accuracy columns are fractions in [0, 1],
so 0.8271 means 82.71%; loss is in nats per token. Parameters include the whole
reversal model, not only the attention layer. Source length 32 is the only
training length; every other listed length is unseen during training.

| Architecture | Source length | Model length | Seeds | Params | Test loss | AR token | AR exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C | 8 | 16 | 3 | 34944 | 1.1245 ± 0.3149 | 0.6351 ± 0.1684 | 0.0182 ± 0.0180 |
| C | 16 | 32 | 3 | 34944 | 2.1653 ± 0.1438 | 0.3504 ± 0.0719 | 0.0000 ± 0.0000 |
| C | 24 | 48 | 3 | 34944 | 2.4767 ± 0.0468 | 0.2690 ± 0.0650 | 0.0000 ± 0.0000 |
| C | 32 | 64 | 3 | 34944 | 1.9911 ± 0.1365 | 0.3808 ± 0.1032 | 0.0000 ± 0.0000 |
| C | 40 | 80 | 3 | 34944 | 7.5811 ± 2.3734 | 0.1393 ± 0.0617 | 0.0000 ± 0.0000 |
| C | 48 | 96 | 3 | 34944 | 10.0330 ± 2.9586 | 0.0619 ± 0.0109 | 0.0000 ± 0.0000 |
| C | 64 | 128 | 3 | 34944 | 10.9975 ± 3.2555 | 0.0476 ± 0.0050 | 0.0000 ± 0.0000 |
| F+C | 8 | 16 | 3 | 43136 | 7.5924 ± 6.6393 | 0.3695 ± 0.2693 | 0.0208 ± 0.0361 |
| F+C | 16 | 32 | 3 | 43136 | 5.4233 ± 3.0303 | 0.4657 ± 0.1805 | 0.0000 ± 0.0000 |
| F+C | 24 | 48 | 3 | 43136 | 4.5962 ± 3.3913 | 0.5965 ± 0.2459 | 0.0000 ± 0.0000 |
| F+C | 32 | 64 | 3 | 43136 | 0.0951 ± 0.0764 | 0.9738 ± 0.0199 | 0.3516 ± 0.5144 |
| F+C | 40 | 80 | 3 | 43136 | 13.5487 ± 1.2727 | 0.1365 ± 0.0807 | 0.0000 ± 0.0000 |
| F+C | 48 | 96 | 3 | 43136 | 14.4280 ± 1.1039 | 0.0785 ± 0.0547 | 0.0000 ± 0.0000 |
| F+C | 64 | 128 | 3 | 43136 | 13.2423 ± 0.6312 | 0.0456 ± 0.0070 | 0.0000 ± 0.0000 |
| F+C2 | 8 | 16 | 3 | 30848 | 1.4528 ± 0.1114 | 0.2051 ± 0.0391 | 0.0000 ± 0.0000 |
| F+C2 | 16 | 32 | 3 | 30848 | 2.1225 ± 0.0674 | 0.1711 ± 0.0149 | 0.0000 ± 0.0000 |
| F+C2 | 24 | 48 | 3 | 30848 | 2.4823 ± 0.0327 | 0.1534 ± 0.0175 | 0.0000 ± 0.0000 |
| F+C2 | 32 | 64 | 3 | 30848 | 2.4445 ± 0.0076 | 0.1392 ± 0.0043 | 0.0000 ± 0.0000 |
| F+C2 | 40 | 80 | 3 | 30848 | 3.8267 ± 0.1261 | 0.1032 ± 0.0127 | 0.0000 ± 0.0000 |
| F+C2 | 48 | 96 | 3 | 30848 | 5.3257 ± 0.0833 | 0.0702 ± 0.0117 | 0.0000 ± 0.0000 |
| F+C2 | 64 | 128 | 3 | 30848 | 6.1064 ± 0.3793 | 0.0299 ± 0.0247 | 0.0000 ± 0.0000 |
| F+C3 | 8 | 16 | 3 | 34944 | 1.0709 ± 1.7721 | 0.7708 ± 0.3322 | 0.5833 ± 0.5061 |
| F+C3 | 16 | 32 | 3 | 34944 | 1.6614 ± 1.4707 | 0.6068 ± 0.3105 | 0.0052 ± 0.0090 |
| F+C3 | 24 | 48 | 3 | 34944 | 1.6443 ± 0.8318 | 0.5677 ± 0.2471 | 0.0000 ± 0.0000 |
| F+C3 | 32 | 64 | 3 | 34944 | 0.9602 ± 0.5807 | 0.6453 ± 0.2618 | 0.0000 ± 0.0000 |
| F+C3 | 40 | 80 | 3 | 34944 | 6.1896 ± 0.2312 | 0.2639 ± 0.1228 | 0.0000 ± 0.0000 |
| F+C3 | 48 | 96 | 3 | 34944 | 8.4446 ± 1.2780 | 0.1198 ± 0.0801 | 0.0000 ± 0.0000 |
| F+C3 | 64 | 128 | 3 | 34944 | 7.6337 ± 3.2767 | 0.0395 ± 0.0004 | 0.0000 ± 0.0000 |
| QK-F / QKV-C | 8 | 16 | 3 | 43136 | 5.7727 ± 3.5060 | 0.4131 ± 0.1619 | 0.0000 ± 0.0000 |
| QK-F / QKV-C | 16 | 32 | 3 | 43136 | 4.0280 ± 1.9928 | 0.4360 ± 0.0830 | 0.0000 ± 0.0000 |
| QK-F / QKV-C | 24 | 48 | 3 | 43136 | 3.3248 ± 0.9885 | 0.5667 ± 0.2155 | 0.0000 ± 0.0000 |
| QK-F / QKV-C | 32 | 64 | 3 | 43136 | 0.4195 ± 0.5913 | 0.8271 ± 0.2485 | 0.3333 ± 0.2891 |
| QK-F / QKV-C | 40 | 80 | 3 | 43136 | 8.0178 ± 2.2813 | 0.2620 ± 0.2065 | 0.0000 ± 0.0000 |
| QK-F / QKV-C | 48 | 96 | 3 | 43136 | 8.9840 ± 2.4074 | 0.1552 ± 0.1555 | 0.0000 ± 0.0000 |
| QK-F / QKV-C | 64 | 128 | 3 | 43136 | 9.3110 ± 2.9674 | 0.0866 ± 0.0731 | 0.0000 ± 0.0000 |
| QKQV-F+C | 8 | 16 | 3 | 39040 | 4.2146 ± 1.4009 | 0.4801 ± 0.1730 | 0.0026 ± 0.0045 |
| QKQV-F+C | 16 | 32 | 3 | 39040 | 3.0269 ± 1.2627 | 0.5124 ± 0.1841 | 0.0000 ± 0.0000 |
| QKQV-F+C | 24 | 48 | 3 | 39040 | 2.0235 ± 1.1229 | 0.6150 ± 0.2577 | 0.0026 ± 0.0045 |
| QKQV-F+C | 32 | 64 | 3 | 39040 | 0.2889 ± 0.3915 | 0.8503 ± 0.1827 | 0.3307 ± 0.5460 |
| QKQV-F+C | 40 | 80 | 3 | 39040 | 6.7465 ± 1.3766 | 0.2951 ± 0.1033 | 0.0000 ± 0.0000 |
| QKQV-F+C | 48 | 96 | 3 | 39040 | 9.1062 ± 1.9480 | 0.1679 ± 0.0869 | 0.0000 ± 0.0000 |
| QKQV-F+C | 64 | 128 | 3 | 39040 | 10.1117 ± 2.1351 | 0.1087 ± 0.0505 | 0.0000 ± 0.0000 |
| QKV-F+C | 8 | 16 | 3 | 34944 | 4.1351 ± 1.2170 | 0.3197 ± 0.0389 | 0.0000 ± 0.0000 |
| QKV-F+C | 16 | 32 | 3 | 34944 | 4.4118 ± 1.1646 | 0.3783 ± 0.0295 | 0.0000 ± 0.0000 |
| QKV-F+C | 24 | 48 | 3 | 34944 | 3.9498 ± 1.5519 | 0.3800 ± 0.1624 | 0.0000 ± 0.0000 |
| QKV-F+C | 32 | 64 | 3 | 34944 | 0.6448 ± 0.3703 | 0.7694 ± 0.1201 | 0.0182 ± 0.0316 |
| QKV-F+C | 40 | 80 | 3 | 34944 | 6.4478 ± 0.7985 | 0.2288 ± 0.0677 | 0.0000 ± 0.0000 |
| QKV-F+C | 48 | 96 | 3 | 34944 | 7.0674 ± 1.3632 | 0.1350 ± 0.0246 | 0.0000 ± 0.0000 |
| QKV-F+C | 64 | 128 | 3 | 34944 | 7.3989 ± 1.0354 | 0.0575 ± 0.0046 | 0.0000 ± 0.0000 |

## Selected setting for QK-F / QKV-C

Validation selected normal initialization with standard deviation 0.04,
F query gain 2, C query gain 4, learning rate 0.001 and a constant schedule.
The gains multiply QF and QC respectively before score calculation; they change
the attention sharpness without adding parameters. The full model has 43,136
parameters. RoPE rotates QF/KF/QC/KC, never VC. This model initialization differs
from the minimal package layer's ordinary `nn.Linear` defaults.

## Earlier two-layer baseline: a different budget

An earlier study included **Cx2**, two ordinary causal Transformer blocks, each
with its own attention, residual connections and MLP. Its source-length-32
RoPE follow-up averaged 99.7% generated exact accuracy, but used 20,000 steps,
batch size 128 and 81.92 million supervised tokens—four times this compact
study's per-run budget. Its tuning/protocol also differed. It cannot be ranked
directly against the current variant from these numbers.

Further balanced validation-only work on initialization, gains, learning-rate
schedules, training duration and seed robustness is needed before comparing
architectures decisively. This report summarizes past experiments; their
training scripts, raw curves and checkpoints are not included in this release.
