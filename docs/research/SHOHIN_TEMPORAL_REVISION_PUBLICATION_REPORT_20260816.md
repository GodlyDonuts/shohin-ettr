# Shohin: Model-Owned Temporal Revision and Commitment

> Publication evidence report — 2026-08-16  
> Primary result: protected Qwen3.5-9B product board  
> Evidence boundary: completed dense transfer and a source-disjoint 35B MoE
> architecture screen; larger-MoE measurements remain unfinished

## Result in one sentence

On a protected 538-problem board, Shohin's same-family
draft → trained revision → learned whole-trajectory commit system solves
`383/538` problems, versus `316/538` for the matched unchanged second pass:
`+67` solved problems and `+8.552` macro-accuracy points (`67.263%` to
`75.815%`).

This is the primary publication claim. It is a measured capability result on
one pinned dense 9B host, not a claim of universal improvement or a completed
MoE scaling law.

## What the system does

Shohin spends inference-time computation through model-owned temporal roles:

1. a same-family model produces an internal draft;
2. a trained same-family revision owner rewrites the complete trajectory; and
3. a learned commit owner compares the unchanged and revised trajectories and
   chooses one coherent answer.

The deployable system does not receive correctness labels, task-router labels,
external verifier feedback, or tool results at inference time. The published
commit stage selects a whole trajectory rather than splicing answer fragments.

## Primary protected-board result

The solved-count denominator is 538: GSM8K 100, MATH-500 100, executable code
40, GPQA 198, and BBH logic 100. AIME 2024 is a separate 30-problem diagnostic
and is not included in `383/538`.

| Matched Qwen3.5-9B system | Solved | Five-domain macro | GSM8K | MATH | Code | GPQA | BBH | AIME |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unchanged second pass | `316/538` | `67.263%` | 85 | 60 | 34 | 62 | 75 | 4 |
| trained revision | `374/538` | `75.005%` | 88 | 69 | 35 | 104 | 78 | 3 |
| learned whole-trajectory commit | **`383/538`** | **`75.815%`** | 87 | 72 | 35 | 114 | 75 | 6 |
| coherent oracle ceiling, not deployable | `399/538` | `78.619%` | 90 | 77 | 35 | 118 | 79 | 6 |

The learned commit adds nine solved problems over trained revision while
retaining the protected executable-code result (`35/40`, versus `34/40` for
the unchanged control). An independently parameterized commit control solves
`382/538`; therefore the supported finding is useful learned coherent
commitment, not a unique benefit from the published antisymmetric scoring
form. Candidate-order consistency is `100%` with zero swap error.

The product application recorded three prompt truncations. They are disclosed
in the immutable product report and are not silently recoded as clean rows.

## Source-disjoint revision confirmation

Before the protected product board, the trained 9B reviser was evaluated on a
1,279-row source-disjoint holdout. It solved `625/1,279`, versus `495/1,279`
for the matched unchanged second pass: `+130` answers (`+10.16` percentage
points). A learned whole-trajectory commit then solved `652/1,279`.

The original IDR1 promotion rule nevertheless failed because its frozen MATH
floor was missed. That failure is retained in the record; the later protected
product result is the qualified publication result.

## Transfer across dense model sizes and families

Every row uses trained revision versus the matched unchanged second pass over
the same source-disjoint identities. These experiments establish repeatable
aggregate transfer, not monotonic improvement on every domain.

| Dense host | Development | Holdout | Evidence boundary |
|---|---:|---:|---|
| Qwen3.5-0.8B | `323/1,289` vs `236/1,289` (`+6.75 pp`) | `328/1,279` vs `242/1,279` (`+6.72 pp`) | aggregate gain; code retention fails, `8` vs `9` |
| Qwen3.5-4B | `529/1,289` vs `371/1,289` (`+12.26 pp`) | `554/1,279` vs `380/1,279` (`+13.61 pp`) | every attribution domain positive |
| Qwen3.5-9B | `589/1,289` vs `464/1,289` (`+9.70 pp`) | `625/1,279` vs `495/1,279` (`+10.16 pp`) | every attribution domain positive; original MATH floor missed |
| SmolLM3-3B | `469/1,289` vs `358/1,289` (`+8.61 pp`) | sealed | cross-family aggregate gain; code retention fails, `4` vs `9` |
| OLMo2-7B | `259/1,289` vs `231/1,289` (`+2.17 pp`) | sealed | positive but too weak to promote |

This table demonstrates that the revision recipe transfers across three Qwen
sizes and to two non-Qwen dense families. It also demonstrates the current
limit: aggregate gains alone do not guarantee conservative retention. The
protected Qwen3.5-4B product board made that boundary concrete: trained
revision improved `272/538` to `320/538`, but GSM8K, MATH, and BBH each
regressed slightly, so its predeclared all-domain-nonregression gate failed.

## First causal MoE transfer

On pinned Qwen3.6-35B-A3B (`35B` total, `3B` active), a 32,784-parameter
temporal residual gate blends frozen owner and revision residuals across the
final 16 MoE layers. It was trained for 256 updates with causal response loss
and zero auxiliary routing supervision.

On a fixed 256-row source-disjoint screen:

| System | Correct | Accuracy |
|---|---:|---:|
| unchanged | `111/256` | `43.359%` |
| temporal causal gate | **`143/256`** | **`55.859%`** |

The gain is `+32` answers / `+12.5` points, with 38 paired wins, six paired
losses, and an exact two-sided McNemar `p = 9.4304e-7`. Retention is
`105/111 = 94.595%`. Domain deltas are logic `+15`, math `+15`, and executable
code `+2`; there are zero empty completions. Mean revision weight rises from
`0.6431` at the first controlled layer to `0.9028` at the last, showing a
nondegenerate learned causal blend.

This is a development screen, not the protected 9B publication confirmation.
It establishes a causal MoE transfer signal and motivates upward
cross-family measurement.

## Scale boundary as of 2026-08-16

| Host | Total / active parameters | Family | Status |
|---|---:|---|---|
| Qwen3.5-0.8B | 0.8B dense | Qwen | completed aggregate revision gain |
| SmolLM3-3B | 3B dense | SmolLM | completed aggregate gain; retention failure |
| Qwen3.5-4B | 4B dense | Qwen | completed gain; protected all-domain gate failure |
| OLMo2-7B | 7B dense | OLMo | completed weak positive / non-promotion |
| Qwen3.5-9B | 9B dense | Qwen | completed protected publication result |
| Qwen3.6-35B-A3B | 35B / 3B MoE | Qwen | completed source-disjoint causal screen |
| Nemotron Super-120B-A12B | 120B / 12B MoE | Nemotron | staged and held; no result claimed |
| Mixtral-8x22B | 141B / 39B MoE | Mistral | staged and held; no result claimed |
| Nemotron Ultra-550B-A55B | 550B / 55B MoE | Nemotron | prepared only; no result claimed |

The completed evidence supports cross-size dense transfer, cross-family dense
transfer, and one causal sparse-host result. It does **not** yet support a
monotonic MoE scaling law. That claim requires the staged 120B and 141B
matched measurements and, if resources permit, the prepared 550B point.

## Immutable evidence

| Artifact | SHA-256 |
|---|---|
| protected 9B product result | `3e86751bb234ee29465885206da5316890060ad8b0b88ea752c4fb012bbf7187` |
| 9B commit holdout result | `9f72644cd39c4880788d5a58b09753bd57f3346810cad3f01226923d6eda5563` |
| independent commit control | `fdf9ead0123001eb14f0f283eb4d4a3463dbddcd48831274213fc0825bb4326b` |
| 9B revision holdout result | `74834cad3ee4c32e1e263d968bbb2f5b1f4dfeb6eca91b124e1a4f5a03148b53` |
| 4B protected product result | `4827808a5d0ec4635e8c72cbdcf23bc6b812f91ffe39b3303f29219854afaada` |
| 35B MoE temporal-gate screen | `1bd645511fd9066aa364e3b4e4c8042b067bb50c1896eaae0810f5c2281b4871` |

The deployable 9B release binds:

- Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`;
- draft adapter SHA-256
  `854a7cc44fbc2b54418f4e5bd09b7efeed0da44fc9ce217b0bb6b1997b722971`;
- revision adapter SHA-256
  `df3c264d426941fef8ba9c10a90fe9fab304ec2864738209a4d79f9f81e0c473`;
- learned commit SHA-256
  `434d1ec0a8e05d49ee8cc6eaaba1ad36657f507d7416a7417931096c23d2aabc`;
- release manifest SHA-256
  `554e841f71edd3a19063411348340e337532db2db05dd5e1e2adc25a3d347e7b`;
- release `SHA256SUMS` SHA-256
  `0dad031312dec0859e35bb7e9daea8aef688ef350b9053f587fba5acdc9c58c5`.

## Supported claims

1. A trained same-family revision owner materially improves a matched
   unchanged second pass on several dense model sizes and families.
2. Learned whole-trajectory commitment adds measurable capability over the
   trained revision on the protected 9B board.
3. A small hidden-state temporal gate causes a large source-disjoint gain on a
   35B-total / 3B-active MoE development screen.

## Claims not supported yet

1. Universal per-domain improvement or perfect retention.
2. A unique benefit from antisymmetric commitment.
3. A monotonic MoE scaling law across 35B, 120B, 141B, and 550B hosts.
4. Superiority on unrelated public leaderboard suites that were not part of
   the matched protected evaluation.

Shohin's strongest present conclusion is narrower and useful: model-owned
temporal revision and coherent commitment can convert additional inference
computation into measurable capability, the effect repeats across multiple
dense hosts, and a causally trained hidden-state version transfers to a
sparse MoE host.
