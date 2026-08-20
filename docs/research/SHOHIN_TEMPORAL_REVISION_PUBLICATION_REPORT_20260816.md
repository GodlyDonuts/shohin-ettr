# Shohin: Model-Owned Temporal Revision and Commitment

> Publication evidence report — updated 2026-08-19
> Primary result: protected Qwen3.5-9B product board  
> Evidence boundary: completed dense transfer; matched source-disjoint MoE
> screens on Qwen 35B, GPT-OSS 117B, and Mixtral 141B; and a 1,023-row
> Mixtral validation

## Abstract

Can distinct learned owners for drafting, revision, and commitment convert an
executed multi-stage inference budget into additional task accuracy?
Shohin tests this question with model-owned temporal roles rather than a
test-time external verifier or answer-level ensemble. On a protected 538-problem board,
a dense Qwen3.5-9B draft → revision → whole-trajectory-commit system solves
`383` problems versus `316` for a matched unchanged second pass. Trained
revision also produces positive aggregate gains across Qwen3.5-0.8B, 4B, and
9B, SmolLM3-3B, and OLMo2-7B. Moving to sparse hosts, a 32,784-parameter
incremental temporal gate reaches `143/256` on Qwen3.6-35B-A3B, versus `111`
for a source-only host and `141` for draft-matched trained revision. On
GPT-OSS-120B, a 1.66M-parameter revision surface reaches `111/256`, versus
`101` source-only and `103` self-refinement, adding 10 and eight answers. On
non-Qwen Mixtral-8x22B, a 3.54M-parameter revision surface reaches
`448/1,023` and beats draft-matched self-refinement by 92 answers, while
exposing severe code and source-only-case retention regressions. These results support
model-owned temporal revision as a parameter-efficient cross-family capability
mechanism, but reject a monotonic MoE scaling law and universal improvement.
The measured auxiliary sparse selector is not conservative, and model-owned
sparse commitment remains untested.

## Contributions

1. **A model-owned temporal architecture.** Drafting, full-trajectory
   revision, and coherent commitment are separately learned roles; no
   correctness label, benchmark router, external verifier, or tool result is
   available at inference time.
2. **A protected end-to-end capability result.** The complete dense 9B system
   adds 67 solved problems over its matched unchanged control on a held
   538-problem board, with immutable model, adapter, runtime, and result
   identities.
3. **Cross-size and cross-family transfer.** The same revision principle
   produces positive aggregate gains on five dense hosts spanning 0.8B–9B
   and three model families.
4. **Three-family MoE transfer.** Learned revision improves both sparse
   controls on Qwen 35B, GPT-OSS 117B, and Mixtral 141B screens; a distinct
   3.54M-parameter surface also beats draft-matched self-refinement on the
   1,023-row Mixtral validation.
5. **Negative results as architecture evidence.** Dense and sparse failures
   localize the unresolved problem: capability-improving revision can regress
   previously correct cases and executable code, while the external auxiliary
   selector restores some capabilities without meeting conservative
   large-host retention. Model-owned sparse commitment remains to be tested.

## Result in one sentence

On a protected 538-problem board, Shohin's same-family
draft → trained revision → learned whole-trajectory commit system solves
`383/538` problems, versus `316/538` for the matched unchanged second pass:
`+67` solved problems and `+8.552` macro-accuracy points (`67.263%` to
`75.815%`).

This is the primary publication claim. It is a measured capability result on
one pinned dense 9B host, not a claim of universal improvement or a completed
MoE scaling law.

The strongest cross-family scale result is now separate and complementary:
on 1,023 source-disjoint rows, a 3.54M-parameter Shohin revision surface scores
`448` correct versus `356` for draft-matched self-refinement. The source-only
host scores `147`. That result demonstrates large sparse-host
capability transfer, while its code and retention regressions establish why
a future model-owned sparse commit is necessary rather than optional.

The third matched MoE family is GPT-OSS-120B: revision scores `111/256`,
versus `101` source-only and `103` for draft-matched self-refinement. Its
`+10` and `+8` gains demonstrate transfer at 117B total / 5.1B active scale,
but `92/101 = 91.09%` retention fails the frozen conservative boundary.

## What the system does

Shohin spends inference-time computation through model-owned temporal roles:

1. a same-family model produces an internal draft;
2. a trained same-family revision owner rewrites the complete trajectory; and
3. a learned commit owner compares the unchanged and revised trajectories and
   chooses one coherent answer.

The deployable system does not receive correctness labels, task-router labels,
external verifier feedback, or tool results at inference time. The published
commit stage selects a whole trajectory rather than splicing answer fragments.

## Architecture

Let `G(θ, p; b)` denote one bounded generation by backbone state `θ` from
prompt `p` with output budget `b`. The dense system first generates a complete
internal draft

`d = G(θ + δ_draft, x; b)`,

then generates a complete revision conditioned on both source and draft,

`r = G(θ + δ_revision, x || d; b)`.

The matched unchanged role generates its own complete second-pass trajectory
`u` from the same source and draft under the same output budget. The commit
owner receives the two coherent candidate trajectories and emits one member
of `{u, r}`. It does not average logits, splice answer fields, or observe which
candidate is correct. In the primary protected system, draft and revision are
same-family Qwen3.5-9B roles; the commit owner is learned exclusively over
their model-owned trajectories.

That matching statement applies to the dense lifecycle. In the sparse Qwen
and Mixtral experiments, unchanged is a source-only host baseline;
self-refinement and trained revision receive the same fixed draft. Comparisons
against sparse unchanged therefore measure an end-to-end draft-conditioned
system difference, while revision versus self-refinement is draft-matched.

For a frozen MoE feed-forward block `m_l` at layer `l`, the transferable
revision surface used on Mixtral is the post-block low-rank residual

`m'_l(h) = m_l(h) + (α / q) B_l A_l h`,

where `q` is the residual rank. Only `A_l` and `B_l` are trained; the native
router, experts, attention, embeddings, and language-model head remain frozen.
The executed Mixtral transfer controls the final 16 layers, uses rank 18 and
`α = 18`, and trains 3,538,944 parameters. Both draft-conditioned Mixtral
arms receive the same immutable Qwen3.6-35B-A3B draft trajectories, making
this a deliberately cross-family model-owned transfer rather than a
same-family draft claim.

The Qwen3.6 causal gate instead freezes two already learned residual branches,
owner `Δ_o,l(h)` and revision `Δ_r,l(h)`, and learns only a tokenwise scalar

`g_l(h) = sigmoid(w_l h + b_l)`.

Its controlled block is

`m'_l(h) = m_l(h) + Δ_o,l(h) + g_l(h)[Δ_r,l(h) - Δ_o,l(h)]`.

Thus `g_l(h)=0` recovers the owner residual, `g_l(h)=1` recovers the revision
residual, and intermediate values form a causal hidden-state blend. The gate
phase trains 32,784 incremental parameters atop two frozen 1,179,648-parameter
branches: 2,392,080 Shohin surface parameters are deployed in total, excluding
the host. No selector target or auxiliary routing label is used in gate training.

![Figure 1: Shohin temporal architecture](figures/shohin_temporal_revision_architecture.svg)

**Figure 1 — Learned ownership across inference time.** (a) A model-owned
draft feeds matched unchanged and trained-revision roles; the commit owner
selects one whole candidate trajectory. (b) On Mixtral, a trained low-rank
post-MLP residual augments each of the final 16 frozen MoE blocks while native
routers and experts remain unchanged. (c) On Qwen3.6-35B-A3B, a tokenwise
causal gate interpolates two frozen temporal residual branches. Blue, orange,
and green denote owner, revision, and commit/gating surfaces; gray modules are
frozen or matched controls.

## Matched experimental design

Every capability comparison holds the evaluation identities and decoding
contract fixed. Dense unchanged and revision receive the same source, draft,
and output budget. In the sparse Qwen and Mixtral experiments, unchanged is
source-only; self-refinement and trained revision receive the same fixed draft.
The dense commit owner sees candidate trajectories but not their labels and
chooses one complete trajectory; it cannot assemble a post-hoc answer from
correct fragments. Source-disjoint screens and holdouts are identity-separated
from training inputs, and protected boards remain outside model-visible
training and generation paths.

Reported gains are paired on exact identities. Where available, the report
preserves discordant win/loss counts and exact two-sided McNemar tests rather
than treating arm accuracies as independent samples. Parameter counts refer
only to trained Shohin surfaces; all reported large-MoE router and expert
weights remain frozen. Predeclared retention and per-domain gates are retained
even when aggregate accuracy improves, which is why several large numerical
gains remain non-promotions.

These are identity- and decoding-matched comparisons, not compute-normalized ones.
The dense commit consumes multiple generated trajectories, and the sparse
draft-conditioned arms consume frozen drafts that source-only unchanged does
not. End-to-end calls, generated tokens, latency, and FLOPs therefore differ;
no equal-call or equal-token superiority claim is made.

![Figure 2: Shohin evidence overview](figures/shohin_temporal_revision_evidence.svg)

**Figure 2 — Capability transfer and its retention boundary.** (a) Trained
revision improves matched dense unchanged on five hosts; `H` and `D` distinguish
holdout from development. (b) Sparse bars explicitly distinguish source-only
from draft-conditioned arms across Qwen, GPT-OSS, and Mixtral. (c) The x-axis
retains source-only-correct cases; the y-axis is each learned sparse treatment
minus its draft-matched comparator. Aggregate gain does not imply conservative
retention. The figure is
reproducible with `pipeline/render_shohin_publication_figure.py`; vector PDF
and SVG are preserved together.

## Relation to prior work

Shohin is closest to four research lines but changes the experimental object in
each. Test-time methods such as
[self-consistency](https://arxiv.org/abs/2203.11171) and compute-optimal
[search or adaptive response refinement](https://arxiv.org/abs/2408.03314)
spend additional inference compute by sampling or searching trajectories,
often with answer aggregation or a verifier. Shohin instead assigns learned
owners to a bounded draft, a complete revision, and a coherent commit; the
reported deployable path does not use correctness feedback or answer-level
majority voting.

[Self-Refine](https://arxiv.org/abs/2303.17651) and
[Reflexion](https://arxiv.org/abs/2303.11366) improve outputs through textual
feedback or reflection. Work on
[intrinsic self-correction](https://openreview.net/forum?id=IkmD3fKBPQ)
also shows that an untrained request to reconsider can degrade reasoning.
Shohin treats generic self-refinement as a matched control: both it and trained
revision receive the same model-owned draft and output budget. The measured
question is therefore whether a learned temporal role adds capability beyond
prompted reconsideration, not whether another inference call is useful.

Parameter-efficient adaptation such as
[LoRA](https://arxiv.org/abs/2106.09685) learns low-rank weight updates, while
[ReFT](https://arxiv.org/abs/2404.03592) learns interventions on frozen hidden
representations. Shohin uses similarly small trainable surfaces but gives them
an explicit temporal contract: the residual is conditioned on a prior
model-owned trajectory, and the causal MoE gate interpolates frozen owner and
revision residual paths token by token.

Conditional-compute work allocates computation across experts, tokens, or
depth. [Switch Transformers](https://arxiv.org/abs/2101.03961) route tokens
among sparse experts; [Mixture-of-Depths](https://arxiv.org/abs/2404.02258)
routes tokens through selected layers; recurrent-depth
[latent reasoning](https://arxiv.org/abs/2502.05171) and
[Coconut](https://arxiv.org/abs/2412.06769) extend internal computation without
requiring every step to be natural language. Shohin's completed MoE results do
not modify the native expert routers. They test a complementary axis: whether
frozen sparse hosts can learn *which temporal state* should own a token or a
whole answer after an explicit draft has already been produced.

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
| Qwen3.5-4B | `529/1,289` vs `371/1,289` (`+12.26 pp`) | `554/1,279` vs `380/1,279` (`+13.60 pp`) | every attribution domain positive |
| Qwen3.5-9B | `589/1,289` vs `464/1,289` (`+9.70 pp`) | `625/1,279` vs `495/1,279` (`+10.16 pp`) | every attribution domain positive; original MATH floor missed |
| SmolLM3-3B | `469/1,289` vs `358/1,289` (`+8.61 pp`) | sealed | cross-family aggregate gain; code retention fails, `4` vs `9` |
| OLMo2-7B | `259/1,289` vs `231/1,289` (`+2.17 pp`) | sealed | positive but too weak to promote |

This table demonstrates that the revision recipe transfers across three Qwen
sizes and to two non-Qwen dense families. It also demonstrates the current
limit: aggregate gains alone do not guarantee conservative retention. The
protected Qwen3.5-4B product board made that boundary concrete: trained
revision improved `272/538` to `320/538`, but GSM8K, MATH, and BBH each
regressed slightly, so its predeclared all-domain-nonregression gate failed.

## First gated MoE transfer

On pinned Qwen3.6-35B-A3B (`35B` total, `3B` active), a 32,784-parameter
temporal residual gate blends frozen owner and revision residuals across the
final 16 MoE layers. It was trained for 256 updates with causal response loss
and zero auxiliary routing supervision.

On a fixed 256-row source-disjoint screen:

| System | Correct | Accuracy |
|---|---:|---:|
| source-only unchanged | `111/256` | `43.359%` |
| draft-matched trained revision | `141/256` | `55.078%` |
| temporal causal gate | **`143/256`** | **`55.859%`** |

The `143`-versus-`111` end-to-end gain is `+32` answers / `+12.5` points,
with 38 paired wins, six paired losses, and exact McNemar `p = 9.4304e-7`.
It does not isolate the gate because unchanged is source-only. Against the
draft-matched trained-revision branch, the gate adds two answers: three
gate-only and one revision-only correct identity (`p = 0.625`). Retention
against source-only is `105/111 = 94.595%`; there are zero empty completions.
Mean revision weight rises from `0.6431` at the first controlled layer to
`0.9028` at the last. These layer means describe the routing profile but do
not by themselves exclude a fixed layer-specific gate.

This is a development screen, not the protected 9B publication confirmation.
It establishes that the gated draft-conditioned system is strongest on this
board; the incremental gate contribution over trained revision is small and
not statistically distinguished from zero.

## Cross-family transfer to a 141B sparse host

Pinned `mistralai/Mixtral-8x22B-Instruct-v0.1` has 141B total and 39B active
parameters. Shohin attaches a shared post-MLP temporal-revision residual to
the final 16 layers, trains exactly 3,538,944 parameters for 256 updates, and
leaves every native router and expert parameter frozen. The intervention is
`0.00251%` of total parameters and `0.00907%` of active parameters.

The matched 256-row source-disjoint screen used identical identities and
decoding for every arm, plus the same fixed Qwen-owned draft trajectories for
the two draft-conditioned arms, self-refinement and trained revision:

| Mixtral screen arm | Correct | Delta vs source-only | Logic | Math | Code |
|---|---:|---:|---:|---:|---:|
| source-only unchanged | `45/256` | — | 35 | 4 | 6 |
| draft-matched self-refinement | `105/256` | `+60` | 75 | 21 | 9 |
| draft-conditioned revision | **`114/256`** | **`+69`** | **81** | **33** | 0 |

Revision records 80 paired wins and 11 paired losses against source-only
(`p = 4.4130e-14`) and improves draft-matched self-refinement by nine answers.
It retains only `34/45 = 75.56%` of source-only-correct identities and loses all 11 code
rows, so the predeclared conservative promotion gate correctly fails despite
the large aggregate effect.

The frozen 1,023-row validation makes the capability effect larger rather
than smaller:

| Mixtral validation arm | Correct | Delta vs source-only | Logic | Math | Code |
|---|---:|---:|---:|---:|---:|
| source-only unchanged | `147/1,023` | — | 106 | 25 | 16 |
| draft-matched self-refinement | `356/1,023` | `+209` | 224 | 121 | 11 |
| draft-conditioned revision | **`448/1,023`** | **`+301`** | **272** | **176** | 0 |
| auxiliary three-way selector | `287/1,023` | `+140` | 222 | 49 | 16 |

The clean draft-matched comparison is 448 versus 356: 131 revision-only and
39 self-refinement-only correct identities (`p = 7.9222e-13`). Against the
source-only host, revision has 353 wins and 52 losses (`p = 4.4389e-56`), a
descriptive end-to-end comparison that also includes draft access. The
auxiliary selector restores source-only-level code but retains `137/147 =
93.20%` of source-only-correct identities—three short of the frozen 95% floor—
and underperforms both draft-conditioned arms. It is therefore a formal failure.

The selector is a task-label-free three-way hashed logistic model over
unchanged, self-refinement, and revision. It is trained from correctness labels
on the 256-row screen and uses source/completion lexical features plus explicit
format and exhaustion markers. It is distinct from the dense architecture's
two-way in-model commit.

The code failure is systematic rather than noisy. An identity-joined replay
of all 1,023 already-scored candidate triples finds that every revision
completion contains a boxed answer and that revision emits only `9.99` tokens
on average (`9` median), versus `436.13` for unchanged and `296.84` for
self-refinement. On the 22 MBPP rows, revision emits a median of `7` tokens:
none contains a code fence, function definition, or return statement. The
unchanged arm has `22/22` fences, `21/22` definitions, and `22/22` returns;
self-refinement has `22/22`, `19/22`, and `21/22`. The learned surface has
acquired an aggressive answer-extraction mode that coincides with higher
scored logic and mathematics accuracy, but the existing analysis does not
separate reasoning improvement from explicit-final-answer compliance. It
violates executable-output modality. The auxiliary selector restores code by
selecting unmodified trajectories. This localizes
the next architecture problem to model-owned output-contract preservation,
not merely more reasoning capacity.

The auxiliary selector already detects part of that boundary without receiving
a task label: it selects unchanged for all `22/22` MBPP identities, thereby
restoring `16` correct programs. Its failure is under-promotion elsewhere. On
504 mathematics identities it selects unchanged `460` times, self-refinement
`34` times, and the much stronger revision only `10` times. The large-host
selector therefore recognizes overt output modality but does not yet preserve
and promote the revision surface's scored answer-valued gains at the same time.

## Scale boundary as of 2026-08-19

| Host | Total / active parameters | Family | Status |
|---|---:|---|---|
| Qwen3.5-0.8B | 0.8B dense | Qwen | completed aggregate revision gain |
| SmolLM3-3B | 3B dense | SmolLM | completed aggregate gain; retention failure |
| Qwen3.5-4B | 4B dense | Qwen | completed gain; protected all-domain gate failure |
| OLMo2-7B | 7B dense | OLMo | completed weak positive / non-promotion |
| Qwen3.5-9B | 9B dense | Qwen | completed protected publication result |
| Qwen3.6-35B-A3B | 35B / 3B MoE | Qwen | completed source-disjoint screen; gate +2 vs draft-matched revision |
| GPT-OSS-120B | 117B / 5.1B MoE | OpenAI | completed 256-row matched screen; `111` revision vs `101` source-only and `103` self-refinement; retention failure |
| Nemotron Super-120B-A12B | 120B / 12B MoE | Nemotron | ModelOpt restoration failed before science; no result claimed |
| Mixtral-8x22B | 141B / 39B MoE | Mistral | completed screen and 1,023-row validation; large aggregate gain, conservative gate failure |
| Nemotron Ultra-550B-A55B | 550B / 55B MoE | Nemotron | prepared only; no result claimed |

The completed evidence supports cross-size and cross-family dense transfer and
positive learned-revision gains on three sparse model families. The new
GPT-OSS point is a completed one-H100 native-MXFP4 measurement: revision gains
10 answers over source-only (19 paired wins, nine losses; `p = 0.0872`) and
eight over draft-matched self-refinement (12 wins, four losses; `p = 0.0768`).
It improves logic by four, mathematics by six, and preserves all 11 MBPP
solutions, but retains only `92/101 = 91.09%` of source-only-correct rows.
Mechanics, fit, 12 independent evaluation allocations, and scoring consume
`26,322` H100-seconds (`7.3117` H100-hours), all with zero retries.

Across the three 256-row MoE points, every learned revision beats source-only
and self-refinement, but the gain is nonmonotonic by active parameters,
retention misses 95% at multiple points, and Mixtral code regresses. The
weighted active-parameter slope is positive under an independent marginal
sampling approximation, while the total-parameter slope interval crosses
zero; only three heterogeneous points are available. The formal conclusion is
therefore **cross-family MoE capability transfer without a supported positive
scaling law**. Nemotron Ultra remains a future high-GPU measurement rather
than a claimed result.

## Immutable evidence

| Artifact | SHA-256 |
|---|---|
| protected 9B product result | `3e86751bb234ee29465885206da5316890060ad8b0b88ea752c4fb012bbf7187` |
| 9B commit holdout result | `9f72644cd39c4880788d5a58b09753bd57f3346810cad3f01226923d6eda5563` |
| independent commit control | `fdf9ead0123001eb14f0f283eb4d4a3463dbddcd48831274213fc0825bb4326b` |
| 9B revision holdout result | `74834cad3ee4c32e1e263d968bbb2f5b1f4dfeb6eca91b124e1a4f5a03148b53` |
| 4B protected product result | `4827808a5d0ec4635e8c72cbdcf23bc6b812f91ffe39b3303f29219854afaada` |
| 35B MoE temporal-gate screen | `1bd645511fd9066aa364e3b4e4c8042b067bb50c1896eaae0810f5c2281b4871` |
| Mixtral 256-row raw score | `ce51617197a9f8e9a8ffdfa08d900746bf7c6cf3c34898d941ebe004f6cc4e50` |
| Mixtral screen evidence report | `618b40e658aa2af1168ac9b1b15114b4dca0aa2c96740e81c8ae224536d2958f` |
| Mixtral 1,023-row raw score | `7befd864dd921ec371c175381b4eccec9f0d603bf291eaddf93fc5039043c3d8` |
| Mixtral validation evidence report | `9d98bff6cc3e244ed175c1ffe3574944ba326ef05b2925ab658eed5f317529b3` |
| Mixtral revision-format and selection analysis | `6d53371877b729747c92ae552ddb5a7a399c937e3f284c276e1c47efde80512c` |
| superseded GPT-OSS mechanics/screen launch receipt | `6107a5b4dd4298dabf3e5889d65df89b1b1e25c2df526c1772007773e987e083` |
| GPT-OSS pre-science admission-failure receipt | `7b4c24b31244a4b79be6a870bd4a6ff2854bb066b26f4cc826561f415ce59c60` |
| corrected GPT-OSS mechanics relaunch receipt | `5ab6381d57ba0ed3f9ff2a0c3b4a7b157d0da1c8b2bed8eaf9dd19d7889b4bec` |
| GPT-OSS pinned-kernel compatibility-failure receipt | `33ea232dbae4c3cb1f48a9f575705fb8c2a58d2a589c4ac8305620bf12bc499e` |
| GPT-OSS 256-row raw score | `906b3b0147e6aeca00e53e9559533665a212bf48162a9758af295915d64a6d59` |
| GPT-OSS screen evidence report | `d60ea1f2147f4e34a668a376507b43d7a4dea11d97cb54834992837e1fb0d810` |
| GPT-OSS Slurm/H100 accounting | `8546d5ba326263d7235eb465a13fcd87081ed5fb346e8f7f8073632b6dfcc36d` |
| three-point MoE scaling analysis | `6434ba95df6c35174fce536e96571d046e9e6fd97459100232120d1976c27550` |
| three-point MoE publication manifest | `70dec71065a3a9f83caa4e58d218cc1831ca89f664a3cab3f036eecacad49c7b` |
| publication architecture figure (SVG) | `2b5a2a25d3ba623b2eb99c469fb1c80e265d4512f605fbef45f66c0a303b07aa` |
| publication architecture figure (PDF) | `df0a324891fb4a7c6e9bafbfedbf58bec1d0dea523c7fd17f2014df2f5f53eed` |
| publication evidence figure (SVG) | `9c405934f3aed5469f27a84e799cc8c7a1ff0dac4aa390a9492fc40aee2aaeac` |
| publication evidence figure (PDF) | `fadf5be75728e6e5bb02272d7b59986c8e8191468555e5cc00a2ce4a05d84213` |
| deterministic six-page preprint (PDF) | `6f319ab3a1615a51286ee7a9ed0bd3ffdb0418e0ba1f96cc5ece91b436019554` |

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
3. A hidden-state temporal gate is the strongest measured system on a
   35B-total / 3B-active MoE development screen, while adding only two answers
   over its draft-matched trained-revision branch.
4. A 3.54M-parameter trained revision surface produces a statistically
   decisive gain over draft-matched self-refinement on a 141B-total /
   39B-active non-Qwen MoE validation.
5. A 1.66M-parameter revision surface improves both sparse controls on a
   third MoE family, GPT-OSS-120B, with nonnegative screen-domain deltas.

## Claims not supported yet

1. Universal per-domain improvement or perfect retention.
2. A unique benefit from antisymmetric commitment.
3. A monotonic or retention-preserving MoE scaling law across 35B, 117B,
   141B, and 550B hosts.
4. A gate-only Qwen effect isolated from draft access, or a compute-normalized
   advantage over equal-call/equal-token baselines.
5. Superiority on unrelated public leaderboard suites that were not part of
   the matched protected evaluation.

Shohin's strongest present conclusion is narrower and useful: model-owned
temporal revision and coherent commitment can convert additional inference
computation into measurable capability, the effect repeats across multiple
dense hosts, a gated draft-conditioned system is strongest on a sparse Qwen
screen, and tiny trained residual surfaces beat draft-matched self-refinement
on GPT-OSS and Mixtral sparse hosts. The same evidence shows that capability
gain and conservative retention are distinct objectives: revision supplies
the gain, while model-owned sparse commitment remains the unresolved
large-host frontier.
