# Shohin MoE Frontier Consultation Brief

Status: resolved historical consultation, updated 2026-08-09. Measured results
are separated from hypotheses. No result in this document comes from
benchmark-label exposure at inference.

## The decision originally posed

Design the smallest falsifiable architectural change that makes Shohin's
model-owned temporal revision transfer to a sparse Mixture-of-Experts model.
The dense mechanism is real and repeatedly measured. Two obvious MoE ports
have failed. We need an MoE-native mechanism, not more training of the same
static LoRA, a larger output selector, or an immediate scale-up of a failed
intervention.

The first proving ground must remain pinned
`allenai/OLMoE-1B-7B-0125-Instruct` so that a new mechanism can be compared
against completed controls cheaply and exactly. A larger MoE is a later scale
test, not a substitute for mechanism identification.

## Resolution

The requested attribution and successor ladder were completed after this brief
was drafted. The small-OLMoE lane is now closed rather than awaiting another
proposal:

| Mechanism | Treatment | Strongest matched control | Decision |
|---|---:|---:|---|
| MTR1 shared attention | `204/1289` | unchanged `191` | positive but weak |
| RCR1 router residual | `194` | attention/unchanged `191` | closed |
| ECR1 final-four expert code | `221` | shared `223` | expert codes causally decorative |
| ECR1 all-16-layer | `240` | shared `239` | misses `256` and `+39` gates |
| SER1 selected-expert residual | `201` | shared `241` | closed |
| RME1 revision micro-experts | `232` | shared `248` | closed |
| CTSR1 causal temporal routing | `249` | temporal shared `245`; static shared `248` | route-change gate fails |

ECR zero/mean/permuted expert-code interventions all score exactly `221`.
CTSR1 changes the top-1 expert on only `0.0248%` of measured decisions. Generic
all-layer post-MoE correction is useful, but static expert identity, dedicated
revision experts, and the tested recurrent router do not earn causal inclusion.
No small-OLMoE holdout or 35B MoE scaling is authorized. Exact later contracts
and receipts are in `SHOHIN_ECR1_EXPERT_CONDITIONED_REVISION.md`,
`SHOHIN_SER1_SELECTED_EXPERT_REVISION.md`,
`SHOHIN_RME1_REVISION_MICRO_EXPERTS.md`, and
`SHOHIN_CTSR1_CAUSAL_TEMPORAL_STATE_ROUTING.md`.

## The working dense architecture

Shohin is a two-role, same-backbone temporal computation:

```text
d = DraftOwner(source)
y = RevisionOwner(source, exact_complete_draft=d)
o = optional_whole_trajectory_commit(d, y)
```

Both owners are states of the same pretrained model family. In current dense
experiments they are low-rank role adapters on one shared frozen backbone.
The draft is generated before the revision. The revision state is trained on
complete verified solution trajectories while seeing the exact model-owned
draft. It emits one complete replacement response.

At inference there is no external proposal model, answer verifier, correctness
label, benchmark/task router, symbolic solver, search tool, or teacher. A
deterministic controller serializes two model calls and, where enabled, a
learned model-owned selector chooses one complete candidate. No answer fields
are averaged.

The causal question is whether training a later role to consume the model's
own earlier trajectory improves capability beyond simply spending a second
generation pass. Controls therefore include the untrained same-family second
pass over the identical draft and prompt, generic self-refinement, equal-token
long generation, best-of-two, and equal-update draft-masked training.

## Dense evidence that the changed factor works

### Source-disjoint scale and family tests

| Host | Trained revision | Matched unchanged pass | Absolute gain | Boundary |
|---|---:|---:|---:|---|
| Qwen3.5-0.8B development | `323/1289` | `236/1289` | `+6.749 pp` | aggregate pass; code retention later fails |
| Qwen3.5-0.8B holdout | `328/1279` | `242/1279` | `+6.724 pp` | code `8/9` |
| Qwen3.5-4B development | `529/1289` | `371/1289` | `+12.26 pp` | every broad domain positive |
| Qwen3.5-4B holdout | `554/1279` | `380/1279` | `+13.61 pp` | every broad domain positive |
| Qwen3.5-9B development | `589/1289` | `464/1289` | `+9.70 pp` | every broad domain positive |
| Qwen3.5-9B holdout | `625/1279` | `495/1279` | `+10.16 pp` | original MATH floor narrowly missed |
| SmolLM3-3B development | `469/1289` | `358/1289` | `+8.611 pp` | aggregate transfer; code `4/9` |
| OLMo2-7B development | `259/1289` | `231/1289` | `+2.172 pp` | effect too weak to promote |

SmolLM3 treatment also beats generic self-refinement (`398`), equal-token
long generation (`420`), best-of-two (`339`), and equal-update draft-masked
training (`371`). It therefore cannot be explained only by a second pass,
more output tokens, another fitted adapter, or the final targets.

### Protected product tests

On Qwen3.5-4B, trained revision scores `320/538` against `272/538` for the
matched unchanged second pass, and raises five-domain macro accuracy from
`51.05%` to `61.39%`. Yet GSM8K, MATH-500, and logic regress by two, one, and
two answers. The all-domain gate fails despite the large aggregate gain.

On Qwen3.5-9B, matched product results are:

| System | Solved / 538 | Five-domain macro |
|---|---:|---:|
| unchanged second pass | `316` | `67.263%` |
| trained revision | `374` | `75.005%` |
| learned whole-trajectory commit | `383` | `75.815%` |
| coherent oracle union | `399` | `78.619%` |

An antisymmetric commit head and a matched independent scorer reach 383 and
382 respectively. The commit capability is useful, but antisymmetry is not a
separately supported mechanism. This matters because we do not want a new MoE
proposal to be justified by decorative mathematics that a matched simpler
control can reproduce.

## Exact first MoE host

- Host: `allenai/OLMoE-1B-7B-0125-Instruct`.
- Revision: `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`.
- Topology: 16 decoder layers, hidden width 2,048, 64 experts, eight active
  experts per token, 7B total and approximately 1B active parameters.
- Context: 4,096.
- Data: the same 9,655 training presentations and 1,289 source-disjoint
  development identities used by the dense transfer contract; sealed holdout
  has not been opened.
- Domains: MATH, logic/science, and execution-verified MBPP.
- Training geometry: 256 AdamW updates, batch 1, accumulation 8, learning rate
  `2e-5`, greedy draft/final generation with a 768-token budget.

## MoE experiment 1: shared attention only

MTR1 placed rank-8/alpha-16 LoRA in shared attention token mixers of the final
four layers. Router and expert tensors remained frozen. Trainable capacity was
524,288 parameters.

| Arm | Correct / 1289 | Accuracy |
|---|---:|---:|
| MTR1 temporal revision | `204` | `15.8262%` |
| unchanged second pass | `191` | `14.8177%` |
| generic self-refinement | `169` | `13.111%` |
| equal-token long generation | `167` | `12.956%` |
| best-of-two | `134` | `10.396%` |
| equal-update draft-masked training | `189` | `14.6625%` |

Treatment gains 13 answers and does not regress any broad domain, but misses
the preregistered margins by a wide amount.

The route trace covers 87 rows and 74,935 tokens. Every expert is used in
every layer and normalized entropy is approximately 0.932. Nevertheless,
trained-versus-base route-count L1 drift is:

```text
layers 0--11: 0
layer 12:      0.00184
layer 13:      0.00381
layer 14:      0.00709
layer 15:      0.01955
all-layer mean 0.002018
```

Interpretation: the adapter can change output behavior slightly while leaving
the sparse expert program almost untouched.

## MoE experiment 2: direct router residual

RCR1 directly changed the final four routing decisions:

```text
router_logits' = router_logits + (alpha / rank) * tanh(B(A(h)))
```

Only `A/B` were trainable. Base routers and every expert remained frozen. The
rank-8 treatment had 67,584 parameters; its matched rank-1 shared-attention
control had 65,536 parameters. Data, target tokens, updates, prompts,
evaluator, and decoding were fixed.

| Arm | Correct / 1289 | Accuracy |
|---|---:|---:|
| RCR1 router residual | `194` | `15.0504%` |
| matched attention control | `191` | `14.8177%` |
| unchanged second pass | `191` | `14.8177%` |
| prior MTR1 rank-8 attention | `204` | `15.8262%` |

The direct router treatment adds three answers, fails every magnitude gate,
and underperforms MTR1. Static late-layer router steering is therefore not the
missing factor.

## What is currently going wrong

The behavioral symptom is simple: both small-MoE interventions complete their
frozen training runs, but their learned revision policies repair too few
unseen source-disjoint trajectories. Shared-attention training leaves routes
nearly identical, while direct router-logit training still does not create a
useful correction operator.

The unresolved causal alternatives are:

### 1. Expert-capacity failure

The needed correction computation may not exist in any frozen expert. If so,
choosing different experts cannot help. Dense LoRA modifies the same MLP path
used for every token, while both MoE tests left all expert MLPs bit-identical.
This is the leading explanation if corrected and persistent-wrong cases have
similar route patterns.

### 2. Credit-assignment failure

The only strong loss is over the final corrected trajectory. Thousands of
discrete top-k choices lie between that loss and early routing decisions.
Router gradients may be weak, noisy, or locally satisfied without learning a
stable revision program.

### 3. Missing temporal state

RCR1 is token-local: `delta_router` is computed from the current hidden state.
It does not carry an explicit diagnosis of the draft across output tokens or
layers. But revision is inherently a sequence-level operation: detect the
earlier trajectory's failure, preserve useful work, construct a correction,
and maintain that decision through serialization.

### 4. Intervention depth failure

Both experiments modify only layers 12--15. Representations and expert routes
in layers 0--11 remain identical to the base model. A late residual may be
unable to recover from an incorrect early decomposition.

### 5. Top-k threshold failure

Small logit changes can leave the selected expert set unchanged; crossing a
threshold can abruptly exchange an expert. This produces an unfavorable
optimization geometry: either no functional route change or a discontinuous
change without a trained receiving expert.

### 6. Active-capacity boundary

OLMoE exposes 7B stored parameters but activates only about 1B per token. The
dense 0.8B host shows aggregate revision is possible near this scale, so
capacity alone is not a sufficient explanation, but sparse fragmentation may
make the usable revision capacity smaller than the parameter headline.

### 7. Capability-preservation failure

Even successful dense revision can break isolated domains. MoE routing may
amplify that problem by moving tokens away from experts that already encode a
working procedure. A good controller must learn both repair and preservation,
not simply maximize route change.

## Attribution protocol used

Before another fit, completed MTR1/RCR1 traces should partition each identity
into four outcome transitions relative to the unchanged second pass:

```text
incorrect -> correct   corrected
correct   -> incorrect broken
incorrect -> incorrect persistent-wrong
correct   -> correct   preserved-correct
```

For every group and layer, measure:

- top-k expert-set overlap and ordered-route agreement;
- route-count L1/Jensen-Shannon divergence;
- router margin at the eighth/ninth expert boundary;
- normalized entropy and expert load;
- changed-route tokens concentrated in draft, reasoning, or final-answer
  spans;
- whether corrected examples require route changes at all; and
- whether broken cases correlate with loss of a previously used expert.

Discriminating interpretation:

- corrected cases with unchanged routes imply expert-side/shared-state
  adaptation matters more than rerouting;
- corrected cases with coherent route changes support a better controller;
- route changes in both corrected and broken cases without separation imply
  unsupervised steering noise;
- persistent-wrong cases with stable high-margin routes suggest a frozen
  expert-capacity or early-representation bottleneck.

## Successor hypothesis evaluated

The proposed persistent-controller direction was tested in bounded pieces.
The equations below preserve the pre-result hypothesis; they are not an active
launch recommendation after the negative ladder above.

### Candidate state

Construct a compact source/draft discrepancy representation from the shared
backbone, then maintain a recurrent state across revision tokens:

```text
z = CrossEncode(source_hidden, draft_hidden)
s_t = GRU_or_tied_transformer(s_(t-1), z, h_t, layer_summary)
```

This state must be model-owned and computed only from source, draft, and
current generation history. No correctness target or verifier is available at
inference.

### Router control

For selected layers, produce a bounded, normalized route delta:

```text
delta_r[l,t] = scale_l * normalize(U_l(s_t) + V_l(h_l,t))
r'[l,t] = r[l,t] + delta_r[l,t]
```

Possible stabilizers include a continuation schedule from soft expert mixing
to hard top-k, an explicit changed-route budget, and a preservation loss on
already-correct drafts. These are hypotheses, not requirements; each must be
matched by a simpler control.

### Expert-side revision capacity

Permit small expert-side low-rank deltas without updating base experts. To
avoid an independent adapter for every expert, share a low-rank basis and let
the recurrent state generate coefficients:

```text
DeltaExpert_l,e(h) = sum_k coeff_l,e,k(s_t) * B_l,k(A_l,k(h))
```

Only selected active experts evaluate the delta, preserving sparse compute.
An even simpler candidate is one shared revision adapter after the sparse MLP,
but that must be treated as a dense control rather than evidence for
expert-specific computation.

### Why this is different from the failed arms

- MTR1 adds shared dense capacity but does not own routing.
- RCR1 owns late token-local routing but adds no correction capacity to the
  chosen experts.
- The successor tests their interaction plus persistent draft diagnosis.

The full joint design is justified only if controls show that each component
contributes. Otherwise the simplest winning component should be retained.

## Required matched experiment

Use the exact OLMoE drafts, split, prompts, evaluator, target-token budget,
256-update budget, and greedy decoding already frozen. Do not open holdout.

Minimum arms:

1. recurrent router + expert-side adapter treatment;
2. recurrent router only;
3. expert-side adapter only;
4. equal-active-parameter shared-attention adapter;
5. draft-masked or within-stratum shuffled-draft controller;
6. unchanged second pass.

Report for every arm:

- overall and per-domain exact accuracy;
- corrected/broken/preserved/persistent transition counts;
- malformed/truncated outputs;
- trainable, total, and active parameters;
- generated tokens, measured latency, peak memory, and estimated FLOPs;
- per-layer expert-set overlap, entropy, load, and route divergence; and
- accuracy per generated token and per estimated FLOP.

The old pass rule should not be weakened after seeing results: at least +5
absolute points over unchanged, at least +3 points over the strongest matched
standard control, nonnegative math/logic-code deltas, complete receipts, and
no protected-weight mutation. A miss closes the exact mechanism. A pass opens
one sealed holdout and only then a larger MoE scale test.

## Questions for the frontier reviewer

1. Given the MTR1 and RCR1 evidence, which causal bottleneck is most likely:
   expert capacity, temporal credit assignment, intervention depth, top-k
   geometry, or active-model capacity? State what evidence distinguishes it.
2. What is the smallest architecture that can test that bottleneck without
   conflating routing, parameter count, and extra inference compute?
3. Should a controller act on router logits, expert outputs, a shared expert,
   attention, or some combination? Specify equations and tensor shapes for
   OLMoE's 16-layer/64-expert/8-active topology.
4. How should the model maintain a draft-level diagnosis across tokens while
   remaining autoregressive and model-owned?
5. How can gradients reach routing decisions without relying on an inference-
   time verifier or benchmark labels?
6. Which matched control would most likely falsify the proposed mechanism?
7. What result would justify transfer to a 35B-total/3B-active MoE, and what
   result should terminate the direction?

Please prefer one implementable, falsifiable proposal over a catalog of
loosely connected ideas. Do not claim novelty from analogy alone. Separate
architectural prior art from the exact proposed combination, and separate
mechanics success from held-out capability evidence.

## Non-negotiable evidence boundary

- No benchmark answers, verifier outputs, correctness bits, or task labels at
  inference.
- No host-side solver, answer repair, expert override, or task-specific route.
- No holdout access before the development conjunction passes.
- No fabricated novelty claim.
- No larger-host campaign used to hide a failed small-host mechanism.
- All comparisons bind exact model revision, data identity, prompt, decoding,
  trainable tensors, update count, and evaluator.

## Source documents

This brief is self-contained. Exact contracts and receipts are retained in:

- `SHOHIN.md`;
- `docs/research/SHOHIN_TRANSFERABLE_TEMPORAL_REVISION_CONTRACT.md`;
- `docs/research/SHOHIN_MTR1_SMALL_MOE_TRANSFER.md`;
- `docs/research/SHOHIN_RCR1_REVISION_CONDITIONED_ROUTING.md`;
- `docs/research/SHOHIN_ECR1_EXPERT_CONDITIONED_REVISION.md`;
- `docs/research/SHOHIN_SER1_SELECTED_EXPERT_REVISION.md`;
- `docs/research/SHOHIN_RME1_REVISION_MICRO_EXPERTS.md`;
- `docs/research/SHOHIN_CTSR1_CAUSAL_TEMPORAL_STATE_ROUTING.md`;
- `docs/research/DIVERGE_IDR1_INTERNAL_DRAFT_REVISION.md`;
- `docs/research/DIVERGE_AQC1_ANTISYMMETRIC_QUOTIENT_COMMIT.md`; and
- `SHOHIN_NATIVE_REASONING_MASTER.md`.
