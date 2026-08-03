# R12 Frozen-Backbone Capability-Floor Campaign

Status: historical capability-floor evidence retained; product-reasoning pivot
active after the one running byte-rail result. Date: 2026-08-02.

> **Operational supersession:** this document preserves the causal/mechanistic
> campaign and its negative evidence. It is no longer the primary execution
> queue. The active queue is
> `docs/research/SHOHIN_72H_PRODUCT_REASONING_EXECUTION.md`, which prioritizes
> Qwen3.5/SmolLM3 end-to-end answer accuracy over maximal source-deletion or
> architecture-native claims. Existing jobs may finish; this local interface
> family receives no new variants.

## Purpose

The v15 -> v19 -> v20 route is complete. V19 and v20 each ran once and both
failed at 25% exact family classification with 100% NOOP predictions. The
planner stopped and v21 was not launched. There will be no v22, width
escalation, longer repetition, seed fishing, or loss-weight search.

After that result, the core question is no longer whether another local head
can fit the Shohin residual. It is whether the ETTR interface is learnable at
all, and if so at what frozen-backbone capability floor.

## Four Backbones

The matrix uses the following frozen parents, each pinned to an immutable
revision before artifact download:

| Candidate | Frozen source | Role |
|---|---|---|
| protected Shohin 125M | step-300k SHA-256 `211d6b2...f66a6` | undertrained negative baseline |
| MobileLLM-R1 360M | [`facebook/MobileLLM-R1-360M@ac72186...ed1`](https://huggingface.co/facebook/MobileLLM-R1-360M) | strong sub-billion reasoner; FAIR noncommercial research license |
| Qwen3.5 0.8B | [`Qwen/Qwen3.5-0.8B@2fc0636...8b17`](https://huggingface.co/Qwen/Qwen3.5-0.8B) | larger hybrid-architecture threshold probe; text path only |
| SmolLM3 3B | [`HuggingFaceTB/SmolLM3-3B@a07cc9a...ac1`](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | high-capacity small-model ceiling |

The stronger parents are post-trained while Shohin is raw-pretrained. This is
intentional for locating a capability floor, but it prevents attributing a
positive result to parameter count alone. Tokenizer, pretraining, and
post-training differences remain explicit covariates.

## Shared Interface

Every arm receives the same semantic ETTR episodes, causal rectangles,
train/development split, update budget, charged positions, source-deleted
evaluator, and architecture-level mechanism. Tokenizer-specific rows must be
semantic-byte-equivalent and independently receipted. A learned projection may
map each frozen hidden width into the common ETTR width; its parameters and
FLOPs count toward both treatment and matched dense control.

Separate compiler/reactor/reader fitting is retired and replaced by one
differentiable model-owned trajectory:

`WORLD -> typed state -> COMMAND recurrence -> terminal state -> late QUERY`

That successor now exists in `train/capability_floor_trajectory.py`. WORLD and
COMMAND use the same `TiedTrajectoryCell` and `UnifiedStateEncoder`; neither
transition accepts QUERY. Per-example STOP is monotone, freezes every later
state update, and is forced at the phase limit. A fixed typed algebra applies
ALLOCATE/WRITE/CLEAR/LINK/UNLINK/SET_ROOT actions. QUERY enters only after
COMMAND termination through `LateQueryReader`. The default mechanism has
31,329,056 trainable architecture parameters. Source SHA-256 is
`b0fef198fe35ade9fcf04f86d70119d6fa9b04feb4ff2d680252523b45040c7f`;
architecture SHA-256 is
`552236f44b4b30d9f384fc3ffe185663c6231eac96e5b4fbf4e996b26a0c53cf`.
This freezes a testable mechanism; it does not establish capability.

## Component Gates

Each backbone receives identical two-seed 2,000-update component budgets.
Composition is forbidden until all three interfaces pass held-out data:

1. Oracle-state query reader: strict WORLD >=95% and strict COMMAND >=95%.
2. Oracle-program executor: exact terminal execution >=95%.
3. WORLD compiler/effect binding: exact public operation/effect binding >=95%.

If every backbone fails the same oracle component, the interface is defective.
That component is redesigned before any larger run; optimization scale is not
an admissible explanation.

## Composition Gate

Only component-qualified backbones receive a two-seed 5,000-update autonomous
composition. Promotion requires all of:

- autonomous exact terminal packets >=90%;
- source-deleted strict WORLD >=90%;
- source-deleted strict COMMAND >=90%;
- binding-deranged, state-reset, query-only, and shuffled-label controls no
  more than two points above their empirical chance rates;
- reproducibility on frozen development before sealed confirmation is opened.

## Favorable Dense Control

Every composed ETTR arm is paired with an ordinary favorable recurrent model
using the same frozen backbone, examples, charged positions, optimizer updates,
and evaluator. Trainable parameters must match within 1%; measured training
FLOPs must match within 5%. The dense control is allowed untied capacity and
full recurrent state; it is not deliberately starved. If it equals or beats
ETTR, ETTR has not earned inclusion.

The implemented control in `train/capability_floor_dense_control.py` has
independent WORLD and COMMAND cells, a full dense recurrent state, learned
terminal-packet heads, and the same late QUERY boundary. Its exact matcher
selects hidden width 424 plus a live 880-wide capacity MLP and 243 live tail
parameters, giving exactly 31,329,056 trainable parameters and zero relative
parameter mismatch. Training FLOPs still require a measured H100 receipt.

## Decisions

- 0.8B or 3B passes while smaller models fail: record the smallest passing
  capability floor and target that scale.
- Even 3B fails autonomous composition: retire the current ETTR mechanism.
- Dense matched control equals or beats ETTR: reject the ETTR claim.
- ETTR beats dense and clears every negative control: proceed to matched-total-
  parameter 360M scratch candidates, one dense and one ETTR-integrated.

No trillion-token training is authorized by this campaign. A later long run
requires a mostly fresh broad corpus, exact decontamination, and a staged
general instruction -> verified reasoning -> RLVR post-training plan.

Machine-readable preregistration:
`artifacts/r12/ettr_capability_floor_preregistration_v1.json`.

## Frozen Interface Contract

The first implementation preflight is
`train/capability_floor_interface.py`. It freezes a raw canonical ASCII input
envelope with no candidate-specific chat template, native required BOS only,
no truncation, token-offset receipts, and a four-tokenizer intersection cohort.
Every frozen parent exposes final post-norm hidden states for all source tokens.
A candidate-specific bias-free projection maps those states to the common
512-wide ETTR interface, followed by RMSNorm; projection parameters count
against the treatment and its dense control.

Optimizer geometry uses semantic microbatches of 16 and exactly four
microbatches per optimizer update. The four are selected through a frozen,
label-stratified replay schedule and losses are normalized globally over the
64-row window. This is paired with fused AdamW, learning rate `3e-4`, betas
`(0.9, 0.95)`, weight decay `0.01`, and gradient clip `1.0`.
Seed pairs are `(31,11)` and `(32,12)`, and promotion uses the minimum score
across seeds rather than their mean. The admitted release contributes 40,000
train episodes, 5,000 development episodes, and a 180,000-row stream with
SHA-256 `8f205de...20f87`. Complete causal rectangles remain atomic.

Qwen3.5-0.8B and SmolLM3-3B official configuration files are pinned and
validated structurally. MobileLLM-R1 is manually gated; the current credential
has not accepted its license, so its exact config and weights cannot yet be
receipted. This is a launch blocker, not permission to guess its geometry or
drop the 360M arm. The generated interface receipt remains non-launchable
until that access, all four tokenizer intersections, a real-corpus mechanics
smoke, a symbolic-to-neural interface-equivalence receipt, a component-specific
stratified replay receipt, and a measured dense-control FLOP receipt exist.
The mechanism hash and exact dense parameter receipt are no longer blockers.

Machine-readable interface contract:
`artifacts/r12/ettr_capability_floor_interface_v1.json`.

## Why the sampling contract changed

The completed v19 and v20 traces reveal a launch-level optimizer defect that
the aggregate corpus audit could not see. In both runs all 100 logged updates
omit at least one of NONE/WRITE/LINK; 33 logged updates contain only one
family; LINK appears in only 33% of logged updates. The aggregate sampled
target distribution is 41.50% NONE, 44.98% WRITE, and 13.52% LINK, but the
final logged update is 75% NONE, 25% WRITE, and no LINK. Both final checkpoints
predict NOOP for every held-out operation. The LINK-only logged update at
position 499 reaches essentially zero loss, so the model can fit a regime but
does not retain it across the ordered stream.

The v19 and v20 trace audit artifact SHA-256 values are
`8ea22620f799fa3346c02260c36d122a1d62cd98669b391a9b55cd8d27df91a1`
and
`454d2233456fc80248be26098620f0abcb3d2d33c2e1c57bc4509fe8aa292ed4`.
These audits cover logged checkpoints only and do not invent labels for the
unlogged steps. They are nevertheless sufficient to reject the old assumption
that one 16-row semantic core is an i.i.d. optimizer update.

Every capability-floor component therefore needs a deterministic schedule
receipt proving coverage of its causal strata in each four-microbatch update:
NONE/WRITE/LINK for execution, WORLD/COMMAND for reading, WORLD factor plus
effect family for compilation, and both intervention factors for composition.
ETTR and dense controls receive the exact same windows and charged positions.

`train/capability_floor_replay.py` implements that scheduler. It treats each
four-row causal rectangle as indivisible, selects four rectangles per 16-row
semantic microbatch, accumulates exactly four microbatches before an optimizer
step, forbids a rectangle from repeating inside an update, proves all required
strata are present, records charged positions, and emits a hashable schedule
shared byte-for-byte by ETTR and dense control. The first cohort-index schema
was rejected before replay publication because it concatenated train and
development cores without a per-row split. Schema v2 now binds the split,
emits strict JSONL, and the loader rejects v1. The atomic publisher in
`train/materialize_capability_floor_replay.py` freezes candidate-specific
charged positions and equal ETTR/dense schedule hashes for every component and
seed pair. A corrected v2 release audit must complete before the final replay
receipt can be materialized.

## First interface deliverable

The earlier 94.0756% operation-family oracle is a symbolic upper bound, not a
neural-interface result. It parses resolved public COMMAND structure exactly
and combines it with oracle preceding-state factors. V20 consumes learned
language residuals, pooled root/direct-child anchors, and a learned typed-state
encoding. Those are source-legal inputs, but their sufficiency was never shown
to equal the symbolic feature set.

Before any four-backbone GPU fit, the preflight must produce three receipts:

1. An exact symbolic reference using only source-visible COMMAND plus the
   allowed oracle component state.
2. A tensor-sufficiency probe using precisely the tensors exposed to the
   model, with no assessor feature available at inference.
3. Renderer-orbit and binding-derangement controls proving that success comes
   from semantic binding rather than layout or batch identity.

If the tensor probe fails while the symbolic reference passes, the interface
is redesigned before scale is tested. This is the first capability-floor
deliverable and prevents all four backbones from being spent on a representation
that discarded the required variable.

`train/capability_floor_sufficiency.py` implements the exact-tensor side of
this gate. The probe can consume only projected source token features, public
renderer span masks, and exact typed-state tensors admitted to the model. It
includes explicit source/state multiplicative binding and binds dtype, shape,
and exact bytes for every tensor in its receipt. Promotion requires at least
95% symbolic, tensor, renderer-orbit, and orbit-agreement scores while
binding-deranged and state-value-permuted controls remain within two points of
empirical chance. The code is ready; real four-backbone feature extraction and
measurement remain outstanding.

## Implemented Preflight Evidence

- Unified mechanism receipt:
  `artifacts/r12/ettr_unified_trajectory_mechanism_v1.json`, SHA-256
  `73996ac5ede6ce209cf6209e5cbec7629e03b50e6d1e204150b7444eec27301e`.
- Favorable dense descriptor:
  `artifacts/r12/ettr_favorable_dense_control_v1.json`, SHA-256
  `7d39abf73ba6025e1b4d6a8a503e5c220f5c74e873101630e10f2bfe42fb3c57`.
- Updated campaign preregistration SHA-256:
  `e7800d87fd8e9e089f4f650966064492735494c994089f85ac3b1bcb04cdb3d1`.
- Updated interface receipt SHA-256:
  `b200e7565008c030c05ef0eeeb592c086b2158b3c399881c2412b57d228aa752`.
- Thirty-seven focused campaign, trajectory, sufficiency, replay, and dense
  tests pass. These are mechanics and custody tests, not the 95%/90%
  capability gates.

## First Real-Tensor Result

Protected-Shohin H100 gate `729554` completed from the immutable step-300k
checkpoint. The 2,000-update probe reaches 71.22% clean family accuracy,
51.77% complete renderer-orbit accuracy, and 62.75% renderer prediction
agreement on 5,208 balanced development examples. This decisively misses the
95% gate before any negative-control interpretation. The report SHA-256 is
`0ca8387bfba4c4b144fd34112b7aed2580884b9574583445b3210dbb37f45c7f`.

The current final post-norm residual interface is rejected. Before testing a
larger backbone, one bounded interface audit compares exact input embeddings
and intermediate frozen-layer role features. If no frozen tap passes, a
shared learned canonical-byte rail may be added, but it is counted and cannot
consume a symbolic AST or assessor field. State binding must be measured on
source-matched cross-WORLD pairs; full-population state corruption cannot be
required to reach chance because public syntax alone predicts roughly 80% of
the coarse family label.

The frozen audit is preregistered in
`train/capability_floor_layer_taps.py`. A single exact forward extraction
captures the input embedding and blocks 0, 4, 9, 14, 19, 24, and 29 from the
same 128-train/128-development core sample used by the failed gate. Each tap
is pooled over the same four public role spans and receives an independently
initialized copy of the same 10.43M-parameter probe for exactly 2,000 updates.
The final block is an algebraic replication of the failed postnorm interface;
earlier taps are the only experimental variable. Extracted tensors and every
probe result are no-overwrite, checksum-bound artifacts. A frozen tap can
advance only if clean and renderer-orbit accuracy reach 95%; promotion still
requires the corrected source-matched binding control before any larger
backbone fit.

That correction is implemented as a post-fit gate. It pairs rows with
byte-identical COMMAND source, renderer view, command factor, and operation
position across the two WORLD factors, retaining only pairs whose exact family
changes. The frozen source tensor is held fixed while the typed preceding state
is swapped. A binding-capable probe must predict the opposite-WORLD target and
remain renderer-stable; a syntax-only probe instead preserves the original
label. This scorer cannot alter weights or choose candidates.

The no-tap fallback is implemented but remains held until the frozen-depth
decision. `train/capability_floor_byte_rail.py` maps each of the eight bytes in
an allowed public role atom to a distinct `(within-atom position, ASCII byte)`
coordinate. The resulting 1,024-wide sparse role vector is lossless for that
atom, shared across all candidate backbones, and contains no AST class,
operation family, target, or oracle successor. It uses the same probe, exact
state, 128/128 core population, 2,000 updates, and evaluator. A failure here
localizes the defect beyond backbone token compression; a pass permits an
architectural source rail only after source-matched binding also passes.

## Frozen-Depth Decision

Jobs `729667--729675` completed cleanly from immutable source `6fd7c94`.
The shared 1.539 GB tap bundle is checksum-bound; every independent probe ran
the fixed 2,000 updates. Held-out clean/orbit accuracies are:

| Tap | Clean | Complete orbit | Orbit agreement |
|---|---:|---:|---:|
| embedding | 69.89% | 69.89% | 100.00% |
| block 0 | 69.89% | 69.89% | 100.00% |
| block 4 | **72.12%** | 58.60% | 74.35% |
| block 9 | 70.83% | 56.37% | 70.89% |
| block 14 | 70.43% | 58.91% | 76.27% |
| block 19 | 69.26% | 55.45% | 71.97% |
| block 24 | 70.12% | 55.53% | 70.66% |
| block 29 postnorm | 70.79% | 53.23% | 65.28% |

No frozen layer approaches 95%; block 4 is best and remains 22.88 points
short. The depth-tap route is closed. This is evidence against a late-layer
erasure-only diagnosis: the permitted mean public-role interface never exposes
the required variable cleanly at any sampled depth. The prespecified lossless
byte-role rail is therefore released. CPU materialization job `729767` is
running and H100 fit `729768` is held `afterok:729767`.
