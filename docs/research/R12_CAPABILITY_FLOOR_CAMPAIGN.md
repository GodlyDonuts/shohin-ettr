# R12 Frozen-Backbone Capability-Floor Campaign

Status: local family route closed; unified interface preflight in progress and
not launchable. Date: 2026-08-02.

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

The mechanism hash is intentionally unset because the local route ended at a
v20 failure. Separate compiler/reactor/reader fitting is retired and replaced
by one differentiable model-owned trajectory:

`WORLD -> typed state -> COMMAND recurrence -> terminal state -> late QUERY`

That successor uses a tied recurrent state core, adaptive STOP, and late-query
readout. It is frozen as one mechanism before any cross-backbone fit.

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
until that access, all four tokenizer intersections, the unified mechanism
hash, a symbolic-to-neural interface-equivalence receipt, a component-specific
stratified replay receipt, and dense-control parameter/FLOP receipts exist.

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
