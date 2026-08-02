# R12 Frozen-Backbone Capability-Floor Campaign

Status: preregistered, not launchable. Date: 2026-08-02.

## Purpose

The current v15 -> v19 -> v20 -> v21 campaign is the final bounded test of
the local operation-family compiler. It will run each selected mechanism once,
with exact warm-start preservation into joint release. There will be no v22,
width escalation, longer repetition, seed fishing, or loss-weight search if
v20/v21 fails the unchanged strict evaluator.

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

The mechanism hash is intentionally unset until the v15-v21 route ends. If
v21 passes exact local state plus both strict factors, its exact artifact is
the candidate. If the joint gate fails, separate compiler/reactor/reader fitting
is retired and replaced by one differentiable model-owned trajectory:

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

Optimizer geometry is fixed at semantic batch 16, fused AdamW, learning rate
`3e-4`, betas `(0.9, 0.95)`, weight decay `0.01`, and gradient clip `1.0`.
Seed pairs are `(31,11)` and `(32,12)`, and promotion uses the minimum score
across seeds rather than their mean. The admitted release contributes 40,000
train episodes, 5,000 development episodes, and a 180,000-row stream with
SHA-256 `8f205de...20f87`. Complete causal rectangles remain atomic.

Qwen3.5-0.8B and SmolLM3-3B official configuration files are pinned and
validated structurally. MobileLLM-R1 is manually gated; the current credential
has not accepted its license, so its exact config and weights cannot yet be
receipted. This is a launch blocker, not permission to guess its geometry or
drop the 360M arm. The generated interface receipt remains non-launchable
until that access, all four tokenizer intersections, the final mechanism hash,
and dense-control parameter/FLOP receipts exist.

Machine-readable interface contract:
`artifacts/r12/ettr_capability_floor_interface_v1.json`.
