# Shohin Transferable Temporal Revision Contract

Status: TTR1 frozen for implementation, 2026-08-08. This contract supersedes
scratch pretraining as the primary project path. It does not authorize a
390M/920M scratch run.

## Mission

The target is a transferable temporal reasoning architecture that improves an
existing pretrained model using only model-owned computation. One state of a
backbone writes a complete draft; a separately trained state of the same
backbone reads the source and that draft, then commits one coherent final
trajectory. The final system contains the backbone, both role states, and a
deterministic two-phase controller in one deployable package.

At claim time there is no external proposal model, verifier, answer router,
tool, host solver, correctness bit, or benchmark-specific route. Training may
use independently verified targets. Evaluation never exposes those targets.

## Existing Boundary

| Host | Development treatment/control | Holdout treatment/control | Boundary |
|---|---:|---:|---|
| Qwen3.5-0.8B | `323/236` (+6.749 points) | `328/242` (+6.724 points) | aggregate transfer; holdout code `8/9` fails conjunctive retention |
| Qwen3.5-4B | `529/371` (+12.26 points) | `554/380` (+13.61 points) | every source-disjoint domain positive; protected product mixed fail |
| Qwen3.5-9B | `589/464` (+9.70 points) | `625/495` (+10.16 points) | every attribution domain positive; original promotion floor missed |

These results establish a real learned revision effect within one Qwen family.
They do not establish family transfer, superiority to every standard
test-time-compute baseline, or a MoE mechanism.

## TTR1 Host And Inputs

The shortest cross-family test uses the already-pinned
`HuggingFaceTB/SmolLM3-3B` causal model:

- revision: `a07cc9a04f16550a088caea529712d1d335b0ac1`;
- local config SHA-256:
  `c72b1031274ff4626e434d0019e88e95a767460135db9ee492eb80652b786af1`;
- draft warm start: existing SmolLM3 B1 final-four-layer rank-8 checkpoint;
- checkpoint SHA-256:
  `c4af49e88d31ef751d7ab8697f12481b645801963e1ecb71174a7da645df6a35`;
- exact source bank: 4,096 MATH, 4,096 logic/science, and 200
  execution-verified MBPP identities;
- bank SHA-256 values: `e0ede832...dbe5`, `5a96859f...017`, and
  `0b6d068b...398`;
- split and evaluator: the existing source-identity-disjoint
  `9,655/1,289/1,279` revision contract and unchanged exact-answer/execution
  evaluators.

No Qwen draft or adapter may enter TTR1. Every draft and final response is
produced by a SmolLM3 role state.

## Changed Factor

Let `G(theta, x, b)` be one bounded generation from pretrained backbone state
`theta`, input `x`, and decoding budget `b`.

1. Draft: `d = G(theta + delta_draft, source, 768)`.
2. Revision: `y = G(theta + delta_revision, source || d, 768)`.
3. `delta_revision` is trained on complete verified final trajectories while
   `theta` and `delta_draft` remain frozen.
4. Commitment is whole-trajectory: the system emits exactly one revision and
   never averages incompatible answer fields.

The frozen treatment geometry is the prior IDR schedule: final four decoder
layers, rank 8, alpha 16, 256 AdamW updates, batch 1, accumulation 8, context
4,096, learning rate `2e-5`, and the existing data/order seeds. The legacy B1
metadata field `unfreeze_layers=null` must be normalized to numeric zero by
the already-tested bitwise-preserving migration before use.

## Matched Controls

All arms use the same SmolLM3 revision, source identities, prompts where
applicable, evaluator, and recorded generated-token and FLOP accounting.

1. **Unchanged second pass:** B1 reads the exact treatment draft through the
   exact treatment revision prompt and receives the same 768-token final
   budget. This is the primary causal control.
2. **Standard self-refinement:** unchanged B1 receives the same draft under a
   generic review-and-correct instruction and the same final budget.
3. **Long single generation:** B1 receives source only and at most 1,536 new
   tokens, matching the maximum two-pass generated-token budget.
4. **Best-of-two/majority:** two source-only B1 attempts share the same total
   generated-token ceiling; exact-answer agreement commits, and deterministic
   first-attempt tie-breaking is reported.
5. **Independent commitment:** an equal-geometry, equal-update LoRA is trained
   on the same final targets without informative draft content. Input shapes
   and attention FLOPs are matched by masking the draft span rather than
   deleting it. This isolates learned finalization from trajectory-conditioned
   revision.

Prompt tokens, generated tokens, wall time, peak memory, trainable parameters,
and estimated attention/MLP FLOPs are reported per arm. Any unmatched control
is labeled diagnostic and cannot satisfy the matched superiority claim.

## Release Gates

First run one mechanics-only smoke on 24 balanced examples. It must prove
generic SmolLM3 loading, exact checkpoint restoration, draft serialization,
masked-span control behavior, and identical treatment/control prompt tokens.
It is not a capability result.

Development opens only once. TTR1 advances to holdout only if:

1. trained revision exceeds unchanged second pass by at least five absolute
   points overall;
2. MATH, logic/science, and executable-code correct-count deltas are each
   nonnegative;
3. trained revision exceeds the strongest fully matched standard control by
   at least three absolute points overall;
4. all 1,289 identities and all compute receipts are complete.

Holdout applies the same conjunctive conditions independently to all 1,279
identities. Only a development-plus-holdout pass opens one protected product
evaluation. Product promotion requires at least `+0.05` five-domain macro
accuracy over the strongest matched control and nonnegative correct-count
deltas in every main domain. A miss closes exact TTR1 without seed, rank,
layer, duration, prompt, decoding, or threshold rescue variants.

## Advancement

A SmolLM3 pass establishes the first cross-family dense transfer. The next
test is one larger dense non-Qwen family with the same contract, followed by
one small/medium open MoE. The MoE starts with frozen router and experts and
role-specific shared-layer adapters. It must additionally report router and
expert utilization, active versus total parameters, latency, memory, total
FLOPs, and accuracy per compute. Large-MoE work is prohibited until both the
cross-family dense and smaller-MoE gates pass.

Scratch Shohin training is optional later efficiency evidence. It is not the
critical path and is not authorized by TTR1.
