# Shohin Transferable Temporal Revision Contract

Status: current evidence contract, updated 2026-08-09. TTR1 established strong
aggregate dense cross-family transfer but failed executable-code retention.
MTR1 shared-attention and RCR1 static-router transfer both failed on small
OLMoE development. The current critical path is MoE route/error attribution
followed by one genuinely MoE-native successor; scratch pretraining and a
larger-MoE campaign remain unauthorized.

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
| SmolLM3-3B | `469/358` (+8.61 points) | sealed | aggregate cross-family transfer; code `4/9` fails conjunctive retention |

These results establish a real learned revision effect across Qwen and
SmolLM families at aggregate level. They do not establish reliable
all-domain family transfer, protected-product promotion, or a MoE mechanism.

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

## Frozen Result

TTR1 development completed all matched arms on all `1,289` identities:

| Arm | Correct | Accuracy |
|---|---:|---:|
| trained draft revision | `469` | `36.3848%` |
| unchanged second pass | `358` | `27.7735%` |
| self-refinement | `398` | `30.8766%` |
| long single generation | `420` | `32.5834%` |
| best-of-two | `339` | `26.2995%` |
| independent commitment | `371` | `28.7820%` |

Treatment exceeds unchanged by `+111` answers / `+8.6113` points and the
strongest matched control, long single generation, by `+49` answers /
`+3.8014` points. Domain correct-count deltas against unchanged are MATH
`+62`, logic/science `+54`, and executable MBPP `-5` (`9 -> 4`). The first,
third, and coverage conditions pass; the all-domain condition fails. Holdout
was not opened.

Comparison SHA-256 is
`a20cdd7567aa2502a62ad5591748bc9241371b3f7fe4620c29a1de71e90c9cd7`.
Treatment and independent-report SHA-256 values are
`98d4a4b4f7b37f9369fe7c81ec3a43f60531c804a4d8b055c27a3b51772c5475`
and `a6974226297da951eca5abf98bd513cd5b04e451b2ff5158dfa8b522847f4dc7`.
Complete execution charged `27.793` H100-hours, including canceled unsharded
work and node failures. Exact TTR1 is closed without a nearby retry.

## Advancement

SmolLM3 established aggregate cross-family transfer but exposed a capability
preservation failure. OLMo2-7B then supplied the larger dense non-Qwen test:
direct revision reached only `259/1,289`, while selective commitment and
error-syndrome revision both failed their frozen gates. More local OLMo
variants are not authorized.

MTR1 on small open `OLMoE-1B-7B-0125-Instruct` is complete and closed.
Final-four-layer rank-8 shared-attention revision reaches `204/1,289`, only
`+13` answers / `+1.0085` points over unchanged, while mean all-layer
route-count L1 drift is only `0.002018`. RCR1 then directly trains a bounded
low-rank residual on the final four router logits and reaches `194/1,289`,
only three answers over matched rank-1 attention and unchanged and below
MTR1's 204. Both holdouts remain sealed.

The next operation is read-only attribution of corrected, broken,
persistent-wrong, and preserved-correct cases against per-layer route changes.
Only then may one draft-conditioned multi-token MoE controller be frozen. The
leading candidate jointly controls bounded routing deltas and small
revision-specific expert-side low-rank capacity, with router-only,
expert-only, equal-budget attention, and draft-shuffled/masked controls.
`docs/research/SHOHIN_MOE_FRONTIER_CONSULTATION_BRIEF_20260809.md` contains the
self-contained problem statement.

Larger-MoE capability work, including the mechanics-qualified
Qwen3.6-35B-A3B path, remains prohibited until a small-MoE development and
sealed-holdout pass.

Scratch Shohin training is optional later efficiency evidence. It is not the
critical path and is not authorized by TTR1.
