# DIVERGE-VCR1: Verified Temporal Correction Reactor

Status: frozen bounded gate, no result yet.

## Capability hypothesis

The failed LTM1/VMT1 objectives assigned semantic roles to exchangeable
parallel trajectories. Their exact objectives admit a collapsed barycenter.
VCR1 removes that symmetry by assigning roles through time:

1. the protected generator emits one autonomous first-pass draft;
2. a model-owned reactor reads the original problem and that draft through
   distinct segment channels;
3. a learned validity state controls a persistent correction prefix; and
4. the same protected generator emits one final response conditioned on the
   prefix.

The hypothesis is that explicit temporal roles let a compact module learn
`wrong -> right` repair without erasing `right -> right` behavior. VCR1 is not
credited for merely fitting teacher-supplied drafts. It must improve autonomous
drafts from the exact protected generator.

## Frozen source

- Backbone: `HuggingFaceTB/SmolLM3-3B` revision
  `a07cc9a04f16550a088caea529712d1d335b0ac1`.
- Protected generator checkpoint:
  `baseline_late2_v12u1000_rollout_replay_u400_r1/checkpoint_0000400.pt`.
- Checkpoint SHA-256:
  `34c82454e0c53609bc1ac6a9f127437080e431f147e28ae63b4080c413d9a82e`.
- Verified pair source:
  `rest_rollout_preferences_freshv3_k4_p2_r1.jsonl`.
- Pair-source SHA-256:
  `38d582fa6f5626f34d7f390e243bc4f17c4114ce06d83fe0feca755bd7b88ba6`.

The pair source contains 7,452 same-generator math/science pairs. Every pair
has one independently verifier-correct response and one wrong response. A
deterministic identity split prevents any problem from crossing train and
teacher-draft development sets. The builder keeps each complete autonomous
draft and derives one concise terminal target from the verifier-correct
response using the same task-specific answer extractor used for scoring. It
admits a pair only when both the wrong-draft correction example and the
correct-draft no-op example fit in the 4,096-token contract without truncating
the draft or terminal target. The first full-response admission attempt is a
mechanical pre-training negative: it left only 211 train-math identities and
produced no board or model result.

The correction instruction matches this target: the model verifies or repairs
the draft internally and emits exactly one terminal boxed-answer line. The
complete draft remains available to the correction reactor; no source trace is
truncated or replaced by a gold trace.

## Architecture

The protected product generator is loaded exactly and then frozen. VCR1 adds
only a correction workspace:

- eight 384-wide persistent slots;
- four applications of one tied recurrent correction block;
- separate question and draft cross-attention channels;
- a signed discrepancy fusion over question context, draft context, and their
  difference;
- slot self-attention and a gated recurrent update;
- a terminal draft-validity head; and
- an eight-token soft prefix projected into the frozen 2,048-wide language
  state.

Let `q` and `d` denote the original-problem and draft token sets. At recurrent
step `t`, the shared block computes

```text
Q_t = Attn_q(S_t, q)
D_t = Attn_d(S_t, d)
C_t = MLP([Q_t, D_t, Q_t - D_t])
S_{t+1} = GatedUpdate(S_t, SelfAttn(S_t) + C_t)
```

The terminal validity probability `p_valid` scales the emitted correction
prefix by `0.05 + 0.95 * (1 - p_valid)`. Correct drafts therefore retain a
small learnable no-op channel while wrong drafts can activate the full patch.
No verifier, gold answer, alternate response, or candidate bank exists at
inference.

## Matched controls

Three systems consume the same autonomous first drafts:

1. **one-pass source**: the protected generator's original draft;
2. **plain two-pass**: the protected generator receives the same correction
   prompt but no learned prefix; and
3. **role-blind control**: the exact VCR1 parameters, recurrent steps, losses,
   and FLOPs, but both cross-attention channels receive the union of problem
   and draft tokens.

The treatment differs from the role-blind control only in preserving the
question/draft fault line. This is a stronger control than a smaller MLP or a
zero-parameter prompt baseline.

## Training contract

- seed `2026080603`; data seed `2026080603`;
- one wrong-draft and one correct-draft presentation per selected pair;
- identical verifier-accepted terminal answer as the target in both
  presentations;
- 4,096 total positions including eight correction slots;
- 200 updates, eight pairs per update through gradient accumulation;
- AdamW, fused, LR `3e-4`, cosine decay, gradient clip 1.0;
- language loss plus `0.20 * BCE(validity)` and
  `0.10 * max(0, 0.25 - correction_wrong + correction_correct)`;
- protected generator parameters remain bit-identical.

Mechanical smoke uses two updates and cannot qualify capability. Treatment
and role-blind control start from byte-identical correction-module weights.

## Autonomous gate

The fixed development gate uses one deterministic first-pass draft per prompt
from the protected generator on 100 MATH-500 and 100 held-out OpenScience
short-answer problems. The same draft bank is consumed by plain two-pass,
role-blind, and VCR1 correction. Report for each arm:

- correction generation is deterministic greedy decoding, capped at 256 new
  tokens inside one common 4,096-position admission contract;

- total and per-domain exact accuracy;
- `wrong -> right`, `right -> right`, `right -> wrong`, and `wrong -> wrong`;
- net correction (`wrong -> right - right -> wrong`);
- generated-token use and exhaustion; and
- validity calibration on the independently scored first drafts.

VCR1 advances only if all are true:

1. finite execution and byte-identical protected source tensors;
2. at least five net autonomous corrections over one-pass source;
3. at least three more correct answers than role-blind over 200 prompts;
4. positive net correction in both math and science;
5. no domain loses more than two answers versus source;
6. `right -> wrong` is at most half of `wrong -> right`; and
7. swapping question/draft masks or resetting the prefix removes at least two
   treatment solves.

If plain or role-blind two-pass equals or beats VCR1, temporal correction may
still be a useful product technique, but this architecture has not earned
promotion. Failure closes VCR1 without a nearby seed, width, depth, loss,
duration, or threshold repair.
