# DIVERGE-CRP1: Causal Revision Packet

Status: closed after the one frozen neural gate. The packet causes substantial
causal revision on scalar and register programs, but misses the absolute,
matched-control, localization, joint, and cross-family promotion thresholds.

## Capability hypothesis

VCR1 proved that a small recurrent prefix can turn incomplete SmolLM3 drafts
into concise answers, but it did not revise a complete explicit wrong
derivation. CRP1 tests a materially different mechanism: preserve a bounded
version space over the location of the first invalid step, evaluate every
location with tied recurrent weights, and commit one whole guarded repair
packet before generating a correction and causally replayed suffix.

The packet has one categorical fault-line variable with values `NO_ERROR` and
`STEP_1 ... STEP_N`. Candidate `e` receives separate source masks for the
valid prefix before `e`, the proposed fault at `e`, and every dependent step
after `e`. All candidates share parameters and remain separate through
recurrence. The generator receives only the prefix from one hard-selected
candidate; candidate fields are never averaged.

The exact matched control instantiates the same parameters, candidate
identities, recurrent updates, training rows, targets, and FLOPs, but every
candidate sees the complete trace in all three channels. A gain over this
control is evidence for the causal guard structure rather than candidate
width or recurrent compute.

## Board

The deterministic board contains scalar arithmetic, noncommuting two-register
programs, and symbolic string transformations. Train/development depths are
4--6. Evaluation depths are 7--9 and also hold out the renderer and value/width
band. Every wrong trace is complete and contains exactly one verifier-created
first error. Every later draft step is recomputed from that corrupted state,
so the suffix is locally coherent but globally wrong. The correct answer is
never equal to the wrong draft answer. Every trace also has a correct no-op
twin.

Supervisor program objects and error certificates are stored for independent
assessment but are never rendered to the model beyond the ordinary problem,
complete draft, and final draft answer. The model must emit the first error,
one corrected step, the replayed dependent suffix, and one boxed answer.

## Frozen one-seed experiment

1. Build and hash 4,800 train, 480 development, and 480 OOD evaluation rows.
2. Run two-update guarded and unguarded mechanical smokes from byte-identical
   packet initialization.
3. On mechanical success, train each arm for exactly 200 updates, accumulation
   8, at learning rate `3e-4`. The SmolLM3-3B product generator remains frozen.
4. Evaluate the exact same 480 OOD identities under prompt-only, guarded,
   unguarded, reset, shifted-selection, and packet-swap conditions. Evaluate
   all three trained/plain arms on the 480 correct no-op twins.

## Frozen promotion gate

All conditions are conjunctive:

- guarded wrong-trace exact answers at least 240/480;
- guarded beats prompt-only by at least 48 solves;
- guarded beats equal-compute unguarded by at least 24 solves;
- guarded autonomous packet localization at least 360/480;
- guarded joint localized-and-correct revision at least 192/480;
- guarded correct-twin answer and no-error packet preservation each at least
  432/480, and no more than 12 solves below prompt-only preservation;
- guarded beats unguarded by at least five solves in each family;
- reset, shifted selection, and packet swap each cost at least 24 solves, and
  at least two cost at least 48;
- no arm exhausts the 384-token correction budget.

Failure closes this exact first-error packet without a seed, width, duration,
loss, depth-band, renderer, or threshold repair. A pass qualifies only a
bounded model-owned causal-revision mechanism. It then requires transfer to
natural verified math/code/science traces before any broader DIVERGE claim.

## Result

The exact board passed construction with 4,800 train, 480 development, and 480
evaluation rows, balanced across the three families. There were zero identity
overlaps, malformed rows, or tokenizer truncations; the longest admitted row
used 824 of 4,096 positions. Train, development, evaluation, and board-report
SHA-256 values are:

- `33dacc5e0f9a72ad01eaa58bbee627c534437db95752b7e6a76a8488af0b2ede`;
- `ad27c4ca8f89a158fc1c4516e1cfce67f792bd5015f0b68e9c30ba39b62cfde5`;
- `db0bde0c22afe3d25f4f1f578249bf67156f4115d38a094c07f1ea36f6be6849`;
- `b9ead3db62c91622231cb310fd5cc6f48d814e69cf26b40ad195c8306a689d8d`.

Both arms completed 200 updates from packet SHA-256
`dc5f273d3fb629fccadbd9ac2209a6b897f2be48d00c9aef71b23116b6a5966e`.
Each trained 3,556,353 parameters on 191,201 charged response tokens, retained
an identical frozen source, and used 12.45GB peak allocated CUDA memory.
Guarded and unguarded training reports / checkpoints are:

- guarded:
  `ff0419e26295125a5451603de078a8b25b7c2ebaa8ad902585b6d5f958000bc7` /
  `588dce4f608fde47516a8b29feedc40bf7ee58d2ff2aa8b344848915dcacb5ce`;
- unguarded:
  `0a0353c450c245acd6f44aaa34974e5ff1caa718ea54c371056aad80e84f0f85` /
  `93e7e71db74e1f4efe68d13b028157cdaea449bb34e2f8e8bdc528af40ce4ced`.

The autonomous OOD result is:

| Arm | Wrong answers | Packet localization | Joint | Correct twins |
|---|---:|---:|---:|---:|
| prompt-only | 1/480 | n/a | n/a | 349/480 |
| unguarded | 194/480 | 64/480 | 60/480 | 476/480 |
| guarded | **213/480** | **313/480** | **183/480** | 475/480 |
| guarded reset | 0/480 | 313/480 | 0/480 | n/a |
| guarded shifted candidate | 36/480 | 23/480 | 28/480 | n/a |
| guarded packet swap | 100/480 | 313/480 | 39/480 | n/a |

Guarded exact answers by family are 135/160 scalar, 78/160 register, and
0/160 symbolic, versus 115/160, 79/160, and 0/160 for unguarded. The net
answer advantage is therefore only +19, concentrated entirely in scalar
arithmetic. The causal-guard effect is much larger on coherent attribution:
guarded creates 129 joint wins not achieved by unguarded and loses only six.
Reset, location shift, and cross-example packet swap cost 213, 177, and 113
answers, respectively.

Transcript inspection confirms a real but bounded capability. Of 183 guarded
joint successes, 181 exactly match the canonical complete correction target
after whitespace normalization and 182 include the complete correct dependent
suffix verbatim. Examples explicitly replace a wrong arithmetic/register step
and replay every later state to the correct terminal value. This is not VCR1's
truncation/finalization effect. It nevertheless does not generalize to the
held-out symbolic family and does not create a large enough answer advantage
over equal-compute generic recurrence.

The formal gate fails. Guarded misses 240 wrong answers, 360 packet
localizations, 192 joint repairs, the +24 unguarded margin, and the positive
advantage in every family. Some destructive controls and the prompt baseline
also exhaust generation, failing the all-arms exhaustion check. Correct-trace
preservation and all causal-drop checks pass. Gate report SHA-256 is
`cdd5a717e55cb3c589fecdefc7455a83903f9ef68376b822c3846f9da7573e8c`.

## Decision

Close exact CRP1 without a nearby seed, width, duration, loss, renderer,
threshold, or prompt repair. Preserve its demonstrated first-error packet and
the `182` audited complete replay successes as causal evidence. A successor
must change the execution/readout substrate materially: packet localization
must drive one persistent model-owned state replay rather than merely
conditioning a frozen language generator, and it must transfer across numeric,
register, and symbolic state domains before natural-trace promotion.
