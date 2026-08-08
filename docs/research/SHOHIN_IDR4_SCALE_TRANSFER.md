# Shohin IDR4 Scale Transfer

Status: exact matched source-disjoint attribution passes on both splits. The
protected product confirmation is now permitted and live.

## Capability Hypothesis

The 9B IDR1 result established that training a model to revise its own earlier
trajectory contributes 130 source-disjoint answers over an untrained second
pass and 58 protected product answers. IDR4 asks whether that learned temporal
reasoning survives when both owners are the same pinned Qwen3.5-4B family.
This is a scale intervention, not a new architecture search.

At inference, one 4B B1 owner generates a complete source-only draft. A later
4B revision owner receives only the original source and that internal draft,
then emits one complete final solution. There is no 9B/4B cross-host proposal,
external verifier, task router, answer label, or tool.

## Frozen Inputs And Compute

- Backbone: `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Draft owner: existing 4B B1 checkpoint SHA-256
  `f7354e6a0c4311ad792b73358b4e62d9dbe0ae1bd2d41896cf55482d9ce81feb`.
- Bank: exact IDR1 source-disjoint 4,096 math, 4,096 logic/science, and 200
  execution-verified code identities.
- Draft decoding: greedy, one sample, no thinking mode, batch 4, 768 new
  tokens, seed `2026080818`.
- Revision: exact IDR1 verified targets, split, 256-update LoRA geometry,
  optimizer, context, order, and seeds, warm-started from the 4B B1 adapter.
- Estimated cost: `10--16` H100-hours for drafts and `2--4` for fit plus
  source-disjoint evaluation. This estimate was declared before launch.

Jobs `745767--745783` are isolated single-H100 draft shards. No downstream fit
may start until their merger proves all 8,392 identities exactly once and
binds the pinned model revision, B1 checkpoint, bank hashes, decoding fields,
and every candidate/report hash.

## Draft And Training Result

All 17 draft shards completed cleanly for `16.952` H100-hours. The 4B B1
first pass solves `2,276/8,392 = 27.12%`:

| Domain | Correct | Exhausted at 768 tokens |
|---|---:|---:|
| MATH | `657/4,096` | `2,931/4,096` |
| Logic/science | `1,500/4,096` | `1,897/4,096` |
| Execution-verified code | `119/200` | `7/200` |

Merged draft SHA-256 is
`7d7e833de5b646ff18e08780c8d6760fb0d4a5ea8d6b3fd43c61c6d6e4a7c5e4`;
receipt SHA-256 is
`d91138a34a294f5edbc472272946e43f46ff11c296fc21028e04d9171f19f7ce`.
Train/development/holdout data SHA-256 values are `a40d209a...ee7b`,
`f7444eef...afa1`, and `ec99bc35...7a70`.

Revision fit `745819` completed all 256 updates in 16m32s, charging 365,028
target tokens at 382.4 target tok/s. Checkpoint SHA-256 is
`ae3847fe0728b1debcc13049822ea7499f744836b62d6d1c5bcb7c1000d8560b`.
The trained revision scores `529/1,289 = 41.04%` on development and
`554/1,279 = 43.32%` on holdout. The corrected eight-shard unchanged-B1
development control scores `371/1,289 = 28.78%`, establishing a
`+158`-answer / `+12.26`-point development gain. Domain deltas are all
positive: math `140->208`, logic/science `223->305`, and executable code
`8->16`.

The corrected holdout control scores `380/1,279 = 29.71%`, versus trained
revision `554/1,279 = 43.32%`: `+174` answers and `+13.61` absolute points.
Holdout domain deltas are math `151->217`, logic/science `223->319`, and
executable code `6->18`; every development and holdout domain is positive.
The control report SHA-256 is
`a3f4c824e644b13aa412535c553edb5706305430780d604fa114b074a0a03248`.

Two original holdout shards on `evc46` failed before model execution because
CUDA was unavailable; exact shard replacements `745877/745878` excluded that
node and completed cleanly. No examples, weights, prompts, thresholds,
decoding fields, or evaluator code changed. Corrected merge `745879` proves
all eight shard receipts and complete identity coverage.

## Matched Gate

The protected baseline is the original 4B B1 adapter performing the same
source-plus-draft second pass with identical prompts, decoding, evaluator, and
token budget. The treatment changes only the trained revision LoRA state.

Pass requires all of:

1. trained revision improves overall source-disjoint accuracy by at least five
   absolute percentage points over the matched untrained second pass;
2. math, logic/science, and executable-code deltas are each nonnegative;
3. the direction reproduces independently on development and holdout;
4. complete custody and generation receipts pass without exclusions.

The pass opens the unchanged protected product board exactly once. Product
confirmation uses a new 4B B1 internal draft for every protected identity,
then compares the trained 4B revision owner with the unchanged 4B B1 second
pass on those identical drafts. It does not reuse 9B product trajectories.
Before any product completion was opened, confirmation success was frozen as
at least 27 additional correct main-board answers, at least `+0.05` five-domain
macro accuracy, and nonnegative correct-answer deltas in every one of the five
main domains. AIME is reported separately rather than used as a 30-item gate.

## Transfer Role

IDR4 is the strongest clean within-Qwen source-disjoint transfer point and a
control for the transferable architecture campaign. Together with 0.8B and 9B
it defines a scale curve, but it does not select a scratch Shohin trunk. The
next missing intervention is model-family transfer under TTR1; the protected
125M checkpoint and former 390M/920M candidates are optional efficiency
baselines rather than the primary deployment target.
