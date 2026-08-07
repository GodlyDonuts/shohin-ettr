# DIVERGE-PQI1: Pretrained Antisymmetric Query Interpreter

Status: frozen after STI1 closure and before data materialization, training, or
neural evaluation. This is one structurally different semantic-interface gate,
not an RRG1/STI1 variant.

## Hypothesis

STI1's fresh byte encoder is exact on five query renderers and exactly inverted
on the sixth. Its mention-marker deletion has no effect. The architecture has
therefore learned renderer polarity without a transferable language semantic.

PQI1 tests whether an already pretrained language residual plus a role-
symmetric pointer is the missing substrate:

> A frozen pretrained 135M language backbone can represent unseen lexical
> target/distractor polarity, while one shared antisymmetric candidate scorer
> prevents independent stage labels or free role slots from selecting an
> incoherent assignment.

Every source entity is replaced by `alpha` or `beta` according only to its
model-supplied occurrence group. Frozen layer-19 residuals are pooled over the
two mentions and the whole query. One shared function emits candidate scores
`s0,s1`; the only legal assignment logits are `(s0-s1, s1-s0)`. Swapping role
columns therefore inverts the assignment exactly by construction. No evidence,
execution, answer, renderer ID, or hidden target label enters inference.

## Matched arms

Three independent one-H100 arms receive the exact 100,000 counterfactually
complete RRG1 QUERY rows, one epoch, example order, batch 128, AdamW schedule,
seed, adapter geometry, and direct evaluator:

1. protected Shohin step-300k backbone;
2. exact imported SmolLM2-135M-Instruct backbone; and
3. SmolLM2 with role labels shuffled once before training.

Shohin and SmolLM2 have the same 576-wide, 30-layer, 9-head geometry. Their
tokenizers differ and token/compute receipts must be reported. The intervention
is the pretrained backbone plus tokenizer substrate, not a claim about weights
alone. All backbone weights remain frozen.

## Split custody

The opened CCR1 board is development-only. Before any arm trains, a new 256-
episode confirmation board is materialized at seed `2026080630` with 32 new
entities and six new query renderers. Exact source, query, identity, and symbol
overlap with RRG1 training, SRP1, and CCR1 must be zero. Selection is generator-
only and records `model_score_used=false`.

Confirmation remains unread by candidate evaluators until all three
development arms and the matched assessor pass. No confirmation template,
threshold, seed, or name may change after training.

## Development gate

The Smol arm must satisfy every condition:

- at least `765/768` exact query assignments;
- every mode at least `254/256` and renderer at least `127/128`;
- at least 64 more exact assignments than the matched Shohin arm;
- shuffled-label Smol at most `430/768`;
- forced role swap loses at least 500 assignments;
- context scrub loses at least 250 assignments; and
- arbitrary entity renaming changes zero assignments and zero logit bits.

Only that conjunction opens confirmation. A development miss closes PQI1
without backbone layer, adapter width, duration, seed, learning-rate, label,
renderer, tokenizer, threshold, or normalization variants.

## Confirmation and natural-plasticity boundary

Direct confirmation repeats the absolute semantic and causal floors. A pass
then permits one zero-retraining composition with protected TOL3 WORLD and NVE1
EVIDENCE. That composite must recover the STI1 end-to-end floors: WORLD and
sealing at least `255/256`, EVIDENCE at least `3,070/3,072`, QUERY at least
`765/768`, and sensitive/invariant/abstention metrics at least `254/256`, with
protected hashes exact.

Only a full composite pass may activate the already frozen natural PL1
contract on development data. PQI1 does not authorize continuation pretraining
or establish open-domain reasoning by itself.

