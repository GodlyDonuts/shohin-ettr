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

## Result: closed negative

Matched H100 jobs `744506/744507/744508` and assessor `744512` completed the
single development gate. Protected Shohin and exact imported SmolLM2 both
score `640/768`; shuffled SmolLM2 scores `128/768`. The two real arms emit the
same assignment vector: renderers 0--4 are each `128/128`, while renderer 5 is
`0/128`. Forced role-slot swap makes renderer 5 exact and all others zero.
Context scrub falls to `384/768`; arbitrary entity renaming is bit-exact. The
stronger pretrained residual therefore does not change the failure through
this pooled mention-scoring interface.

The matched assessment SHA-256 is
`c8b482de5384e8b75796c34436948e26927bde3ef3289deb696322582ab0db48`.
The three development-result SHA-256 values are
`e7bdfbfc7c5e077f390de8a4dd870cfc303ae55236d4aa23664632594f09b3bb`,
`588ad26fa2ee88c3f03c9e06b433c91712f16e963fbe190854191f875fb62e2d`,
and `7ebb61273220bcf4b67e18cf49985c65fab73748fb81be68898a84a0d9a6f7a7`.
Confirmation stayed sealed.

Read-only autoregressive attribution jobs `744542/744543` score both raw
backbones at `384/768`, with renderers 0/2/4 exact and 1/3/5 zero. Their result
hashes are `fc53ad879ecd6a42c7906509879ebc47fca7105dd34b9b24d2d702a080d87f5e`
and `b92c481ad0efe12d4895f6146d317ad97fb2978264bc2791fbdd947bfc332f1e`.
Independent Stokes job `767010` regenerated the sealed board byte-for-byte
with SHA-256
`27f198680cc7bcd7e0203949fe4dee1658fc057fe75302b7e2d43d74321201b8`.
PQI1 is closed without variants.
