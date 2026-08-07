# DIVERGE-GTI1: Generative Transaction Interpreter

Status: frozen after the PQI1 development failure and before GTI1 training or
evaluation. PQI1 confirmation remains unopened.

## 1. Failure boundary and hypothesis

PQI1 fit all `100,000` counterfactually complete QUERY rows with both protected
Shohin and exact imported SmolLM2-135M, yet both emitted the identical
development assignments: `640/768`, with one complete `0/128` renderer
inversion. The shuffled-label control reached `128/768`. A stronger pretrained
residual therefore did not rescue a shallow pooled mention scorer.

GTI1 changes the computation boundary rather than the PQI1 layer, width,
duration, seed, renderer, tokenizer, or threshold:

> Preserve the pretrained decoder's causal language computation, adapt only
> low-rank projections in its final four blocks, and make the model emit one
> complete typed `READ(alpha|beta)` transaction autoregressively. Score whole
> transactions by conditional sequence likelihood; never classify pooled
> mention fields independently.

The transaction is model-owned but the existing source symbol table still
supplies the two mention groups. GTI1 is a semantic-interface gate, not an
open-domain reasoning or architectural-novelty claim.

## 2. Frozen data and model arms

All arms use the exact immutable RRG1 QUERY corpus: `100,000` rows, `50,000`
complete role-order pairs, ten lexical families, two clause forms, and SHA-256
`2d325c860e707307886f782350e7ec35ae8c23ae275260b0a937bbb738078c1c`.
The opened CCR1 board with SHA-256
`299237068f436ba33a68487b5300fcd724f8c98bd8bfe6b1916a4ebc7541ebf7`
is development-only. The already-generated PQI1 board with SHA-256
`27f198680cc7bcd7e0203949fe4dee1658fc057fe75302b7e2d43d74321201b8`
remains sealed until admission.

Matched arms are:

1. protected Shohin step-300k plus GTI1 LoRA;
2. exact imported SmolLM2-135M-Instruct plus GTI1 LoRA; and
3. the same SmolLM2 arm with one fixed shuffled transaction target per row.

Shohin and SmolLM2 share the same 576-wide, 30-layer, 9-head geometry. Each
real arm installs rank-8, alpha-16 LoRA in every linear projection of the final
four blocks. The embedding, LM head, first 26 blocks, normalization, and every
non-LoRA tensor stay frozen and hash-checked.

## 3. Frozen transaction protocol and budget

Source names are replaced by anonymous `alpha` and `beta` mention identities
using the same gold-independent runtime mention masks as PQI1. The fixed prompt
is:

```text
Read the instruction and emit exactly one typed transaction.
Instruction: {canonical query}
Transaction:
```

The only legal completions are ` READ(alpha)` and ` READ(beta)`. The model is
trained by causal LM loss only on completion tokens. Inference computes the
complete conditional log likelihood of both legal transactions and commits to
one argmax transaction; no tokenwise repair, beam, parser fallback, threshold,
or answer/state signal exists.

Every arm receives one epoch, batch 128, seed `2026080641`, AdamW betas
`(0.9, 0.95)`, weight decay `0.01`, gradient clip `1.0`, peak LR `1e-4`, and
cosine decay to zero. The shuffled control uses the same order, update count,
and optimizer state geometry.

## 4. Development gate

Promotion is conjunctive:

- SmolLM2 QUERY at least `765/768`;
- every mode at least `254/256` and every renderer at least `127/128`;
- SmolLM2 exceeds protected Shohin by at least `32/768`;
- shuffled SmolLM2 is at most `430/768`;
- replacing semantic context by occurrence order loses at least `250`;
- swapping `alpha` and `beta` in the source flips at least `765/768`
  transactions after mapping back;
- arbitrary entity renaming changes zero canonical prompts and predictions;
- all non-LoRA backbone tensors are bit-identical before and after training;
  and
- every emitted transaction is one of the two legal complete transactions.

Only that matched assessor may open the sealed PQI1 board. A miss closes GTI1
without prompt, completion vocabulary, rank, layer, duration, LR, seed,
renderer, tokenizer, or data variants.

## 5. Confirmation and successor

Direct confirmation repeats the absolute semantic and causal floors. A pass
authorizes one zero-retraining composition with protected TOL3 WORLD, protected
NVE1 EVIDENCE, exact factorized execution, and the qualified GTI1 QUERY owner.
Only a complete end-to-end composition pass may connect the already-qualified
oracle PL1 plasticity mechanics to natural language. Neither a direct GTI1
pass nor a component composition authorizes continuation pretraining.

If SmolLM2 passes while Shohin fails, the result establishes a same-geometry
pretraining capability floor and the qualified Smol owner becomes a teacher or
scratch-training target. If every real arm fails, the isolated semantic-parser
sequence is closed; the next system must learn transaction semantics inside a
broader end-to-end natural task rather than add another token/span/pointer or
renderer-local reader.

## 6. Result: closed negative

Immutable commit `4e3b6c5` ran matched one-H100 jobs
`744544/744545/744546`; assessor `744547` wrote its report and exited `1` on
the frozen gate. All three arms completed exactly 782 updates. Training
exactness is `99,291/100,000` for Shohin, `99,989/100,000` for SmolLM2, and
`50,060/100,000` against shuffled supervision (`49,486/100,000` against true
roles). The shuffled arm therefore stays at chance while both real arms fit.

Transfer is negative:

| Arm | Development QUERY | Exact renderers |
|---|---:|---|
| Shohin | `512/768` | 1, 2, 3, 5 |
| SmolLM2 | `384/768` | 1, 3, 5 |
| shuffled SmolLM2 | `384/768` | 2, 3, 4 |

Every miss is a complete `0/128` renderer block. Context scrub is exactly
`384/768` with zero mean signed margin for every arm. Mention swapping is not
equivariant, while entity-renaming prompts, predictions, and score tensors are
bit-exact and every non-LoRA tensor remains unchanged. The system learned
template-level transaction lookup rather than target/distractor meaning.

Development-result SHA-256 values are
`b81677ac3a26b55abb2151b3fb66860c2537ba8393504aedee15add7b71eaf96`,
`1c4b759e56f1c3902ba26874577dbf98369630d78c185bef244f05c2d0826002`,
and `b06b874e0fe447b18a1b4d5e717656beb28e060b91ce7bcd560e970ce1f6498f`.
The matched assessment SHA-256 is
`62176a6409700f89c185c888f8697badf8fda3037c2568b8f800f90a89edf231`.
Fail-closed confirmation job `744549` was canceled by dependency and the
sealed board was never opened. GTI1 is closed without prompt, LoRA, layer,
duration, seed, renderer, tokenizer, or data variants.
