# CFR1: Verified Counterfactual Revision

Status: data mechanism frozen before corpus output.

## Hypothesis

IDR1 proves that a same-family source-plus-draft revision role can improve
reasoning, but 34.1% of its training presentations use answer-only repair
targets. VFR1 proves that generating replacement traces with the current 9B
teacher is not reliable enough. CFR1 removes that dependency: start from a
source-disjoint, benchmark-decontaminated corpus whose complete solutions
already pass answer or execution verification, construct one deterministic
counterfactual fault in the model-visible draft, and train the revision role
to emit the untouched verified full solution.

Each source contributes two presentations:

1. `verified_clean`: draft and target are the same verified full solution;
2. `counterfactual_fault`: math/science/procedural drafts receive a
   contradictory wrong final answer, while code drafts receive a guaranteed
   runtime failure before the verified program. The target remains the exact
   verified full solution.

This is deliberately a bounded fault curriculum, not a claim that one
synthetic corruption family covers natural reasoning errors.

## Matched control

The control has identical source questions, exact targets, row order, target
token multiset, model initialization, optimizer, and update budget. It sees a
different source's clean/faulted draft from the same domain and nearest
available draft-length ordering. No source is assigned its own draft. This
separates learning from aligned revision state from generic source-to-solution
SFT.

## Data gate

The builder consumes the new Qwen-tokenizer-specific 16M verified mix. It
must report:

- zero source/donor identity matches and identical target multisets;
- equal aligned/control rows with one clean and one fault presentation per
  admitted source;
- zero prompt or target truncation at 4,096 tokens in either arm;
- only math, science, execution-verified code, and answer-verified procedural
  sources; and
- complete source, tokenizer, output, and report hashes.

The unverified teacher bucket is excluded. A failed data gate closes CFR1
before training.

## Capability gate

After data admission, train exactly one aligned and one shuffled arm from the
immutable 9B B1 adapter with identical final-four rank-8 LoRA, 512 updates,
batch 1, accumulation 8, 4,096 context, learning rate `2e-5`, and matched
seeds. Evaluate once on the existing 1,289-row source-disjoint IDR1
development board. Promotion requires all of:

- aligned at least `603/1,289`;
- aligned at least `+10` answers over shuffled draft control;
- math at least `223`, logic/science at least `349`, and code at least `17`;
- complete matched token, parameter, memory, latency, and protected-hash
  receipts.

Any miss closes exact CFR1 without corruption, ratio, update, rank, layer,
seed, or threshold rescue. Holdout remains sealed until a conjunctive pass.
