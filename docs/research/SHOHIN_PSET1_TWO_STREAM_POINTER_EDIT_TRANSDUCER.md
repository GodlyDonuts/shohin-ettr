# PSET1: Two-Stream Pointer Edit Transducer

Status: frozen prospective development canary, 2026-08-09. DSET1-v0 is closed.
No PSET1 model output exists at freeze time. Holdout remains unopened.

## Hypothesis

DSET1 proves causal draft use but loses reliability while autoregressively
serializing an old surface and a replacement surface. PSET1 instead treats
revision as sequence transduction. The source and draft have separate causal
streams. A model-owned policy emits an operation, draft-character pointers,
and at most sixteen UTF-8 replacement bytes. A generic deterministic executor copies every
untouched draft byte and applies the predicted splice. It has no semantic
knowledge, verifier, solver, answer label, task route, or repair rule.

For a Unicode draft `D=(c_0,...,c_n)`, PSET1 predicts

\[
 a\in\{KEEP,REPLACE\},\quad i,j\in[0,n),\quad r_{1:m},\ m\le 16.
\]

`KEEP` returns the exact draft bytes. `REPLACE` requires valid character
boundaries and returns `draft[:i] + utf8_decode(r) + draft[j+1:]`.
Malformed pointers, absent offsets, overlong replacement, or decode mismatch
fail closed. The final trajectory is therefore causally impossible without the
draft: all but the bounded edit are copied through model-owned pointers.

## Encoder and edit head

- pinned OLMoE host and frozen DSET1 aligned update-512 stage owner;
- all host weights, routers, experts, and the DSET residual frozen;
- source problem and draft encoded in separate frozen forward passes;
- shared 256-wide projections followed by one draft-query/source-key
  cross-attention block and feed-forward block;
- action and start/end pointer heads over character states obtained by
  broadcasting each contextual draft-token state to its tokenizer-reported
  character span and adding a learned hashed-character embedding;
- a two-layer 256-wide autoregressive replacement decoder cross-attending to
  source states and the selected draft state;
- replacement input/output is a compact 257-symbol byte alphabet (256 bytes
  plus EOS), avoiding BPE-boundary ambiguity;
- no full-response autoregressive decoder.

The aligned and label-permuted heads have identical initialization, parameter
count, source/pair order, updates, optimizer, and token geometry. The
label-permuted arm trains clean drafts to restore the known fault and fault
drafts to KEEP. Draft-hidden and same-family near-length shuffled-draft
interventions are evaluation-only causal falsifiers of the aligned head.

## Data and custody

Stage 0 reuses only the already-opened DSET1 development lineage:

- 4,096 DSET1 train identities and 256 DSET1 diagnostic identities, selected
  deterministically at the frozen 7:1 numeric/choice ratio;
- exact train/diagnostic source disjointness inherited and rechecked;
- tokenizer offsets must cover every draft character exactly once;
- replacement UTF-8 bytes must decode byte-for-byte to the registered new
  surface and contain at most sixteen bytes;
- source and complete draft each fit 4,096 tokens independently;
- all selection, drop, span, offset, and hash receipts are stored.

No public benchmark or sealed holdout is part of Stage 0.

## Optimization

- seed `2026080917`;
- 512 updates, one same-source clean/fault pair per update;
- AdamW, LR `3e-4`, cosine decay, fused optimizer;
- equal aggregate action, start-pointer, end-pointer, and replacement-token
  losses on fault examples; clean examples contribute action loss only;
- gradient norm clipping at 1.0;
- maximum replacement generation sixteen bytes plus EOS.

## Frozen Stage-0 gate

All conditions are conjunctive on the 256-identity source-disjoint diagnostic:

1. aligned exact edit-program accuracy `>=95%` overall and per family;
2. aligned executed complete-trajectory accuracy `>=95%`;
3. aligned clean copy `>=99%` and fault repair `>=90%`;
4. aligned same-source counterfactual consistency `>=95%`;
5. aligned exceeds label-permuted, draft-hidden, and shuffled-draft executed
   accuracy by at least 13 answers each;
6. hidden and label-permuted exact edit-program accuracy are each `<=60%`;
7. forcing the gold program versus a wrong program changes downstream executed
   trajectories in the expected direction on at least 95% of fault rows;
8. zero accepted malformed pointers, offset violations, replacement decode
   mismatches, or generation exhaustion;
9. complete parameter, update, token, FLOP, memory, latency, and hash receipts.

A one-update mechanics receipt precedes the two fits. A miss closes exact
PSET1-v0 without width, layer, seed, duration, pointer, or replacement-budget
variants. A pass permits one broader natural-draft gate, not a public reasoning
claim.

## Controls and claim boundary

DSET1 full-script generation, source-only/draft-hidden, shuffled draft, and
label-permuted pointer learning are explicit controls. A pass would establish a
reliable model-owned copy/edit mechanism on deterministic final-span faults.
It would not establish arbitrary multi-edit planning, natural fault discovery,
general reasoning, or large-MoE transfer.
