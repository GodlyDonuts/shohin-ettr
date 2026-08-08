# Shohin Phase 2 Scale Decision

Status: capacity-boundary decision live, 2026-08-08. The protected 4B product
aggregate closed as a mixed conjunctive failure; the frozen 0.8B transfer now
selects the scratch scale. This document does not authorize a large pretraining
launch by itself.

## Evidence Boundary

The original 125.08M Shohin completed 300,000 updates and 157.286B nominal
token presentations over a 57.826B-token mounted corpus. It is a useful
protected baseline, but its public and direct-interaction results do not meet
the project objective.

The strongest causal result is now same-family internal draft/revision:

- pinned Qwen3.5-4B draft owner;
- the same 4B base with a learned revision state;
- one complete internal draft followed by one complete revised answer;
- no external proposal model, tool, verifier, answer router, or host solver at
  inference.

On source-disjoint data, trained revision beats the unchanged second pass by
`+158/1,289` development answers and `+174/1,279` holdout answers. Every math,
logic/science, and executable-code delta is positive. On the protected
568-case product board, treatment scores `320/538 = 59.48%` and `61.391%`
five-domain macro versus unchanged-B1 `272/538 = 50.56%` and `51.053%`.
Science improves `52->90` and code `6->21`, but GSM8K, MATH-500, and BBH logic
regress by `2/1/2` answers. The no-domain-regression product gate therefore
fails despite the `+48` aggregate lift. The 4B artifact is retained as causal
evidence and is not packaged as a qualified product.

## Candidate Trunks

Both scratch candidates use the existing pre-norm GQA/SwiGLU/QK-norm Shohin
trunk, tied embeddings, a 4,096-token training context, and a 49,152-token
vocabulary. Parameter counts are exact and exclude no trainable tensor.

| Candidate | Layers | Width | Heads/KV | FFN | Exact parameters |
|---|---:|---:|---:|---:|---:|
| `shohin_390m` | 30 | 1,024 | 16/4 | 2,816 | 388,563,712 |
| `shohin_920m` | 32 | 1,536 | 24/8 | 4,352 | 918,656,512 |

The training CLI requires `--vocab-size 49152`; tokenizer selection remains an
explicit launch field rather than being hidden in the size preset.

The deployed reasoning architecture is a shared trunk with two model-owned
temporal roles:

1. **Draft role:** generate one complete source-only trajectory.
2. **Revision role:** reread source plus the internal trajectory and generate
   one coherent corrected answer.

The initial implementation may store the two role states as adapters. A later
joint checkpoint may package them together, but packaging cannot change the
measured inference path.

## Scale Selection

The decision is capability-driven rather than parameter-count-driven.

1. Preserve the closed 4B mixed failure without local threshold, prompt,
   decoding, rank, or duration rescue variants.
2. Run one matched 0.8B internal-draft/revision transfer using the already
   pinned Qwen3.5-0.8B B1 state, exact source-disjoint bank, exact 256-update
   revision schedule, and unchanged control.
3. Select `shohin_390m` only if 0.8B preserves at least a five-point revision
   gain with nonnegative math, logic/science, and code deltas on both splits.
4. Otherwise select `shohin_920m`. Do not spend the remaining compute on a
   scratch scale already contradicted by the capacity-boundary result.

The frozen 0.8B campaign is live as jobs `745961--745979`. Seventeen
single-H100 source-only draft shards feed one hash-bound data build and an
automatic train/evaluate/compare dispatcher. The comparison is conjunctive
across development and holdout and writes the scale decision once; there is no
nearby 0.8B retry family.

This test determines the smallest plausible scratch trunk. It is not a nearby
variant of the failed QST1 workspace: the changed factor is the already
qualified complete-trajectory revision mechanism.

## Data Requirement

Current optimizer-authorized general data are 62.426B historical tokens from
FineMath, OpenWebMath, and Python code. This stream is too narrow and too small
for the intended scratch run without heavy replay.

The most developed fresh candidates include:

- 8.764B selected FineWeb-Edu tokens;
- 4.350B balanced peS2o technical/science tokens;
- 0.308B Essential-Web reasoning-core tokens;
- bounded FinePDF core and verified math/science/code sources.

They remain candidates until their existing provenance, residualization,
privacy, contamination, license, and utility gates close. The production
corpus must be mostly fresh broad data rather than repeated historical math
and code.

Minimum staged targets:

| Stage | Purpose | Unique admitted tokens |
|---|---|---:|
| transport/optimization canary | prove multi-H100 scaling and resume | 0.25B |
| capability canary | compare 390M/920M learning at equal tokens | 5B |
| first milestone | broad NLL and downstream-skill checkpoint | 50B |
| scale run | competitive base representation | >=300B for 920M or >=700B for 390M |

Repeated presentations are reported separately and never counted as unique
corpus size.

## Training Sequence

1. **Base pretraining:** broad next-token training with source-balanced token
   sampling and immutable evaluation monitors.
2. **Reasoning mid-training:** interleave verified explanatory math, code,
   science, logic, and long-form decomposition without answer-only oversampling.
3. **General instruction alignment:** teach concise direct answers, complete
   solutions, code boundaries, and refusal/uncertainty behavior.
4. **Internal draft/revision training:** train both temporal roles on complete
   trajectories, including initially wrong drafts that can be repaired.
5. **Verified-reward training:** begin only after the model generates a useful
   positive rate on independently checked math/code/science tasks; preserve an
   identical no-RLVR continuation control.
6. **Milestone evaluation:** greedy pass@1 plus separately reported test-time
   compute on GSM8K, MATH-500, AIME, executable code, GPQA, and logic, followed
   by direct interaction and untouched distribution-shifted boards.

## Launch Gate

A large run starts only after all of the following are true:

- the 4B product comparison and 0.8B capacity-boundary decision are recorded;
- the selected configuration instantiates at its exact parameter count;
- 1/2/4/8/16-H100 scaling is measured and the efficient geometry selected;
- the physical data contract proves source weights, unique tokens, tokenizer,
  licenses, contamination scans, and shard hashes;
- exact checkpoint/resume and data-cursor continuation pass;
- a 5B-token matched canary shows stable optimization and no broad regression;
- projected H100-hours are recorded against current quota before launch.

The target is not merely a benchmark-specialized adapter. The target is one
deployable Shohin model that retains broad language and code utility while its
own draft/revision path materially improves difficult reasoning.
