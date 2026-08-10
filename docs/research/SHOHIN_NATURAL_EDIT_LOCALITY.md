# Natural Trajectory Edit-Locality Audit

Status: completed model-free development audit on 2026-08-10; raw-text pointer
successor rejected before GPU use.

## Question

After the DSET/GSET/ISET/FRET/RIFT/OCET/RSOT/BSOT cascade closed, one proposed
structural escape was to replace full autoregressive rewriting with explicit
COPY/DELETE/REPLACE/INSERT operations over a model-owned draft. That is useful
only when independently generated wrong and correct trajectories share enough
ordered text to make copying materially cheaper and easier than regeneration.

This audit tests that prerequisite on the immutable CVG1 natural candidate
corpus. It is not a model score and does not open a sealed evaluation.

## Custody

- Input: `cvg1_pairs_a960206_r1/pairs.jsonl`
- Input SHA-256:
  `45f1d66ce5e87dc2a1f4c3594bdde2bae26e9417e879d16eb4eddb228b696afe`
- Scored splits: `train`, `development`
- Holdout scored: **false**
- Eligible rows: exactly two natural candidates with exactly one independently
  correct candidate
- Train: 1,449 pairs (`753` logic/science, `673` MATH, `23` code)
- Development: 180 pairs (`96` logic/science, `81` MATH, `3` code)

The single-splice ceiling preserves the longest exact prefix and nonoverlapping
suffix of the correct trajectory. The multi-span ceiling uses an unrestricted
ordered character diff. Both are optimistic upper bounds: they assume an
oracle already knows every copy address and replacement boundary.

## Result

| Metric | Train | Development |
|---|---:|---:|
| Correct trajectory, median characters | 709 | 656 |
| Single-splice copy fraction, mean | 1.2346% | 1.0245% |
| Single-splice copy fraction, median | 0% | 0% |
| Single-splice replacement, median characters | 706 | 653.5 |
| Multi-span copy fraction, mean | 42.0460% | 44.9179% |
| Multi-span copy fraction, median | 41.0714% | 44.6383% |
| Multi-span copy runs, mean | 53.12 | 56.97 |
| Multi-span copy runs, median | 48 | 51 |

Report SHA-256:
`c8b73a13d6c243accf34a6e770a8660e2ffb0330d9d6acf16586c5f0c799f1a9`.

## Decision

Reject a raw-natural-text pointer/edit model as the immediate successor. A
single edit degenerates into replacing essentially the complete answer. An
unrestricted multi-span model still must infer roughly fifty copy fragments
and generate more than half of the target. That adds alignment and execution
failure modes without reducing the hard semantic generation problem.

This result reinforces, rather than reopens, the prior PSET1 negative. PSET1
already showed that a separate byte replacement decoder was the failed owner;
the natural corpus additionally shows that free-form trajectories do not
provide a favorable copy geometry.

## Next Architectural Boundary

Any further explicit transduction must create edit locality by construction.
The next admissible mechanism is a structured addressable draft ledger:

1. a first model-owned pass compiles the source into canonical typed
   operations, values, dependencies, and state snapshots;
2. revision acts on whole ledger records with stable addresses, never on
   arbitrary prose fragments;
3. one tied recurrent state core replays the committed ledger;
4. a final model-owned renderer emits the user-facing trajectory; and
5. hidden-ledger, shuffled-record, state-reset, and equal-compute
   full-regeneration controls test causality.

No H100 fit is authorized until a CPU data admission proves exact executable
records, broad operation coverage, source-disjoint splits, and a substantial
copy/edit advantage over raw natural text. Existing TOL3/NTA3/NVE1 mechanics
are controls and reusable interfaces, not evidence that this broader natural
compiler already works.

