# TSVC1: Trained Source-Trajectory Semantic Commit

**Status:** prospectively frozen after WTV1 failure and before TSVC1 features  
**Date:** 2026-08-10  
**Host:** pinned Qwen3.6-35B-A3B  
**Scope:** source-disjoint development; holdout sealed

## Thesis

The edit arms contain an exact complete trajectory on all 125 development
disagreements, but an untrained natural-language verifier selects only 56.
TSVC1 replaces prompt-based judgment with a trained, small semantic commit
head over frozen same-host hidden states. It receives the source and one
complete candidate trajectory, scores candidates independently, and commits to
one whole trajectory. It cannot edit, merge, regenerate, execute, or consult a
verifier at inference.

The training objective is identifiable. Each training identity presents the
same source with exactly two complete trajectories: the verified clean draft
and its deterministic faulted counterpart. Exactly one is correct, candidate
order is hash-randomized, and a pairwise ranking loss directly supervises the
commit decision. The Qwen+DSET host stays frozen. Only the existing compact
late-state correctness head is trained.

## Frozen experiment

- Training: all 7,639 source-disjoint ISET training pairs, one clean and one
  faulted complete trajectory per source.
- Development: the exact 125 DSET/GSET/ISET disagreement groups from WTV1,
  with duplicate complete trajectories removed.
- Causal control: the identical development candidates with sources rotated
  within corruption-family and clean/fault buckets. A singleton bucket rotates
  within the same corruption family across pair member so every source is
  genuinely replaced.
- Baseline: the existing label-blind completion-shape reranker at matched
  candidate geometry.
- Frozen host features: final, tail-mean, and completion-mean pools from layers
  -1, -2, -4, and -8; complete source/candidate custody at 4,096 tokens.
- Head: existing 128-wide correctness head and frozen pairwise+BCE training
  implementation; no backbone, adapter, router, or expert update.

## Gate

All conditions are conjunctive:

- zero source/candidate truncation in train, aligned, and shuffled features;
- aligned selects at least 105/125 exact disagreement trajectories;
- combined aligned exactness is at least 1,874/1,908;
- aligned choice exactness is at least 220/256;
- aligned beats the source-shuffled control by at least 13 disagreement rows;
- aligned beats the shape-only selector by at least 13 disagreement rows;
- the head's hash validation split on training identities selects at least 90%.

A pass authorizes one sealed holdout and an efficiency consolidation study. A
miss closes this exact semantic-commit interface without head-width, layer,
seed, duration, or loss variants.

## Claim boundary

A pass would establish model-owned semantic commitment over complete
source-conditioned reasoning trajectories on a current MoE. It would not yet
show that three candidate producers are compute-efficient, nor establish a
new reasoning primitive independent of candidate diversity.
