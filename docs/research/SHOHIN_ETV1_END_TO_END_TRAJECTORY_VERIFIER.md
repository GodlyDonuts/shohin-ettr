# ETV1: End-to-End Whole-Trajectory Verifier

**Status:** prospectively frozen after TSVC1 failure and before ETV1 output  
**Date:** 2026-08-10  
**Host:** pinned Qwen3.6-35B-A3B  
**Scope:** source-disjoint development; holdout sealed

## Thesis

TSVC1 reaches 99.61% validation on held-out deterministic clean/fault pairs
but transfers to only 69/125 model-generated disagreements, exactly equal to
the source-shuffled control. A frozen representation plus small head therefore
learns corruption style rather than source-conditioned semantic correctness.

ETV1 makes the verifier representation itself trainable. It retains the frozen
Qwen base, router, and experts, but updates the existing 1,179,648-parameter
shared post-MLP revision residual during a dedicated verifier pass together
with the existing process-verifier head. Each training unit is the same source
with one verified clean and one deterministic faulted complete trajectory;
pairwise ranking and balanced binary losses force source-candidate comparison.
At inference ETV1 scores each complete candidate and commits to one lineage.
It cannot edit, regenerate, merge, execute, or consult a host verifier.

## Frozen experiment

- exact TSVC1-r3 train/aligned/shuffled candidate corpora;
- pinned Qwen+DSET initialization and NF4/BF16 compute;
- process-verifier `leader` scope, 300 updates, minimum 100, gradient
  accumulation 8, two candidates per identity, backbone LR 2e-6, head LR
  2e-4, seed 20260809;
- 4,096-token context with zero accepted truncation;
- exact 125-group aligned and source-shuffled model-candidate diagnostics;
- label-blind shape selector and frozen TSVC/WTV results as controls.

## Gate

All conditions are conjunctive:

- training, aligned evaluation, and shuffled evaluation have zero truncation;
- internal final split on training identities selects at least 90%;
- aligned selects at least 105/125 disagreement trajectories;
- combined exactness is at least 1,874/1,908;
- choice exactness is at least 220/256;
- aligned exceeds source-shuffled by at least 13 disagreement rows;
- aligned exceeds shape-only by at least 13 disagreement rows.

A pass authorizes one sealed holdout and candidate-producer consolidation. A
miss closes this verifier family on the current data/host without scope, LR,
duration, head, layer, seed, or prompt variants.

## Claim boundary

A pass would establish a trained same-host semantic commit over coherent
model-owned trajectories on a current MoE. It would still incur extra
candidate and verifier passes; efficiency and transfer remain separate gates.
