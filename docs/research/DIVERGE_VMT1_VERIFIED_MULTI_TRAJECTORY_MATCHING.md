# DIVERGE-VMT1: Verified Multi-Trajectory Matching

Status: frozen fit-only successor after the negative LTM1 gate.

## Capability hypothesis

LTM1 proved that smooth recurrent trajectory credit is learnable, but all
exchangeable lineages collapsed because every prompt supplied the same single
teacher trace. VMT1 changes the learning substrate. Every prompt supplies two
different complete autonomous trajectories generated from candidate sample
positions 0 and 1, with exactly one independently verifier-correct outcome.
The correct sample position is balanced, so neither latent branch can become
an always-correct lane.

VMT1 forms two complete sticky internal trajectories. It computes an exact
two-permutation assignment between internal trajectories and observed full
trajectory traces. The assignment is whole-lineage: no hidden field, trace
step, or response token can be matched independently. A model-owned terminal
validity head learns which matched internal trajectory corresponds to the
verified-correct outcome. At inference, the head selects one complete latent
prefix before ordinary decoding; no candidate text, verifier, gold answer,
teacher, host solver, or external model is present.

This is not ordinary DPO. DPO compares output likelihoods but contains no
persistent alternative states and no model-owned trajectory selection at
inference. It is not an LTM1 diversity repair: branch separation is supervised
by two semantically distinct observed trajectories under a bijective matching
constraint, not by a generic distance penalty.

## Frozen fit board

- Source: corrected fresh 8,192-prompt K=4 candidate bank, SHA-256
  `20a496867c1afc46d094a1ee2762cc553bd0460bd2915cc7e60d9c53025aa816`.
- Candidate positions: exactly 0 and 1.
- Every row has one correct and one wrong non-empty, non-exhausted completion.
- Four cells: math/correct-0, math/correct-1, science/correct-0,
  science/correct-1.
- Four deterministic rows per cell, 16 rows total.
- Both complete responses and the rendered prompt must fit exactly in 1,024
  tokens after reserving eight latent slots. Truncation is forbidden.
- Board seed: `2026080602`.

## Frozen architecture and objective

- Backbone: pinned `Qwen/Qwen3.5-0.8B` revision
  `2fc06364715b967f1860aea9cf38778875588b17`.
- Trainable backbone path: rank-8/alpha-16 LoRA in the final four text layers.
- Source pass is differentiable through LoRA; non-LoRA tensors remain frozen.
- Two sticky latent trajectories, width 384, eight slots, eight tied recurrent
  steps, eight attention heads, feed-forward multiplier two.
- Exact two-way permutation marginal over complete ordered trace assignments.
- Only the lineage assigned to the correct observed trajectory receives
  response-language NLL. The wrong response is a semantic trace target, not a
  language-model imitation target.
- A model-owned terminal validity head receives each complete final trajectory
  probe. It is trained with correctness BCE plus a pairwise margin.
- Loss weights: correct response NLL 1.0, trace assignment 1.0, validity 0.25,
  monotone halting 0.01. Assignment temperature 0.1; validity margin 1.0.
- Optimizer: fused AdamW, LR `2e-4`, betas `(0.9, 0.95)`, weight decay 0.01,
  cosine decay, gradient clipping 1.0.
- One seed `2026080602`, 100 updates, batch one, accumulation 16.

## Frozen pass/kill gate

After one mechanical smoke, VMT1 receives exactly one 16-row fit. It advances
only if all conditions hold:

1. all 16 source-selected correct-response NLL values improve;
2. model-owned validity selection is at least 15/16 overall and at least 7/8
   in both correct-position orientations;
3. mean matched trace cosine is at least 0.85;
4. matched trace cosine exceeds crossed trace cosine by at least 0.10;
5. final internal trajectory cosine is at most 0.95;
6. every tensor and gradient remains finite;
7. every non-LoRA backbone tensor remains hash-identical;
8. swapping validity scores across the two coherent prefixes reduces fit
   selection by at least 25 points.

Failure closes VMT1 without another seed, branch count, width, depth, trace
representation, loss weight, margin, duration, layer count, or schedule.
Passing only authorizes a separate matched broad contract; it is not itself a
reasoning or benchmark result.
