# DIVERGE-VMT1: Verified Multi-Trajectory Matching

Status: closed negative after the one frozen fit.

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

## Result

CPU board job `743309` completed in 26 seconds. From 32,768 corrected
candidate rows it found 369 structurally admissible opposite-outcome pairs and
208 exact tokenizer-admissible pairs. The selected board has four rows per
group/orientation cell, maximum total length 1,010 of 1,024, and zero
truncation. Board/report SHA-256 values are
`4e5677e00bcf3c1fd72cff11d36a994ec949c9ce658edaedd43676ee8754f685` /
`aac1fd11b2ea2207b6a015eac511bc31b3cc732780d4f58a1ae8f4a7296d5ae7`.

Mechanical H100 smoke `743310` completed cleanly on `evc32`: all ten on-node
tests passed, all 473 pinned Qwen tensors loaded, two finite updates ran, and
the frozen hash was unchanged. It used 4,610,052 trainable parameters and
17,659,465,216 peak allocated CUDA bytes. Smoke report/checkpoint SHA-256
values are `cb4c715c7d1a753e768143e4c68594bd94ced99197afdda2c277cf7947d9c8db` /
`20aac48be82fced5457bb2f27b5067ac8e96b99163104937d1c44a7fbc38dad8`.

The sole fit `743311` completed all 100 updates and both exact audits in
1,606.677 seconds. It consumed 658,200 logical correct-response tokens,
1,316,400 candidate tokens, and 1,138,900 trace-target tokens at 409.665
logical tokens/s, peaking at 19,281,572,352 allocated CUDA bytes. All tensors
remain finite and every non-LoRA parameter is hash-identical.

Language and average trace fit succeed, but coherent alternatives do not:

- token-weighted selected NLL improves `1.084766 -> 0.183080`, with 16/16
  rows improved;
- matched trace cosine reaches 0.867081, above the 0.85 threshold;
- crossed trace cosine also reaches 0.866763, leaving only 0.000318 advantage
  versus the required 0.10;
- internal trajectory cosine rises `0.807822 -> 0.998769`, failing the 0.95
  ceiling;
- model-owned selection reaches 11/16, split 7/8 versus 4/8 across the two
  balanced observed-correct orientations, below the 15/16 and per-orientation
  gates;
- swapped selection reaches 5/16, so the swap drop passes, but it acts on a
  weak asymmetric selector rather than two separated semantic lineages.

The failure has an exact local explanation. When both internal trajectories
coincide, the identity and swapped assignments have equal cost and posterior
0.5. The expected validity gradient is exactly zero, every element of the
2-by-2 trace-cost matrix receives the same 0.25 gradient, and both language
NLLs receive the same 0.5 gradient. The collapsed barycenter is therefore a
symmetric stationary point of the frozen objective. The distinct response
texts are not the problem: all 16 selected pairs have different final
predictions, median character-prefix overlap is 0.65%, and median word-set
Jaccard overlap is 22.05%.

VMT1 is closed. There is no hard-matching, temperature, width, depth, seed,
loss, duration, or trace-target repair. A next mechanism must remove
exchangeable semantic roles rather than merely repel branches. The ordered
candidate data support a materially different hypothesis: a model-owned first
draft followed by a prompt-conditioned verified correction trajectory, with
correct-draft no-op cases and wrong-draft-to-correct targets. That successor
must be compared with an ordinary two-pass correction baseline and must close
the teacher-draft/autonomous-draft gap before any broad claim.

Fit report/checkpoint SHA-256 values are
`058ac42381dbd9d023a5e6bc716476b60716b871add6ed55dd36de7eb888ab9b` /
`a6ca16175804b8346ad0c8906f1cca3b0a587fb248386963cedbda4d02b50741`.
