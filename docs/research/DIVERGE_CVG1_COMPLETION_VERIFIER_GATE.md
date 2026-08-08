# DIVERGE-CVG1: Completion-Verified Whole-Lineage Gating

Status: conditional successor, frozen before implementation or score. Launch
only if the exact SAG1 prompt-only gate closes.

## Measured Motivation

QPT1 is not a uniformly stronger or weaker model than B1. On the already-open
538-example development board, a static oracle that may select one complete
B1 or QPT1 answer reaches `73.798%` five-domain macro and `366/538` solved,
versus QPT1 `62.588% / 317` and B1 `55.630% / 248`. The union retains `34/40`
code answers and reaches GSM8K `96/100`, MATH-500 `64/100`, GPQA `97/198`,
and BBH logic `75/100`. This oracle is diagnostic, not a model result.

The observed bottleneck is therefore arbitration after a complete trajectory,
not absence of complementary expert capability. SAG1 tests prompt-only
arbitration. CVG1 is structurally different: it evaluates completed candidate
trajectories before committing.

## Architecture Hypothesis

1. Keep the qualified B1 and pointer-transaction expert as frozen, coherent
   whole lineages.
2. Generate one complete candidate from each lineage under identical decoding.
3. A small model-owned critic reads `prompt + candidate completion` and emits
   calibrated probability of task success plus an abstention risk.
4. Commit to exactly one whole candidate. Never average logits, tokens, or
   internal fields across candidates.
5. If critic evidence is insufficient, choose B1 exactly.

The critic receives no evaluator task name, benchmark identity, gold answer,
external solver output, or execution result at inference. It is trained only
on source-disjoint rollout pairs whose outcomes can be established during
training by exact math/code/logic/science assessors. Evaluation labels remain
sealed.

## Fast Gate

- Freeze both generators before rollout collection.
- Build a balanced outcome corpus containing base-only wins, expert-only wins,
  both-correct, and both-wrong pairs; split by source and normalized prompt
  identity before training.
- Compare against always-B1, always-expert, random disagreement choice,
  teacher-forced-loss choice, and SAG1 prompt-only routing.
- Require on the unchanged development board: at least `30/40` code, at least
  +3 macro points and +15 solved over the strongest frozen single lineage,
  gains in at least three domains, and no domain regression over two points.
- Require calibration and selection accuracy on a source-disjoint rollout
  holdout before opening development evaluation.
- A development pass opens one larger source-disjoint confirmation. A miss
  closes exact CVG1 without critic-width, threshold, seed, duration, or
  benchmark-label variants.

This lane tests whether model-owned post-trajectory verification can recover
the measured 49-answer headroom beyond QPT1. It does not claim the static
oracle score as achieved capability.
