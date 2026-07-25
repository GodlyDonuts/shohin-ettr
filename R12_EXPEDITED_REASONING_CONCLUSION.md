# R12 Expedited Reasoning Conclusion

## Decision

As of 2026-07-25, Shohin does **not** demonstrate native general reasoning.
The architecture campaign found real learned competence and two useful
mechanics, but neither transfers across the rule, composition, renderer, and
task-family boundaries required by `R12_GENERAL_REASONING_GATE.md`.

This is a completed scientific conclusion, not a claim that further progress
is impossible. It closes open-ended architecture search under the current
evidence and usage budget.

## Protected Base

- checkpoint: `train/flagship_out/ckpt_0300000.pt`
- parameters: 125,081,664
- SHA-256:
  `211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6`
- pretraining: held by explicit user instruction
- active Newton jobs at closeout: zero

## What Is Demonstrated

1. **Local learned competence exists.** Random-label controls repeatedly
   collapse while lawful training produces nonzero held-out performance.
2. **Full-trajectory temporal exposure helps.** The matched recurrent
   classifier reaches 1,058/2,048 = 51.6602% on unseen larger instances of
   the matrix-reduction proxy. The best individual seed reaches 332/512 =
   64.8438%.
3. **Exact state identity is causally useful.** A frozen exact anti-revisit
   barrier improves 820/2,048 to 957/2,048 and removes 89.4% of cycle events.
4. **The useful effects are architecture-native at inference.** Final
   candidate rollouts use no oracle, search, or verifier.

## What Is Not Demonstrated

1. No retained mechanism has passed unseen-rule and unseen-task-family
   transfer.
2. Learned episodic semantics are not established. Feature-shuffling the
   learned memory preserves its accuracy, while a semantic-similarity barrier
   is harmful.
3. Exact anti-revisit is not a learned reasoning law. It is a fixed,
   task-specific execution constraint.
4. Full-trajectory recurrence has not beaten a generic task-specific
   classifier interpretation.
5. The raw 300k language model remains at low-single-digit public reasoning
   accuracy: GSM8K maj@4 4/100, GSM8K pass@1 2/100, MATH-500 2/100,
   HumanEval 6/164, and MBPP 0/100.

## Mechanisms Rejected

The following hypotheses failed their matched causal controls or absolute
gates:

- wider soft value iteration;
- repeated value propagation;
- raw one-step successor exposure;
- on-policy DAgger;
- scaled search distillation;
- Lyapunov/Bellman scalar potentials;
- proof-carrying local contracts;
- learned similarity-based episodic memory;
- semantic similarity cycle barriers; and
- barrier-aware exact-memory training as a superior policy.

External search and host-computed schedules solve selected proxies, but they
are counted external algorithms and are not Shohin-native reasoning.

## Retained Baseline

The only justified learned baseline is the full-trajectory recurrent
classifier:

- added parameters: 10,132,198
- complete system: 135,213,862 parameters
- aggregate strict score: 1,058/2,048 = 51.6602%
- best seed: 332/512 = 64.8438%
- preserved model:
  `artifacts/r12/ssqac_episodic_anti_cycle/best_full_trajectory_classifier_seed20260910.pt`
- model SHA-256:
  `a7ebd0a0487d9fa75318faa5b5693a48439b799895cc4e88277be5c666ad5d1e`

Exact discrete anti-revisit may remain an optional execution primitive, but
it is not part of the best learned baseline.

## Conclusion

Shohin has progressed from near-zero raw reasoning to material, autonomous,
task-specific controller competence. It has **not** crossed the boundary from
competence to general reasoning.

The bottleneck is not parameter count or GPU utilization. It is systematic
law acquisition and composition: the model can learn local action policies,
but the tested mechanisms do not induce a reusable algorithm that transfers
to new rule systems.

## Only Admissible Next Gate

No further architecture should be promoted, and no continuation pretraining
should begin, until one frozen experiment tests the retained recurrent
mechanism with all of the following:

1. at least three genuinely different task families;
2. train/test separation by rule or law, not just instance geometry;
3. unseen compositions and execution depths;
4. opaque-name and renderer holdouts;
5. source deletion before candidate execution;
6. one shared mechanism with no family-specific executor or head;
7. recurrence-disabled, rule-binding-shuffled, equal-compute classifier, and
   random-label controls; and
8. a preregistered minimum of 85% per family plus a material causal margin.

Anything weaker would produce another proxy result rather than answer whether
Shohin reasons.
