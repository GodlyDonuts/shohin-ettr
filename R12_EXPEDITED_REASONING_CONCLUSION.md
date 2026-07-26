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
6. Frozen Shohin residuals do not supply renderer-invariant role semantics to
   the source-deleted multi-family compiler. The connected smoke exactly ties
   the standalone compiler at 12/24 development cases and remains 0/6 on both
   held-out-renderer cells.

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

The final multi-family smoke sharpens that diagnosis. Both a 152,933-parameter
byte compiler and a 331,589-parameter compiler connected to frozen Shohin
blocks 17, 25, and 29 fit all 36 training episodes and solve all 12
unseen-law/longer-composition probes. Both fail all 12 probes that require a
new renderer. Frozen language-model features therefore add no exact transfer
on the decisive boundary. This is a representation-learning failure before it
is an execution-depth or parameter-capacity failure.

## Final Multi-Family Evidence

Newton job `702764` completed in 3 minutes 26 seconds on one H100:

| System | Fit | Unseen law | Longer composition | Renderer | Joint | Development |
|---|---:|---:|---:|---:|---:|---:|
| Standalone byte compiler | 36/36 | 6/6 | 6/6 | 0/6 | 0/6 | 12/24 |
| Frozen-Shohin connected compiler | 36/36 | 6/6 | 6/6 | 0/6 | 0/6 | 12/24 |

The connected system has 331,589 learned compiler parameters and 125,413,253
conceptual complete parameters. It uses a 1,728-wide frozen residual from
three Shohin blocks and makes zero candidate-time oracle, search, or verifier
calls. Its report SHA-256 is
`06345448d5a1696678b5eb439a3ca414b26132cbbf9ee6782af68f0606e9688d`.

The report is negative-only evidence, not a claim-bearing qualification. Its
checkpoint bytes and reported parameter count match the protected checkpoint,
and loading hard-fails on a wrong SHA-256, but the stricter runtime-semantic
receipt remained false in the CUDA execution environment. Three fail-closed
repeats (`702769`, `702771`, and `702773`) emitted no replacement report.
This audit limitation can invalidate a positive claim; it cannot turn the
observed exact tie and zero renderer transfer into evidence of success.

## Highest-Value Next Step

Do not spend the next budget on a larger controller. The highest-value next
step is a frozen **representation curriculum** that trains renderer invariance
and episode-local role binding before testing recurrent execution. It should:

1. retain the frozen 1,344-row audit board and source-deletion boundary;
2. add many training renderers while keeping one structurally distinct
   renderer fully held out;
3. supervise role equivalence across counterfactual renderings of the same
   anonymous machine;
4. train one shared encoder/compiler across all three families;
5. measure compiler sealing separately from recurrent execution;
6. retain rule-binding-shuffled, renderer-shuffled, recurrence-disabled,
   equal-compute, and random-label controls; and
7. require at least 85% exactness in every family/cell before any reasoning
   claim or continuation pretraining decision.

If that curriculum cannot transfer the held-out renderer, further recurrence,
memory, search distillation, or parameter growth is not justified. The
protected pretraining hold remains in force, and there are zero active Newton
jobs at this closeout.

## Post-Closeout Positive Result

The authorized representation curriculum has now passed its bounded
qualification. Renderer-neutral incidence typing plus counterfactual
target-first supervision reaches 120/120 across five seeds and all three
leave-one-family-out folds, versus 65/120 for the equal-budget
direction-shuffled control. Every treatment fold scores 8/8 and all 15
treatment-control directions are positive.

This revises one conclusion: reusable transfer is possible on the anonymous
finite-machine board when representation and execution are factorized
correctly. It does not revise the general-reasoning conclusion. The mechanism
still receives complete transition tables, exploits fixed incidence geometry,
shares one machine ontology across families, and is not integrated into the
Shohin trunk. Full evidence and scope:
`R12_RENDERER_CURRICULUM_QUALIFICATION.md`.

## Final Variable-Topology Boundary

The learned global semantic-partition compiler now passes the variable-
topology gate. Across five independently generated boards and all three
leave-one-family-out folds, treatment reaches 360/360. The same treatment
weights reach 152/360 when source direction is swapped and 0/360 when either
the episode-global key scores are negated or query roles are swapped. Every
paired direction, collision cell, and joint cell passes the independent
audit.

This revises the bounded capability conclusion:

`global_semantic_partition_passes_variable_topology_gate`

The 60,613-parameter sidecar has learned to compile unseen complete anonymous
machines across variable cardinality, action count, topology, renderer,
composition, and family. It makes zero candidate-time oracle/search/verifier
calls. Audit SHA-256 is
`049bbbd398e6f1456d6f1809bebb701351053302e0ce44147e70735d5b155fa2`.

The general-reasoning conclusion remains negative. Every episode still
provides a complete transition table, all tasks share one finite-machine
ontology, execution is a fixed discrete operator, process-level source
deletion is not attested, and the compiler is not integrated into Shohin.
The next justified gate is sparse latent-law induction, not more optimization
of this solved board. Full evidence:
`R12_VARIABLE_TOPOLOGY_SEMANTIC_COMPILER_RESULT.md`.

## Final Sparse-Law Boundary

The sparse successor gate is now closed under three distinct formulations:

| Candidate | Transition accuracy | Complete maps | Exact queries |
|---|---:|---:|---:|
| Direct set attention | 46.5000% | 0/60 | 4/60 |
| Learned generator factorization | 15.7083% | 0/60 | 1/60 |
| Supervised internal microcode | 22.1250% | 0/60 | 0/60 |

The microcoded model contains a fixed internal ALU and receives exact
preparation-time family/parameter labels, yet its learned controller reaches
only 1/204 exact unseen programs. Observation-shift and observation-zero
controls collapse, so it uses the demonstrations; it simply does not infer a
transferable identification procedure. Internal execution machinery is
therefore not the missing sufficient mechanism.

This makes the expedited conclusion final for the current budget: Shohin has
bounded systematic finite-machine compilation, but no demonstrated native
general reasoning or sparse unseen-law induction. Do not launch another
proxy-specific architecture branch. Preserve the protected checkpoint and
the user pretraining hold. Full result:
`R12_SPARSE_LAW_MICROCODE_RESULT.md`.

## Final Adversarial Revision: Episode-Local Solver

The apparent positive revision is rejected as a Shohin reasoning claim.

Hardened H100 job `704792` reaches 11/11 exact development queries and 11/11
complete target maps, including 2/2 episodes from a completely held-out
random-permutation generator family. Record-order reversal and consistent
support recoding remain 11/11. Deleted necessary witnesses, zeroed
observations, and deranged support semantics each recover 0/11 exact queries;
witness deletion and zeroed observations reject every packet. There is zero
training/development target-law or raw-map overlap.

The actual learned component is a 232,065-parameter record-direction reader.
Shohin is never loaded. Exact regex parsing, exhaustive 127-word enumeration,
softmax matching, packet sealing, and Python execution are host machinery.
Fourteen of 22 development target-word instances overlap training, and source
deletion is not process-level. Report SHA-256 remains
`226a36d9156101617b769f698550eb51ebec57a8ffa01464bdd7a64d8805caad`.

Revised decision:

`reject_architecture_native_shohin_reasoning_retain_neurosymbolic_solver`

The system is useful as a bounded neuro-symbolic control, not as the best
Shohin reasoning baseline. The general-reasoning conclusion remains negative.
Keep pretraining held. The next claim-bearing experiment must connect the
actual Shohin trunk, remove semantic host parsing/search/execution, enforce
process-level deletion, and transfer across genuinely different ontologies.
Full audit: `R12_EPISODIC_GENERATOR_ADVERSARIAL_AUDIT.md`.
