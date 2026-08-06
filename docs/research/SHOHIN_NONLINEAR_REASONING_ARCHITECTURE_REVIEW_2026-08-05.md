# Changing How Shohin Thinks

## Literature review and architecture recommendation — 2026-08-05

## Executive conclusion

The promising research question is not whether a language model should produce
more chain-of-thought tokens. It is whether one model can maintain several
mutually incompatible, persistent hypotheses; improve them through recurrent
computation; actively search for evidence that separates them; and commit to
one coherent hypothesis without averaging incompatible pieces.

The recommended mechanism is a **Falsification-Coupled Particle Transformer
(FCPT)**. FCPT is a native latent deliberation layer containing a small set of
complete reasoning particles. Each particle carries an entire candidate world
state and controller state. Shared recurrent weights update all particles.
Particles challenge one another through a bandwidth-limited contradiction bus,
receive evidence-conditioned scores, and undergo whole-particle selection or
resampling. The model never averages fields from different candidate programs.

This recommendation follows Shohin's strongest negative result. Its existing
parallel schedulers can learn useful local fields, but different seeds discover
incompatible complete programs. Fieldwise averaging produced a causally inert
"Frankenstein" schedule. The next architecture should therefore preserve
**whole-hypothesis identity**, not merely add width, recurrence, a larger MLP,
or more supervision.

FCPT is not clean-room invention. It combines ideas from recurrent-depth
transformers, latent reasoning, differentiable particle filters, modular/global
workspace networks, object slots, adaptive computation, and counterexample-
guided synthesis. The potentially novel contribution is their exact
combination inside a language model:

1. persistent structured hypothesis particles rather than independent decoded
   text samples;
2. endogenous pairwise falsification rather than passive diversity;
3. whole-particle resampling rather than soft state aggregation;
4. behaviorally certified merging rather than coordinate similarity; and
5. ordinary-text self-supervision plus sealed compositional evaluation.

That novelty claim must remain a hypothesis until a broader search and matched
experiments establish it.

## Execution addendum — 2026-08-05

This review preceded the completed PCSD and FCPT pilots. Do not use its
immediate recommendation to relaunch either lane.

- PCSD reached `19.092% / 14.038%` exact answers at depths 8/12 versus
  `20.996% / 14.209%` for the matched dense arm. It enforced its learned
  invariant almost perfectly, but exact terminal state remained zero, causal
  projection ablations cost less than one point, and throughput was 3.36x
  worse. PCSD is closed as a standalone architecture.
- FCPT v1 reached `12.174%` macro versus `11.475%` for whole-particle
  selection. A corrected coverage objective increased behavioral uniqueness
  from roughly 1.2--1.5 to 3.06--4.28 of eight candidates, but capability
  remained `12.093%` versus `11.800%`, lost on two cohorts, and missed the
  fixed pilot gate. FCPT is closed without a full matrix or confirmation run.
- The successor Prompt-Selected Presented Algebra (PSPA) changes the
  hypothesis class rather than adding another particle or projection
  objective. Its repaired mechanics pilot reaches 100% exact answers and
  complete-presentation recovery on six depth-shift cohorts, versus `9.961%`
  tied recurrence and `10.840%` Transformer controls. Shuffled challenge
  outcomes reduce it to `52.507%`; whole-lineage swaps reduce it to `13.574%`.

PSPA is not yet a learned language architecture. Its current source compiler
enumerates a small structured presentation family. The result authorizes only
a learned language-to-presentation compiler and a parameter/FLOP-matched
gate. It does not authorize long pretraining or a general-reasoning claim.

Later execution found the missing optimization boundary. Joint Sinkhorn
compilation failed (`9.147%`), but learning local evidence first and imposing
whole-presentation closure once at commitment reached `58.643%`. That closure
still ignored counterexamples. Adding explicit counterexample selection over
the confidence-derived complete presentations produced CSDC: `99.577%`
development and `99.723%` unchanged-weight confirmation, with large predicted
failures under shuffled challenge outcomes and whole-lineage swaps. CSDC now
replaces FCPT as the protected synthetic architecture baseline. Its remaining
boundary is typed source challenges rather than ordinary-language semantic
compilation.

The bounded semantic bridge then trained a 75,912-parameter source-only
parser while freezing CSDC. End-to-end exact answers reach `95.573%` on
development and `93.896%` under an unseen challenge field order, versus
`99.593% / 99.723%` with typed source fields. Shuffling parsed outcomes cuts
the scores to `57.503% / 54.671%`, and swapping committed lineages cuts them
to `13.525% / 13.167%`, so the learned path is causally used. The frozen gate
still fails: complete ordered challenge words reach only `89.290% / 84.153%`
and selected presentations `90.706% / 86.377%`. This parser is closed without
variants. The result localizes the remaining controlled-language loss to
ordered sequence compilation rather than counterexample selection or late
execution; it does not reopen PCSD or FCPT.

The materially different role-gated copy interface then removes that loss.
Instead of regenerating a word from one record summary, it predicts source
token roles and copies model-selected state/operator tokens in source order.
With fewer parameters and the same 192,000 examples, frozen CSDC reaches
`99.593% / 99.723%` development/held-renderer answers, exactly matching its
typed oracle. Complete challenge tuples are 100%, selected presentations are
`99.007% / 99.284%`, shuffled outcomes score `53.630%`, and lineage swaps
score `13.623% / 13.346%`. This becomes the protected controlled
rendered-source baseline. It validates preservation of source identity at the
semantic-symbolic boundary; it is not yet an unrestricted language claim and
still does not reopen PCSD or FCPT.

The subsequent SmolLM2 lexical integration reaches `99.691% / 95.915%`
development/combined-shift answers and retains large shuffled-evidence and
lineage-swap failures, but all-eight shifted challenge tuples are exact in only
`17.920%` of episodes. Its first-subtoken copy interface is closed. The one
frozen whole-mention span-quotient successor also closes. It reaches 99.691%
development answers, 100% complete tuples, and 99.202% selected tables, but
under lexical shift falls to `84.294% / 17.920% / 74.447%`. Exact shifted gold
mentions are 83.021%; duplicate and missing role assignments dominate despite
100% tokenizer representability. Causal challenge/lineage interventions still
collapse and no nonexact span is accepted. Whole-span pooling plus exact-
surface occurrence messages therefore do not solve unseen nominal grounding
or coherent record-level role assignment. No repair variant follows.

After that one gate, the ordered successor is **DIVERGE**, not another PCSD or
FCPT run. DIVERGE extends the protected CSDC principle from a small explicit
Cartesian candidate set to a source-sealed factorized version-space packet:
shared typed graph/state, episode-local fault-line variables, guarded patches,
hard constraints, calibrated support, provenance, verifier-checked nogoods,
safe equivalence merges, and query-invariant commitment or abstention. Its
first deliverable is an exact CPU reference and delayed-disambiguation board,
not neural scaling. Full particles remain a matched control. The bounded
candidate contribution is the learned source-sealed conjunction and sharing
advantage, not particles, BDDs, conflict clauses, or typed state individually.

DIVERGE's first learned source-boundary result is now available. Whole-option
pooling fails, but a materially different token-role/source-copy compiler
reaches exact packet construction and delayed recovery on development and
held renderer/ontology confirmation for five of five seeds. The result
qualifies the compiler only. The full resource-matched A--G matrix, late-query
types, causal interventions, and sharing receipts remain required before any
architecture claim.

The subsequent balanced, source-sealed A--G CPU gate initially appeared to
pass all five compiler seeds, but that V3 result is not accepted: its executor
materialized one state per world while charging only static packet bytes. The
resource-corrected V4 bitset/state-group runtime keeps DIVERGE at 100% over
2,160 late queries, but the fair whole-particle control rises to 72.222% and
the sharing ratio at four worlds is only `1.893x`, below the frozen `2x` gate.
Broad DIVERGE promotion is therefore negative. The retained result is narrower:
exact delayed recovery and increasing factorized savings at eight or more
worlds, reaching `27.365x` storage savings at 64 worlds.

The qualified token-role source compiler also contained a hidden scaffold: it
received each gold record and option in a separate model call. DIVERGE-SC1
tested the missing boundary with one unsegmented raw-source pass, token-role,
gap-boundary, and pairwise binding scores, and one complete-object decoder.
Its exact CPU mechanics pass 1,000/1,000 episodes, but its one frozen neural
seed is a decisive negative. Local role/boundary/pair accuracies reach
`99.999994% / 100% / 98.381054%`, yet exact packets are zero in train and all
three shifts, support recall is at most 0.781%, and 32.422--58.984% of episodes
overflow. Follow-on seeds were canceled. One read-only component substitution
audit may localize this combinatorial amplification; it cannot reopen SC1.
DIVERGE therefore still lacks a learned autonomous raw-source front end and
has no unrestricted language claim.

That read-only localization is complete. The learned boundary is exactly
correct over 128 audited episodes, and keeping it while replacing role and
pair factors restores 100% packets in every cohort. Either learned roles or
learned pairs remain independently fatal. Pair precision is only 37.162% and
recall 78.809%; role activation recall is 98.873%, with the remaining errors
concentrated in record-kind cues and alias starts. These errors double option
proposals and multiply complete-record proposals by 7.2--9.4x. The failure is
therefore not a lack of downstream DIVERGE capacity. It is the use of
independent edge labels to construct a conjunctive object. A future source
compiler, if separately authorized, must emit a bounded globally normalized
whole-record assignment with exact-one constraints rather than repair SC1's
thresholded pair graph.

## Terminology

Do not call this "schizophrenic AI" in code, a paper, a resume, or a public
description. Schizophrenia is a serious clinical condition and is not the same
as multiple personalities. The technically accurate concepts are **plural
inference**, **competing hypotheses**, **particle deliberation**, and
**dialectical reasoning**.

## 1. What a standard autoregressive Transformer is doing

A decoder-only Transformer maps a fixed prefix through a fixed number of
layers, then commits to one next-token distribution. It is parallel across
token positions during training, but its output-time computation is causal and
left-to-right. Once a reasoning trace emits an early token, later computation
can condition on it but cannot revise it in place.

The limitation is not that a Transformer is mathematically incapable of
algorithms. Transformers can implement rich sequence programs, and recurrent
or sufficiently deep variants are computationally powerful. The practical
problems are inductive bias and optimization:

- fixed depth gives every problem essentially the same latent compute;
- next-token loss rewards plausible continuations, not globally coherent
  latent programs;
- one residual stream can superpose incompatible explanations;
- early autoregressive commitments create path dependence;
- local field accuracy does not guarantee a coherent complete program; and
- uncertainty is commonly represented as a diffuse vector or token
  distribution rather than persistent, separately testable hypotheses.

Shohin's experiments sharpen this diagnosis. The model often recovers factual
or local schedule information, but causal composition is seed-sensitive.
Averaging four learned schedule distributions preserved factual accuracy while
driving strict WORLD and COMMAND causality to zero. This is evidence against
the assumption that the seeds are noisy estimates of one unimodal solution.
They are better interpreted as different complete-program basins.

## 2. Literature map

### 2.1 Recurrent depth and adaptive computation

The [Universal Transformer](https://arxiv.org/abs/1807.03819) applies shared
transformations recurrently and adds dynamic per-position halting. [Adaptive
Computation Time](https://arxiv.org/abs/1603.08983) and
[PonderNet](https://arxiv.org/abs/2107.05407) learn how many computational
steps to take. Recent work has made this direction substantially stronger:

- [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) combines a shared
  recurrent stack with token-level depth routing.
- [Scaling up Test-Time Compute with Latent Reasoning](https://arxiv.org/abs/2502.05171)
  trains a recurrent-depth language model and scales latent iterations at
  inference.
- [The Recurrent Transformer](https://arxiv.org/abs/2604.21215) introduces
  layerwise recurrent memory and reports parameter-matched language-modeling
  gains at 150M and 300M parameters.
- [Loop, Think, & Generalize](https://arxiv.org/abs/2604.07822) finds that
  recurrent depth can improve systematic and depth generalization, while also
  documenting overthinking when recurrence is excessive.
- [Hierarchical Reasoning Model](https://arxiv.org/abs/2506.21734) and
  [Tiny Recursive Model](https://arxiv.org/abs/2510.04871) show that small
  recurrent systems can perform strongly on structured puzzles.
- [Universal Transformers Need Memory](https://arxiv.org/abs/2604.21999)
  reports that memory slots are necessary in its Sudoku setting and that ACT
  initialization can trap models in a shallow-halting equilibrium.

**Implication:** adding a loop, adaptive halt, hierarchical timescale, or
weight tying is worthwhile as a control, but is no longer a credible novelty
claim by itself. FCPT should use recurrent depth as infrastructure, not as its
headline contribution.

### 2.2 Latent reasoning instead of verbal chain of thought

[Coconut](https://arxiv.org/abs/2412.06769) feeds a final hidden state back as
a continuous thought and reports that a continuous state can encode multiple
possible next reasoning steps. [Quiet-STaR](https://arxiv.org/abs/2403.09629)
learns tokenwise internal rationales from ordinary text and improves zero-shot
reasoning after continued pretraining. [Large Concept Models](https://arxiv.org/abs/2412.08821)
move prediction from tokens to sentence-level semantic representations.

**Implication:** "reason silently in latent space" is established prior art.
A Shohin contribution must specify what is represented, how alternatives
remain distinct, how they interact, and why the update avoids collapse.

### 2.3 Parallel reasoning paths

Multiple-path reasoning is also crowded:

- [Self-Consistency](https://arxiv.org/abs/2203.11171) samples several textual
  chains and marginalizes their answers.
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601) explicitly searches,
  evaluates, and backtracks over textual reasoning branches.
- [Mixture-of-Agents](https://arxiv.org/abs/2406.04692) layers multiple LLM
  agents and aggregates their responses.
- [Parallel Latent Reasoning](https://arxiv.org/abs/2601.03153) creates
  multiple continuous reasoning streams with diversity regularization and
  learned aggregation.
- [MPCoT](https://arxiv.org/abs/2606.06245) initializes several latent
  hypotheses, refines them recurrently, reward-scores them, and softly
  aggregates before action decoding.
- [Guided stochastic exploration for recursive models](https://arxiv.org/abs/2605.25230)
  treats deterministic recurrence as a one-particle limit, perturbs latent
  trajectories, and reweights them with the model's guide.

**Implication:** "several thoughts at once" is not novel. The open opportunity
is to preserve complete candidate identity and use disagreement to generate
falsification pressure. In particular, FCPT must not use the soft weighted
mean of candidate program fields, because Shohin has already measured that
failure.

### 2.4 Non-autoregressive global revision

Diffusion language models provide a second way to escape irreversible
left-to-right thought. [Beyond Autoregression](https://arxiv.org/abs/2410.14157)
uses discrete diffusion for planning and structured reasoning.
[LaDiR](https://arxiv.org/abs/2510.04573) denoises blocks of latent thought
with bidirectional attention, enabling holistic revision. A theoretical study
of [masked diffusion reasoning](https://arxiv.org/abs/2510.13117) connects
masked diffusion models to padded looped Transformers and identifies tasks
where parallelism is more efficient than chain of thought.

**Implication:** diffusion is a serious alternative control. It enables global
revision, but it does not inherently keep semantically distinct hypotheses
separate. Rebuilding Shohin as a diffusion language model would also confound
the reasoning mechanism with a wholesale training-objective change.

### 2.5 Persistent objects, modules, and shared workspaces

[Slot Attention](https://arxiv.org/abs/2006.15055) iteratively binds inputs to
exchangeable slots through competition. [Recurrent Independent
Mechanisms](https://arxiv.org/abs/1909.10893) use mostly independent recurrent
modules that communicate sparsely. A [Shared Global Workspace](https://arxiv.org/abs/2103.01197)
uses a capacity-limited communication channel through which specialist
modules compete and coordinate. [Neural Turing Machines](https://arxiv.org/abs/1410.5401)
couple a controller to differentiable external memory.

**Implication:** modules, slots, memory, and a blackboard all have strong prior
art. FCPT should borrow persistent identity and limited communication, but its
particles represent mutually exclusive *whole explanations*, not object parts
or task-specialist experts.

### 2.6 Particle inference and active falsification

[Particle Filter Networks](https://arxiv.org/abs/1805.08975) integrate
weighted particles, learned transition/observation models, and differentiable
resampling into one end-to-end network. Particle methods preserve multimodal
beliefs instead of reducing them prematurely to a mean.

[Program synthesis by sketching](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-176.html)
and later counterexample-guided inductive synthesis alternate candidate
construction with verification. [CounterExample Guided Neural
Synthesis](https://arxiv.org/abs/2001.09245) combines neural proposals with
formal counterexamples. Recent [counterexample-guided learning with reasoning
agents](https://arxiv.org/abs/2606.11521) shows that verifier-produced
counterexamples can materially improve difficult regex induction.

**Implication:** particles and counterexample loops are individually mature.
The research gap is an end-to-end language-model layer in which latent
reasoning particles challenge one another and the network learns which
observable disagreement to test, without an external solver at claim time.

### 2.7 Equilibrium and attractor views

[Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) solve directly for
a fixed point of a weight-tied network. [Modern Hopfield
Networks](https://arxiv.org/abs/2008.02217) connect attention to associative
memory and characterize global, metastable, and single-pattern fixed points.

**Implication:** convergence to an attractor is not enough. If the objective
has an averaged metastable basin, convergence can make the wrong mixture more
stable. FCPT therefore needs explicit particle identity and a commitment rule,
not merely an energy-minimizing shared state.

## 3. Candidate directions and decision

| Direction | Potential | Novelty risk | Shohin fit | Decision |
|---|---:|---:|---:|---|
| More recurrent depth / ACT | High | Very high | High | Required control, not headline |
| HRM/TRM-style timescales | High on puzzles | High | Medium | Control only |
| Coconut-style continuous thought | Medium-high | High | High | Control only |
| Diffusion latent planner | High | Medium-high | Low-medium | Strong alternative, too confounded for first test |
| Multi-agent textual debate | Medium | Very high | Low | Reject as architecture claim |
| Parallel latent streams with soft averaging | Medium-high | High | High | Reject: repeats measured Frankenstein failure |
| Persistent whole particles with endogenous falsification | High | Medium | Very high | Recommended |

### 3.1 Concurrent PCSD hypothesis

A concurrent agent has implemented a bounded **Prompt-Conditioned Syndrome
Dynamics (PCSD)** falsifier. PCSD compiles linear parity checks from the prompt
and projects every recurrent state update back onto the resulting invariant
manifold. It is a sensible test of whether composition failure is accumulated
state drift.

Its prior-art boundary is tighter than it first appears. [Differentiable
Projection for Constrained Deep Learning](https://arxiv.org/abs/2111.10785)
already inserts differentiable constraint projections into neural networks.
[Deep Conservation](https://arxiv.org/abs/1909.09754) enforces conservation
laws in learned latent dynamics, and
[ConCerNet](https://arxiv.org/abs/2302.05783) learns unknown invariants and then
uses a neural projection layer to preserve them. PCSD's possible separator is
that the constraints are newly compiled from each language prompt and applied
to a reasoning trajectory, rather than fixed physical laws or dataset-level
invariants.

PCSD should complete its already-bounded synthetic falsifier, but should not
become the sole architecture program unless it transfers beyond
conservation-shaped tasks. General reasoning legitimately changes many facts,
so a sticky linear syndrome can either suppress necessary computation or learn
vacuous checks. The strongest long-term synthesis is to treat PCSD as an
optional **within-particle stabilizer** inside FCPT: each candidate preserves
its own inferred invariants while particles represent different complete
interpretations of what those invariants and programs should be.

## 4. Proposed architecture: Falsification-Coupled Particle Transformer

### 4.1 State

Given encoded context `H`, initialize `K` exchangeable particles:

```text
P_i^0 = (S_i^0, M_i^0, log_w_i^0, lineage_i),  i = 1...K
```

- `S_i` is a complete candidate ETTR typed state, not one field or one token.
- `M_i` is private recurrent controller memory.
- `log_w_i` is evidence-conditioned credibility.
- `lineage_i` records whole-particle ancestry through resampling.

Use one shared compiler and one shared reactor. Learned particle seeds break
symmetry, but particle-index permutation must only permute outputs. No particle
has a permanent semantic role.

### 4.2 Proposal step

Each particle independently proposes a complete next state with shared weights:

```text
(S_tilde_i, M_tilde_i) = T_theta(S_i, M_i, H_command, t)
```

The proposal can use ETTR's hard typed transactions or a continuous relaxation
whose deployment projection is exact. Recurrent depth supplies additional
compute without additional parameters.

### 4.3 Contradiction bus

For each selected pair `(i,j)`, a shared falsifier receives only their
structured disagreement and emits a bounded probe:

```text
C_ij = F_phi(delta(S_tilde_i, S_tilde_j), summary(H))
```

The probe is not a free-form natural-language critique. It identifies a
specific relation, value, transition, predicted consequence, or masked
evidence unit on which the particles disagree. A capacity-limited global bus
broadcasts the highest-information probes to every particle.

On ordinary text, probes can be trained by masking evidence/future chunks and
rewarding disagreements that best predict the revealed chunk. On generated
causal episodes, paired interventions and renderer orbits provide exact
counterfactual evidence. The same falsifier is used at inference; no Qwen,
host solver, gold graph, or hidden answer is available.

### 4.4 Evidence update

Each complete candidate receives a proper-score evidence update:

```text
log_w_i <- log_w_i + sum_j Score_psi(S_tilde_i, C_ji, observed_evidence)
```

Train the scorer for calibration, not merely ranking. Include deliberately
false counterfactuals and common-mode wrong particles so unanimous agreement
cannot substitute for evidence.

### 4.5 Whole-particle selection and resampling

When effective sample size falls below a threshold, select or resample entire
particles:

```text
ancestor_i ~ Categorical(softmax(log_w / tau))
P_i <- CloneWhole(P_ancestor_i) + bounded_private_perturbation
```

A straight-through categorical estimator, Gumbel resampling, or another
explicitly tested estimator may provide gradients. Crucially:

- never average `S_i` fields across particles;
- never construct a terminal program by independently selecting opcodes,
  pointers, and relations from different lineages;
- preserve at least one elite unchanged; and
- inject diversity only into private proposal memory, not committed facts.

### 4.6 Behavioral merge

Two particles may merge probability mass only when they are behaviorally
equivalent on the current probe basis:

```text
Merge(i,j) only if Execute(S_i, C_q) == Execute(S_j, C_q) for every admitted q
```

The state vectors are not averaged. One representative survives and receives
the combined log mass. This imports Shohin's terminal-state quotient insight:
syntactically different programs may be equivalent, but equivalence must be
defined by consequences, not coordinate proximity.

### 4.7 Halt and answer

The reactor halts when one coherent lineage has sufficient posterior mass and
no admitted challenge separates it from its remaining equivalence class. The
late reader consumes only the selected particle's terminal state. If no
particle clears the evidence threshold, the model abstains or spends another
recurrent round.

## 5. Why this is different from obvious prior art

| Prior family | What it already does | FCPT separator |
|---|---|---|
| Self-consistency / ToT | Multiple decoded text paths | Native latent structured particles trained end to end |
| Coconut | One recurrent continuous thought that may superpose alternatives | Explicit persistent alternatives with lineage and separate weights |
| PLR / MPCoT | Parallel latent paths and learned aggregation | Active pairwise falsification and no soft field aggregation |
| Guided stochastic recursive exploration | Perturbed recurrent trajectories and guide reweighting | Learned contradiction probes, structured world states, certified equivalence merge |
| RIMs / Global Workspace | Specialized modules coordinating through sparse communication | Exchangeable mutually exclusive hypotheses, not permanent specialists |
| Differentiable particle filters | Learned particles, weights, and resampling | Language-conditioned reasoning states plus endogenous challenge generation |
| CEGIS | Candidates refined by an external verifier | Model-owned bounded falsifier trained as part of inference |
| ETTR | One explicit typed state and recurrent controller | Posterior over several complete ETTR states with whole-lineage competition |

The safest novelty statement is:

> FCPT is a hypothesis for an end-to-end language model that performs
> falsification-coupled sequential Monte Carlo over structured latent programs,
> preserving whole-program identity through recurrent inference.

Do not claim that particles, recurrence, debate, typed state, or
counterexamples are themselves new.

## 6. Training without replacing ordinary pretraining data

The architecture should remain trainable on ordinary text. It does not require
one trillion tokens of custom program traces.

### 6.1 Ordinary-text objective

For normal pretraining sequences:

1. encode a prefix or surrounding evidence;
2. initialize several latent hypotheses about a masked future chunk;
3. have the falsifier select evidence units on which hypotheses disagree;
4. reveal the actual chunk under the standard next-token target;
5. update particle weights with a proper scoring rule; and
6. predict tokens from one selected whole particle.

The standard language-model loss remains. Auxiliary losses train calibration,
useful disagreement, lineage diversity, and adaptive halting.

### 6.2 Structured causal episodes

A minority curriculum supplies exact causal pressure:

- paired counterfactual worlds with matched surface statistics;
- renderer and entity renaming orbits;
- ambiguous prefixes where multiple hypotheses remain valid until later
  evidence;
- noncommuting operation order;
- variable cardinality and depth;
- contradictory or unsatisfiable episodes requiring abstention; and
- query deletion so the state must be useful before the question arrives.

These episodes train the mechanism, but the claim must be evaluated on unseen
generators and natural tasks.

### 6.3 Loss sketch

```text
L = L_LM
  + lambda_pred * L_masked_evidence_prediction
  + lambda_cal * L_particle_calibration
  + lambda_div * L_behavioral_diversity
  + lambda_probe * L_information_gain
  + lambda_state * L_typed_state_validity
  + lambda_halt * L_adaptive_compute
  + lambda_orbit * L_renderer_equivariance
```

Behavioral diversity must compare predicted consequences, not raw cosine
distance. Otherwise particles can satisfy the loss through meaningless latent
rotation while representing the same program.

## 7. Minimal matched falsifier

Do not attach FCPT to a trillion-token run first. Build a small causal test in
which plural inference is necessary.

### 7.1 Board families

Use at least three independently generated families:

1. **Ambiguous noncommuting programs:** local event marginals match, but
   operation order changes the result.
2. **Binding and graph programs:** repeated names, variable cardinality,
   renderer changes, and late evidence distinguish complete bindings.
3. **Counterexample-guided function induction:** several candidate functions
   match initial examples; additional generated examples eliminate them.

Train at depths 2--4 and cardinalities 3--6. Develop at depths 5--7 and
cardinalities 7--9. Keep a post-freeze family, renderer, depth, and generator
seed sealed.

### 7.2 Arms

All arms receive identical data and parameter ceilings:

- A: ordinary Transformer;
- B: single-stream recurrent-depth Transformer;
- C: `K` independent recurrent particles with no communication;
- D: parallel particles with soft mean aggregation;
- E: whole-particle selection without learned falsification;
- F: full FCPT;
- G: FCPT with shuffled contradiction messages;
- H: FCPT with particle lineage randomly swapped before answer.

Report both parameter-matched and FLOP-matched comparisons. For the latter,
give the single-stream recurrent control the same total proposal-step budget
as all FCPT particles combined.

### 7.3 Pass criterion

FCPT advances only if all conditions hold:

1. at least **+10 absolute points** in exact OOD joint accuracy over the best
   matched control, aggregated across three families;
2. positive gain on every family and at least four of five seeds;
3. monotonic or saturating benefit from `K=1,2,4,8` particles at fixed trained
   weights;
4. falsifier ablation loses at least five points;
5. soft aggregation performs materially worse than whole-particle selection;
6. shuffled challenges and lineage swaps cause the predicted degradation;
7. no gain survives when counterfactual labels are randomized;
8. ordinary language loss and direct capability stay within a preregistered
   non-regression bound; and
9. the unopened confirmation result passes without threshold tuning.

Kill the lane if a deeper single recurrent stream matches it at equal FLOPs,
if particles collapse to one behavior, if the scorer only learns benchmark
labels, or if gains disappear under new renderers/generators.

## 8. Integration path for Shohin

1. Freeze the current 125M checkpoint and 63.0296% Qwen-hosted product system
   as controls. Neither is the architecture claim.
2. Reuse ETTR's typed packet, source deletion, endpoint-aware relations, late
   reader, and exact state validator.
3. Replace the single ETTR compiler/reactor state with `K=4` exchangeable
   complete particles for the first pilot.
4. Use one shared compiler/reactor across particles; add particle seeds,
   scorer, contradiction bus, and whole-particle resampler.
5. Start with soft proposal states but hard whole-particle selection at the
   late reader. Introduce hard transactions only after the plurality mechanism
   clears its causal gate.
6. Train the minimal generated board first. Only after a matched win should
   continued pretraining mix ordinary language and causal episodes.
7. If the mechanism survives sealed transfer, then scale width, recurrence,
   and pretraining tokens.

The first implementation should be smaller than the existing 67.7M ETTR when
possible. Weight sharing means `K` increases activation memory and FLOPs, not
parameter count. That makes the architectural comparison cleaner.

## 9. Principal risks

- **Particle collapse:** all particles learn the same hypothesis. Mitigate
  with behavioral diversity, ambiguous-prefix training, and resampling noise.
- **Artificial diversity:** particles differ in coordinates but not behavior.
  Measure consequence disagreement only.
- **Frankenstein recurrence:** any fieldwise average reintroduces the known
  failure. Ban it in the schema and test it explicitly.
- **Self-confirming verifier:** all particles and the scorer share one error.
  Use false-label controls, held-out generators, direct evidence prediction,
  and exact synthetic oracles during development.
- **Gradient failure through resampling:** compare straight-through,
  Gumbel-based, and stop-gradient selection in a tiny gate before scaling.
- **Overthinking:** extra rounds may degrade correct hypotheses. Train a halt
  distribution and report accuracy as a function of rounds.
- **Compute inflation:** `K` particles can merely buy more FLOPs. Match total
  recurrent proposals against deeper single-stream controls.
- **Weak substrate:** architecture cannot invent facts or primitives absent
  from the base. Separate reasoning transfer from stored knowledge.
- **Synthetic overfit:** success on generated state machines is not enough.
  Require new generators and natural transfer before any broad claim.

## 10. Research claim ladder

1. **Mechanics:** particle permutation equivariance, exact lineage,
   no field averaging, valid resampling, finite gradients.
2. **Learnability:** particles remain behaviorally distinct and scorer weights
   correlate with correctness on unseen episodes.
3. **Causal plurality:** corrupting/removing the winning particle changes the
   answer; shuffled challenges reduce performance.
4. **Compositional transfer:** depth, cardinality, renderer, and generator OOD
   gains over recurrent controls.
5. **Language transfer:** ordinary-text continued pretraining improves sealed
   reasoning tasks at matched language loss.
6. **Scaling:** more particles or rounds improve accuracy predictably without
   increasing parameters.
7. **Architecture claim:** the full mechanism beats strong standard and
   adjacent-prior controls under sealed matched evaluation.

Anything below rung seven is useful research, but not a revolutionary
architecture claim.

## Recommended immediate decision

Freeze product-score work except as a control. Implement one small
Falsification-Coupled Particle Transformer pilot. The decisive scientific
question is:

> At fixed parameters, data, and total recurrent proposal compute, does
> preserving and actively falsifying several complete latent programs produce
> systematic compositional generalization that a Transformer, a single
> recurrent stream, independent particles, and soft aggregation do not?

If yes, Shohin has a credible architecture program. If no, close the lane
before expensive pretraining and move to the diffusion/global-revision
alternative.

## 11. Superseding architecture decision and DIVERGE result

The later owner directive closes PCSD and FCPT and orders DIVERGE as a
factorized extension of the protected CSDC role-copy result. The bounded gate
is now resolved. Learned source compilation and delayed conflict recovery are
exact across five seeds, but the first V3 comparison omitted active
state-group memory. The corrected exact runtime raises the matched
whole-particle control from 33.333% to 72.222%; DIVERGE remains 100%, soft and
no-conflict controls remain 66.667%, and independent particles average
65.278% across 720 episodes / 2,160 queries.

The broad gate nevertheless fails on all five seeds because four-world
effective storage is `1.893x` versus the frozen `2.0x` minimum. Sharing becomes
material at eight worlds (`3.412x`) and reaches `27.365x` at 64. This supports
only a narrow delayed-recovery/version-space result above the observed
amortization boundary. It does not authorize CUDA profiling, long pretraining,
or a general architecture claim. Aggregate SHA-256:
`8e4405920379b7c0a2f4a0c9acc463839a3816b4a87e84f78fcb9d656e17aaab`.
