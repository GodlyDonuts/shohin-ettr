# R12 Opcode Program Failure Tree

**Status:** active preregistered routing document
**Scope:** contract-v8 syntax graph, valid-program projection, and the next
coherent-program successor
**Promotion rule:** unchanged source-deleted strict WORLD and COMMAND gates

## Why this exists

The campaign must not wait for one treatment to fail before inventing the
next one. Three orthogonal measurements now run concurrently:

1. The uninterrupted 15,000-update syntax-graph trajectory asks whether a
   delayed generalization phase exists under unchanged optimization.
2. Zero-parameter registry projection asks whether the trained local opcode
   probabilities already contain a coherent valid program that independent
   argmax destroys.
3. The CPU public-syntax identifiability audit asks whether the permitted
   WORLD and COMMAND inputs contain enough information to identify an opcode
   skeleton before any neural architecture is blamed.

These are not seed duplicates. They isolate optimization time, decoding, and
input identifiability respectively.

## Fixed outcomes and successors

### A. Projection restores exact terminal packets and both strict axes

Replicate the result on a fresh population. If it survives, replace the
non-differentiable evaluation projection with a trainable globally normalized
program distribution over the same hash-bound registry. Keep operands
compositional and preserve the exact executor.

### B. Projection restores exact terminal packets but strict causal axes stay zero

Opcode coherence is useful but insufficient. Freeze the projected opcode
skeleton and train opcode-conditioned source, target, relation, type, and
value heads. The first diagnostic is oracle-operand replay partitioned by
field; the successor architecture aligns each dynamic operand with a syntax
node and state slot under the fixed skeleton.

### C. Projection leaves exact terminal packets and strict axes at zero

Reject decoding-only repair. Route by the CPU audit:

- If public WORLD+COMMAND syntax has high train-to-development modal accuracy,
  compile the opcode program directly from renderer-normalized public ASTs.
  The neural module predicts only residual choices not fixed by syntax.
- If train coverage is high but modal accuracy is low, current opcode
  skeletons alias semantically distinct programs. Replace the label space
  with a hierarchical grammar of edit atoms and state preconditions.
- If train-to-development coverage is low while the development Bayes ceiling
  is high, exact signature lookup is the wrong generalizer. Train a
  tree-structured parser over canonical AST productions, with held-out
  production combinations as the gate.
- If even the development Bayes ceiling is low, the opcode skeleton is not
  identifiable from permitted public inputs. Reject the current target
  ontology and compile the query-independent terminal-state quotient through
  syntax-local edit atoms with explicit state guards.

The high-identifiability successor is already implemented behind
`--registry-projected-opcode-training`. Unlike the rejected flat selector, it
does not learn a 2,530-way class-prior head. It derives a globally normalized
distribution from the compiler's renderer-normalized per-step opcode evidence,
backpropagates one complete-program likelihood, and keeps compositional
operands plus the exact executor. This branch remains held until the public
syntax audit demonstrates that the target program is identifiable.

The low-identifiability successor is not another opcode variant. Its target is
a syntax-local edit ontology: public AST productions propose guarded state
edits, anonymous-token equality supplies bindings, and the exact state algebra
applies those edits. It cannot inherit assessor-trace opcode class names or a
global program registry. This branch is selected only if the audit rejects
public identification of the current opcode ontology.

### D. The 15k trajectory crosses both strict axes

Replicate at the same budget on a fresh population and compare the timing of
loss, exact schedule, exact terminal packet, and strict causal transitions.
Only a replicated late transition counts as delayed emergence. Then use its
weights as the structured-decoder source rather than restarting from 1k.

### E. The 15k trajectory does not cross both strict axes

Close additional same-objective duration scaling. The lower loss and higher
local field accuracy already observed at 5k cannot justify longer runs.

**Measured:** job `725574` completed all 15,000 updates. Joint schedule
accuracy reaches 15.20%, exact query programs remain 18.75%, and exact
oracle-initial terminal packets remain `32/512 = 6.25%`; fully autonomous
WORLD and COMMAND remain exactly zero with zero causal margin. Outcome E is
final. Contract-v8 duration scaling is closed.

## Resource policy

- Stokes CPUs run exhaustive corpus audits, parser invariance tests, and data
  materialization.
- Newton V100s run bounded architecture/evaluator diagnostics when memory
  permits.
- Newton H100s are reserved for successors with a distinct causal mechanism
  or for replication after a strict-gate gain.
- No capacity is spent on seed multiplication before a mechanism crosses the
  advancement gate.
- Jobs may be submitted held or with a future begin time so scheduler latency
  overlaps CPU gates, but only the branch justified by the frozen report may
  execute.

This policy uses all useful compute while preventing many simultaneous jobs
from answering the same question.

## Measured routing outcome (2026-08-02)

The full corrected public-syntax audit is frozen at
`artifacts/r12/ettr_public_opcode_identifiability_e5dcb41_r1/report.json`
(file SHA-256
`951fbd78006352db6c3c2d5cd4b192831a9d4891de792fbc2a981e2dff683985`,
payload SHA-256
`2f7479c763165f62d8e75401e160a5971580f6fcaa8ce16e7462ba835ac7e022`).
It covers 40,000 train cores and 5,000 development cores. The current full
opcode-sequence target is not reliably identifiable under any high-coverage
public quotient:

- exact alpha-normalized WORLD+COMMAND syntax has a 75.36% development Bayes
  ceiling but only 5.66% train-to-development signature coverage;
- operator-abstracted WORLD+COMMAND syntax has 79.71% coverage but only
  55.18% development Bayes accuracy and 43.04% all-instance modal accuracy;
- topology has 99.31% coverage but only 43.59% development Bayes accuracy and
  40.02% all-instance modal accuracy.

The globally normalized registry-training job therefore remains held. The
audit routes to the syntax-local branch, but it also localizes a narrower
missing interface before replacing the complete target ontology: public
COMMAND documents explicitly declare opaque identifiers as
`call(3, identifier, ordinal, payload)`, while contract-v8 makes applications
discover that declaration through three synchronous equality/AST hops. The
new declaration-binding treatment resolves that exact public path before
learned graph propagation. It reads no sidecar, QUERY, answer, trace, target,
or oracle state, adds no parameters, and remains equivariant to arbitrary
opaque-name renaming. The unchanged exact executor and source-deleted gates
remain decisive.

Failure routing is fixed:

1. If exact-mask-only crosses both axes, replicate exact masking first and
   treat declaration binding as an ablation.
2. If declaration binding crosses both axes while exact-mask-only does not,
   replicate the binding on a fresh population, then promote the explicit
   public symbol-table edge.
3. If declaration binding improves local fields or terminal packets but not
   both strict axes, retain it as an interface correction and replace ordered
   opcode imitation with syntax-local guarded edit atoms.
4. If it is neutral, close learned multi-hop declaration resolution as the
   primary bottleneck and move directly to a state-grounded edit ontology;
   do not run more seeds, depth, width, or duration on contract-v8.

The first branch is now measured. Exact-mask-only V100 job `725610` completes
cleanly with 7.687% joint schedule accuracy, zero exact terminal packets,
60.9375% autonomous factual top-1, and zero strict WORLD/COMMAND. Exact
masking is retained as an interface invariant but closed as a standalone
reasoning repair. The declaration-binding gate is therefore the sole live
contract-v8 treatment; its negative result will immediately activate branch
3 or 4 rather than another contract-v8 scale trial.

The branch-3/4 successor is implemented before that result. Contract-v8 of
the terminal-state line combines the exact cover-verified AST and public
declaration resolver with the previously audited atomic typed-edit algebra.
It predicts one query-independent guarded state difference, not an ordered
64-step opcode serialization. Initial state identity is copied unless a
categorical node/relation/root/disposition edit is purchased, and the fixed
algebra enforces valid hard state. This directly tests whether the public
syntax can identify the semantic quotient even when it cannot identify one
arbitrary low-level program. It remains isolated and must stay held until the
declaration-bound schedule gate is read.

The declaration-binding gate is now measured. V100 job `725613` raises joint
schedule accuracy from the exact-mask control's 7.687% to 9.651% and recovers
`32/512 = 6.25%` exact oracle-initial terminal packets, so the public binding
edge is retained. Autonomous factual top-1 remains 60.9375%, however, and both
source-deleted strict WORLD and COMMAND remain exactly zero with no strict
margin. This is outcome 3 above: a local interface correction without causal
answer movement. The redundant H100 mirror was canceled and the already-held
guarded-edit V100 job `725616` was released immediately. Its exact source is
private commit `17b4e605689a707254699d01c7a543e9b88c9420`; its sealed runtime
manifest SHA-256 is
`b9fa32a0572e55a2d03f884a3cc827d7524e4501328fcb8630a2f3f46ec35e33`.

The guarded-edit gate is also now measured. V100 job `725616` decreases
oracle-initial value accuracy from 74.345% to 70.961% and autonomous-initial
value accuracy from 58.543% to 55.049%; exact packets and both strict axes
remain zero. This closes one-shot episode-wide terminal editing even after
the public declaration repair. The next branch is not another edit loss or
seed. Public COMMAND syntax exposes 1--6 top-level applications beneath
`call(13)`. The v9 successor extracts those roots in semantic child order
independently of renderer layout and applies one tied latent state update per
visible operation before emitting the guarded terminal difference. This
directly tests multi-operation composition without learning or consuming the
assessor's arbitrary low-level opcode schedule.

Before reading that GPU endpoint, a CPU identifiability successor is staged.
It uses assessor operation traces only as labels and measures whether a
declaration-resolved public operation target generalizes under four contexts:
operation only, WORLD plus operation, WORLD plus operation prefix, and WORLD
plus complete COMMAND and rank. High coverage and Bayes accuracy justify an
operation-aligned intermediate-state curriculum; low accuracy rejects that
label ontology before another GPU run.

The operation-recurrent endpoint is now measured. V100 job `725618` learns
the terminal disposition and reaches 57.8125% autonomous factual top-1, but
oracle-initial value accuracy falls from 74.344978% to 66.375546%, autonomous-
initial value accuracy falls from 58.542576% to 52.265284%, exact packets stay
zero, and both strict axes remain exactly zero. This closes recurrence that
updates only a latent slot tensor and emits one episode-terminal edit. The
failure is not routed to more recurrent depth, width, duration, or an
autonomous-initial duplicate. The next neural mechanism must expose a state
transition after every public operation and must receive operation-boundary
credit. Stokes job `760906` selects the target ontology: identifiable local
mutations admit cumulative intermediate-state supervision; aliased mutation
labels require a state-delta quotient audit first.

The high-identifiability branch is code-complete before the audit returns.
It is not another opaque recurrent latent: after each public operation it
decodes an atomic typed edit, applies that edit through the fixed algebra,
and supplies the resulting state to the next tied transition. Training alone
replays the frozen transaction labels to recover cumulative operation-
boundary state targets and localize the first divergence. A final tied edit
writes outcome/disposition. Inference remains source-deleted and reads no
trace or target. This branch stays held until operation-target public
identifiability is measured; a low-identifiability result routes to the
state-delta audit instead of spending a GPU on an aliased label.

The first exact-operation audit attempt `760906` failed technically rather
than scientifically. Its parser omitted the valid direct local COMMAND form
`call(15, nuisance, call(13, operations...))`. Commit `70e5dcb` aligns the
audit with the already-correct production operation router and freezes a
regression test. Repaired exact-label job `760907` now runs concurrently with
state-delta quotient job `760908`, each using 48 Stokes CPU cores. The latter
removes cursor and mutation ordering and separately audits per-operation
semantic deltas plus cumulative runtime states under exact,
operator-abstracted, and topology WORLD contexts. This precomputes both
failure branches: high public identifiability releases held operation-state
job `726970`; low identifiability cancels it and requires a different public
target ontology.

Both reports are now measured and select the low-identifiability branch.
Exact mutation labels reach only 20.41% transfer from resolved operation at
full coverage; adding exact WORLD lowers useful all-instance transfer to
9.20% at 45.38% coverage. The order-independent semantic delta improves to
43.49% from operation alone and 44.75% from topology plus operation, but those
values sit at their corresponding 43.55% and 45.78% Bayes ceilings. Cumulative
runtime state reaches at most 9.43% in the exact-WORLD operation context and
falls below 0.4% in high-coverage abstract contexts. This is target aliasing,
not a request for more neural capacity.

V100 smoke `726972` proves the operation-boundary implementation is
mechanically healthy: all six boundaries reconstruct, forward/backward is
finite, and custody seals after two updates. Held 1,000-update job `726970`
was therefore canceled for scientific, not technical, reasons. The next CPU
gate factors the operation delta into shape, addressed coordinates, and
payload transformations. Its result determines whether the next architecture
is a grammar action classifier plus state pointers/value heads or whether the
current state/edit ontology must be rejected entirely.

The first factor audit is now measured and rejects that coarse three-way
quotient as a sufficient target. Job `760909` completed 60,512 development
operation instances on 48 Stokes cores in 23m39s. At broad coverage, exact
shape transfers at only 54.64% (`world_alpha_operator_operation`, 98.73%
coverage) against a 59.84% development Bayes ceiling; exact addresses reach
45.59% (`world_topology_operation`, 99.98% coverage) against 47.00%; and exact
payloads reach 47.66% (`world_alpha_operator_operation`, 98.73% coverage)
against 52.46%. These are modest modal improvements, not a clean compiler
interface. Report SHA-256 is
`f3c7102121fe389f6c7c37e213a7deef71e1674d60910be5d61eca252c88ea43`;
payload SHA-256 is
`bed3110cd323b0b0c8adcc60d973a96c1738f2cd0df00b1179ac48b3f31d925a`.

The failure branch is already concrete. A sparse operation compiler now
predicts node-edit, relation-link, and relation-unlink cardinalities before
binding addresses and payloads to the current typed state; hard inference
applies exactly those counts through the existing fixed algebra. It remains
unlaunched because the coarse shape audit does not validate those individual
marginals. Audit v3 scores each cardinality, field-change histogram, and
status change independently. High cardinality transfer releases one bounded
factorized compiler gate. Low cardinality transfer rejects even this sparse
effect ontology before GPU allocation and routes to state-conditioned public
semantic primitives rather than more edit-head capacity.

The release rule is frozen before reading v3. One broad-coverage mode must
cover at least 95% of development instances and transfer at least 75% for
node-edit count, 90% for edge-add count, 90% for edge-remove count, 65% for
node-field histogram, and 95% for status change. Every threshold must pass.
Passing releases the single held sparse-effect run; missing any threshold
cancels it without allocation and activates the operation-effect set
transducer design, where a bounded unordered set of typed effects replaces
episode-wide dense coordinate classification.

The v3 gate is now measured and fails. Stokes job `760912` completed 60,512
development operation instances in 24m35s on 48 cores. The report is frozen at
`artifacts/r12/ettr_public_operation_state_marginals_e5cb34c_r1/report.json`
(file SHA-256
`bd6c8f35d93cf4ba2d99562a28825b4f86d43a18dd974de49d428f8b903d2dff`,
payload SHA-256
`3e828f59d31a7aac651c48d349520f4e863a3fb0925a2219eb5c54f3f0f269ec`).
Node-edit count is 74.8248% at full coverage, edge-add count is 79.6635% at
98.7308% coverage, edge-remove count is exact, node-field histogram is
74.8248% at full coverage, and status change is exact. The first two miss the
preregistered 75% and 90% thresholds, so held GPU job `726975` was canceled
without allocation.

The successor is mechanism-distinct rather than another cardinality head.
Contract v12 predicts up to 16 unordered typed effects with kinds allocate,
write, clear, replace, link, unlink, root clear/set, commit, halt, and reject.
Each effect binds operands to the operation-conditioned typed state and the
existing fixed algebra applies the aggregate. Training performs detached
Sinkhorn bipartite matching against a canonical state-difference effect set;
inference remains a single deterministic hard set and cannot inspect targets
or select among candidates. Link and root pointers may address a node
allocated by another effect in the same set, which is required by the
simultaneous algebra. The sealed contract records the 16-slot capacity and
matching rule. Local compiler, integrated loss, evaluator, supervision, and
custody gates pass 34 tests. The next branch is a two-update GPU smoke; target
capacity overflow is a technical geometry result and may raise the bounded
capacity to the measured requirement, while finite smoke success alone only
authorizes a bounded 1,000-update causal gate.
