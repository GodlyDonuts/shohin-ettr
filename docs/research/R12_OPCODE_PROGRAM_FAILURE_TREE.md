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
The v3 node-change count is 74.8248% at full coverage, while edge-add count is
79.6635% at 98.7308% coverage, edge-remove count is exact, node-field histogram is
74.8248% at full coverage, and status change is exact. The first two miss the
preregistered 75% and 90% thresholds, so held GPU job `726975` was canceled
without allocation.

Post-result review found one label-alignment defect that does not change that
decision: v3's node count includes root-only changes because root membership
is part of the audited node tuple, while the deployed atomic algebra has a
separate root action. Audit v4 preserves the old labels and adds exact atomic
node, root, disposition, and total-effect counts. It also freezes train and
development total-effect histograms and maxima so the neural set capacity is
corpus-bound rather than guessed. The independent edge-add miss remains
decisive, so the factorized run stays rejected.

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

The first two-update attempt, Newton job `726976`, produced no model evidence:
it stopped before construction on the pilot's obsolete 200M parameter guard.
The user had already removed that ceiling. The guard is deleted, parameter
count remains contract-sealed, and held dependent job `726977` was canceled
untouched. A new immutable-runtime smoke replaces it; the v4 Stokes capacity
audit runs concurrently rather than waiting on the GPU result.

The next sealed smoke, `726978`, passed lineage and model construction but
failed during the first hard pre-training interface forward: effect kinds were
FP32 and node pointers BF16 at their `einsum` boundary. It performed zero
updates and produced no report. Held dependent `726979` was canceled. Masked
pointer distributions are now explicitly promoted to FP32, with a BF16
compiler regression test. This preserves the treatment and requires one more
identical two-update smoke from newly sealed source before any long fit.

## Contract-v12 advancement and failure routing

The effect-set treatment is not an open-ended invitation to add slots, seeds,
or duration. Its first 1,000-update gate has four preregistered outcomes:

1. **Exact local effects and both strict axes improve.** Replicate the same
   source, budget, and seed geometry on a fresh population. Only a replicated
   WORLD and COMMAND gain can promote the unordered effect set.
2. **Effect kinds collapse to NOOP or one dominant class.** Replace anonymous
   learned effect queries with public-AST role anchors derived from the current
   operation's semantic children. This branch changes the binding mechanism;
   it does not increase free-slot count.
3. **Entity effects become accurate but relation additions remain wrong.**
   Split the algebra into two causal phases. Phase A applies allocate/write/
   clear/replace effects. Phase B then predicts link/unlink/root/disposition
   against the resulting post-entity state, making newly allocated endpoints
   explicit rather than requiring simultaneous cross-slot coordination.
4. **Local typed effects and terminal packets improve but fully autonomous
   WORLD/COMMAND remain zero.** Freeze the compiler and rerun crossed
   oracle-program/autonomous-state plus autonomous-program/oracle-state
   isolation. If state is sufficient under an oracle program, the remaining
   defect is query/compiler consumption; if not, close this state quotient.

A mechanically clean run with falling loss but no exact local or causal gain
is a rejection, not evidence for grokking. Width, more seeds, and longer
duration are prohibited until one of the mechanism-specific local gates moves.

## Contract-v12 mechanical and capacity result

Newton V100 smoke `726980` completes two updates from immutable source
`c630a5596c98a074198c25898d4362bb407b3e81` with finite BF16
forward/backward, evaluator contrasts, report sealing, and custody. The
authoritative runtime is
`scratchpad/shohin_ettr_operation_effect_set_runtime_c630a55_r2`; its 3,582
files are read-only and its SHA256SUMS SHA-256 is
`d0380af5da80fc0a2825869f0edb366a35374887d6e4265d2bcba6d9fa135668`.
The complete system has 200,785,290 parameters, including a 44,846,396-
parameter effect compiler. The smoke is a mechanical pass only: two updates
leave strict WORLD and COMMAND at zero.

Independent Stokes audit `760916` then measures the exact algebra-aligned
effect-set cardinality over 494,480 train and 60,512 development operations.
Both splits have maximum 10 effects; train has 72 ten-effect operations and
development has six. Report
`artifacts/r12/ettr_public_operation_state_delta_v4_3c5b954_r1/report.json`
has file SHA-256
`894c287c80a093c9f4235358c00d7b7dfb42853b2218ac5b31a85b287eea8095`
and payload SHA-256
`88eb06581864b850e34045b0b2f96fbd78e797dc80a28bb4784f87f0f8a75bf5`.
The sealed 16-slot compiler therefore has measured capacity headroom. This
released the preregistered 1,000-update job `726981`; it does not relax any
local-effect or source-deleted causal advancement gate.

## Contract-v12 result and contract-v13 role-anchor route

The first released long run `726981` found an empty trailing operation-rank
padding defect and wrote no scientific artifact. The exact replacement
`726984` completed all 1,000 updates from immutable source `16455f3` and sealed
its output with SHA256SUMS SHA-256
`1d1688b5fa027111f79e8eb945f4c924c5e5d6bd54eff6efe774b69e7545d89f`.
Independent held-out V100-16GB evaluator `726988` then compared immutable
initial and final compiler weights on data seed 13. Its sealed report has file
SHA-256 `63c49577615be4ebf88d0cb9e72cf6ec841e4decc91799abf0e9a1fd3dc78c6c`.

The result selects preregistered outcome 2 decisively. Before training, the
37,888 hard motor predictions occupied only effect kinds 5, 8, and 10. After
training, every one is kind 0/NOOP. Complete effect-set, complete dense-edit,
operation-state, terminal-state, positive entity, and positive relation-link
exactness are all zero before and after. The apparent relation-action rise
from 26.56% to 67.57%, root rise from 57.26% to 100%, and unchanged roughly
78% disposition score are negative/default-prior effects, not learned edits.
Fully autonomous factual accuracy and both source-deleted strict axes remain
exactly zero. This closes anonymous permutation-symmetric effect queries;
loss reduction, extra seeds, width, duration, or more free motors cannot
reopen them.

Contract v13 breaks only the failed symmetry. Role 0 is the visible public
operation root. Roles 1 onward are direct semantic children in the exact
renderer-invariant child-rank order reconstructed by the public AST. The
first implementation attaches two typed-effect motors to each role. Invalid
or absent roles are mechanically forced to NOOP. The mechanism does not decode ontology,
declaration payload, QUERY, target, assessor trace, answer, or host semantics;
state operands and the fixed atomic algebra remain unchanged. A full-corpus
CPU audit must first prove that role and valid-motor capacity cover every
operation. Then a two-update isolated GPU smoke must prove finite hard forward/backward,
nonempty role gradients, evaluator compatibility, and custody. Only those two
mechanical gates release one 1,000-update fit.

The post-v13 failure tree is frozen in advance:

1. Exact local effects, terminal state, and both strict axes improve: replicate
   the same contract on a fresh population before promotion.
2. Role-local kinds remain collapsed or positive effects remain zero: reject
   fixed motors-per-role and replace categorical motor emission with a
   role-conditioned typed transducer whose action cardinality is explicit.
3. Entity effects become exact while positive links fail: apply entity motors
   first and bind relation/root motors against the resulting post-entity
   state.
4. Local effects and terminal state improve while strict axes remain zero:
   freeze v13 and run the crossed program/state sufficiency isolation; do not
   spend on another state fit.
5. No exact local or causal metric moves: close the effect-set state quotient
   and advance to an architecture that compiles explicit public operation
   transitions rather than predicting assessor-derived deltas.

The first mechanical pass exposed a capacity flaw before long training.
Stokes `760920` measured maximum public arity three, hence four public roles,
but the original audit checked only whether eight reserved roles were enough.
Because absent roles are forced to NOOP, a four-role operation with two motors
per role exposes only eight valid motors while the independent exact-delta
audit has a ten-effect maximum. The `760920` capacity verdict is therefore
invalid for release even though its arity measurements are retained. Newton
smoke `726992` proves only that the old geometry executes and backpropagates;
held long job `726993` was canceled untouched.

Corrected v13 fixes four roles and five motors per role: twenty total motors
and at least ten valid motors for every legal nonempty operation. The
replacement capacity audit binds the earlier exact effect-capacity report by
file SHA-256, checks matching corpus/tokenizer/operation counts, and requires
both role and minimum-valid-motor capacity. This corrected geometry must pass
a new two-update smoke before the bounded 1,000-update gate can be released.

Both corrected mechanical gates now pass. Stokes job `760922` audited 40,000
train and 5,000 development cores, comprising 494,480 and 60,512 operations,
in 4m55s on 48 CPU cores. Both splits require four roles, both have a measured
maximum of ten exact effects, and the corrected geometry guarantees at least
ten valid motors. The report is
`artifacts/r12/ettr_public_operation_role_capacity_33785eb_r2/report.json`;
its file SHA-256 is
`5d155f98c17431ddfcb6a5a4f5d2bd977d3ba1edeac522b42058cc0530c7dde8`
and payload SHA-256 is
`1d14493922ce6b9c27d982b3f2a6c4b33428ed952f7d8d60aec7cbfbc5e43041`.
The first submission `760921` failed after five seconds because the exact
prior capacity receipt had not been transferred to Stokes; it read no corpus
and produced no scientific result. The receipt was transferred atomically,
verified against its frozen SHA-256, and the unchanged audit was rerun.

Newton V100-16GB smoke `726995` then completed two updates in 2m33s from
immutable source `33785eba316905f63eaa084c9e3d8f12ac53b7cc` and runtime
`scratchpad/shohin_ettr_effect_role_runtime_33785eb_r3` (SHA256SUMS SHA-256
`bd244b7553957858943a27c97a3646444b3a4363a3d8c2c527c2796287012970`).
Its complete system has 200,787,338 parameters. The report SHA-256 is
`0ee84f77b97fbb0567825c03c9da4bf0446ba3a92db677883f2226e249a330dd`,
contract SHA-256 is
`d4718a6d39194920f98b052dc2829a94ee33c19b3ff3fb8eb6585e22f2ea5bb2`,
and SHA256SUMS SHA-256 is
`88d93ea90772b1bae1b7c848eb60308871bd92dd27a8c01ff0061455f18ec0fe`.
The released H100 job had a ten-hour scheduler estimate because every H100
was allocated, so it was canceled before allocation and replaced by the exact
V100 scientific treatment as job `726997`. Prior matched v12 timing predicts
roughly 23 minutes rather than a ten-hour idle wait.

## Contract-v14 preregistered cardinality successor

Contract v14 is implemented before reading the v13 endpoint. It targets one
specific predicted failure: role anchors may break query permutation symmetry
while every motor still independently chooses NOOP under hard argmax. V14
separates three decisions:

1. a pooled operation head predicts the exact total number of real effects;
2. a role-conditioned activity head ranks the twenty anchored motors and the
   hard executor activates exactly the predicted top-k valid motors;
3. activated motors choose only among the eleven non-NOOP typed effects,
   while inactive or absent-role motors are forced to NOOP.

Training retains detached typed bipartite matching and adds a class-balanced
exact count loss. Inference remains one deterministic hard trajectory with no
QUERY, target, answer, trace, candidate search, or host semantic execution.
This is not another seed or duration arm: it removes NOOP competition from
the typed-kind decision and turns cardinality into an explicit public
operation-level variable. Focused compiler, gradient, custody, evaluator,
audit, and routing tests pass.

## Contract-v13 result and contract-v14 release

V100-16GB job `726997` completed 1,000 updates in 25m46s from exact source
`33785eba316905f63eaa084c9e3d8f12ac53b7cc`. Its run, report, and contract
SHA-256 values are respectively
`6a9fd5b145ff9f9d58cf9e8752c55ee518588dad1230a3e05646c073d1fdcd48`,
`b7ee0010c58831f83bf6c4d3db82c1cf231c67cc5f75db9bfdf12d563a7765c4`,
and `f1e22b279713a3a8c8d9d253c3690b4b29f63ca2fa8e011b144134a9a56d72fe`.
Independent evaluator `726998` completed in 3m10s; its SHA256SUMS and report
SHA-256 values are
`44816524e2e2de287af4c6019970ece098bb0d19587c331b75f886d6cb3ba6d2`
and `be7dbda5f6f2a3b12872d5b848ceeb6107b025ccb113799f223cae2f7c5257d0`.

The predicted fixed-motor failure occurred. Before training, hard predictions
used several kinds. After training, 45,440/47,360 predictions are NOOP and
1,920 are COMMIT. Positive entity and relation-link exactness are both zero;
complete effect-set, dense edit, terminal-state, WORLD strict, and COMMAND
strict are all zero before and after. The exact contract-bound router records
NOOP share 95.9459% and selects `explicit_effect_cardinality_gate`. V13 is
closed; more seeds, width, duration, or motors are prohibited.

Stokes job `760928` independently replayed 494,480 train and 60,512
development operations on 48 CPU cores. Every exact non-NOOP target is either
WRITE or LINK: train has 278,020/395,200 and development has 33,927/50,995.
The maximum-to-minimum ratios are only 1.4215 and 1.5031. Report file and
payload SHA-256 values are
`9505409bf8cbd1723d8a35e5371852c672d2d6ee672cc88ddd7b336d71010a8e`
and `980520d87a505f00d4deb277efa85d1703d2a947cb60bdbf0d86b49dff53bbff`.
This rejects broad non-NOOP class imbalance and makes explicit cardinality the
minimal justified intervention.

V14 is sealed from exact source
`0ca7408abf06a039736c07027d0ddf485cd7582a` at
`scratchpad/shohin_ettr_effect_cardinality_runtime_0ca7408_r2`. It contains
3,645 files, zero writable files, zero links, passes full checksum replay, and
has SHA256SUMS SHA-256
`eee96d9544817ef45042dc86f51c5d99379895018de17cc20aa00612512627fd`.
The first two-update any-GPU smoke `727090` completed cleanly on V100-32GB
`evc13` in 3m07s. Its run, report, and contract SHA-256 values are
`190584b41e93cbfcb6ffc17a062fcf3f1e9cde2b5288af252b7c8d963060b4bd`,
`a31531cb2a09981b0ec5f0048540aebc18ef9f4282898f07b86d0a537613c740`,
and `4e4f6bf6ee20a8616dbb1470dfa7492de478005394b1cb673c337dd867b22ce2`.
The exact v14 schema, explicit count/top-k objective, finite loss/gradient,
report seal, and full custody pass. This releases sole 1,000-update any-GPU
fit `727778`; independent evaluator `727875` is dependency-held behind it.
No duplicate seed, width, duration, or device arm is authorized.
