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
