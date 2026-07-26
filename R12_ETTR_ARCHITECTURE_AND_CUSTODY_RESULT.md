# R12 ETTR Architecture and Custody Result

## Decision

`architecture_continuation_contract_complete_h100_profile_pending_pretraining_held_capability_unproven`

Shohin now has a concrete architecture core, a complete falsification matrix,
and an exact causal continuation/training contract for later pretraining and
post-training. It does not yet have demonstrated general reasoning. The user
has explicitly held continuation pretraining until the architecture is
qualified.

## Architecture

The Endogenous Typed Theory Reactor has four causal stages:

1. **World compiler:** actual Shohin residuals from raw world tokens
   cross-attend anonymous object slots and produce categorical value codes,
   latent types, a capped sparse relation ledger, activity, root, commit, and
   halt state.
2. **Command-conditioned reactor:** after world sealing, a separate raw-token
   command stream enters a shared recurrent controller. The controller emits
   only `ALLOC`, `WRITE`, `CLEAR`, `LINK`, `UNLINK`, `SET_ROOT`, `COMMIT`, and
   `HALT`, plus a distinct terminal `REJECT`.
3. **Typed graph update:** an edge-aware typed message bus preserves the
   identities and directions of relation endpoints before each transaction;
   each transaction changes only generic state. No
   family opcode, arithmetic routine, rewrite matcher, resource scheduler,
   search, repair, or answer callback exists in the runtime.
4. **Late-query reader:** after execution, a causally masked reader receives
   every declared terminal-state field and late-query residuals.

Hard transactions are bit-exact in the forward pass, while their
pre-discretization probabilities remain available for corrective gradients.
The two persistent status bits encode four always-visible dispositions:
`OPEN`, `ANSWER`, `ABSTAIN`, and `REJECT`. Deployed state rejects continuous
value channels, non-one-hot codes/types, non-binary control state, and
relation ledgers above 256 edges. Any terminal disposition freezes subsequent
structural writes. The production packet has 64 slots, 16 typed relation
roles, and 256 categorical symbols. It represents ordered hyperedges and
multi-byte values by reifying their role/value nodes rather than collapsing
them into one scalar label.

## Parameter Receipt

| Component | Parameters |
|---|---:|
| World compiler | 21,466,377 |
| Command-conditioned reactor | 29,757,217 |
| Late-query reader | 16,474,177 |
| Added architecture | **67,697,771** |
| Protected Shohin | 125,081,664 |
| Complete system | **192,779,435** |
| Remaining below 200M | **7,220,565** |

The protected step-300k checkpoint hash matches
`211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6`
and strictly loads with zero missing or unexpected tensors.

## Exact Offline Ontology Mechanics

| Ontology | Theories | Held-out structure | Independent comparisons |
|---|---:|---|---:|
| Typed Horn closure | 20 | entire ontology in leave-one-out fold | 7,560 |
| Typed term rewriting | 15 | 8 unseen rule combinations | 960 |
| Guarded resource process | 60 | 36 unseen length-2/3 programs | 174,960 |
| **Total** | **95** | rules, compositions, roles, halt/deadlock | **183,480** |

All independent oracle comparisons agree. The boards contain four opaque
renderers and 352 exact singleton, ambiguous, contradictory, or coherent-
alternate evidence episodes. These are assessor mechanics, never candidate
runtime imports.

## Four-Process Custody

State crosses processes only through an immutable safetensors file with seven
allowlisted tensors and three metadata fields. It contains no source text,
token offsets, source hash, residual cache, KV cache, parser state, executable
callback, or assessor product.

The serial test executes:

```text
compiler -> physical compiler-directory deletion
executor -> physical executor-directory deletion
late query -> physical query-directory deletion
independent assessor
```

World/compiler artifacts are absent before execution. Query inputs do not
exist during execution. Expected answers do not exist until candidate exit.
Every output is write-once and read-only.

## Frozen Seven-Variant Matrix

The three boards now materialize real alpha/reorder, alias-split, relation-
reification, type-twin, execution-semantics-twin, and ambiguity-deletion
transformations rather than relabeling renderer variants. The joined matrix
contains 3 folds, 24 held-out theories, 168 source worlds, 384 canonical
challenges, and 2,688 primary executions. It audits 1,472 invariant cases,
750 separating outcomes/directives, 384 abstentions, zero family-label leaks,
24 disjoint theory hashes, and 2,688 unique row hashes. Payload SHA-256 is
`d1904b54a0fab8e59cfcb0b0dd464f5c8778e5b828907028ec8614aeae76d5d5`.

## Causal Continuation Contract

The frozen architecture source adds:

1. `CausalETTREpisodeRunner`, with independent batch rows and explicit
   `WORLD`, `COMMAND`, and `QUERY` reset boundaries;
2. reset-safe token targets plus initial-packet, free-running terminal-packet,
   transaction, initial/terminal equivariance, commit/halt, sparsity, and
   anti-bypass losses;
3. a canonical continuation batch and immutable manifest that reject live
   writers, family labels, malformed geometry, and snapshot drift;
4. disjoint protected-base and architecture optimizer groups, with base
   freezing and an embedded WSD update cursor;
5. atomic no-replace checkpoints covering exact model/optimizer/schedule/RNG/
   data state and protected-checkpoint provenance, admitted only at a complete
   optimizer and between-episode boundary; and
6. a bounded accumulation/update component that has no filesystem, shard,
   launcher, or network access.

The complete ETTR/cross-ontology architecture and custody inventory passes
**174/174**.
Reset-boundary tests include interior segment starts, exact native
Muon/AdamW resume, and next-update equivalence after restore. A
degree-preserving edge-swap falsifier holds every per-slot in/out relation
count fixed while changing endpoints; both the reactor and query reader change
their outputs. The previous degree-summary architecture was invariant to this
necessary distinction and is superseded.

The training boundary additionally proves exact optimizer/model parameter
identity, rejects mutable or forged causal targets, binds batches to immutable
manifest and dataset hashes, prevents scheduler steps past the frozen horizon,
and uses pre-discretization policy probabilities for hard-forward
supervision. Redundant per-segment LM losses are disabled in the composite
train step. Commit `8cac6ce5a97597ab8a6cd47eda0aa4924590a762` additionally
binds `output.terminal_state` to offline terminal packet targets and proves
that this loss backpropagates through the recurrent reactor into the compiler.
The packet loss normalizes jointly over initial and terminal support, so the
existing family weight does not silently double. This closes the prior
independent-initial-packet/transaction supervision gap.

Commit `5771c64` closes the paired packet/command anti-bypass gate with
nontrivial factorial interchange. Each immutable 2x2 rectangle contains two
semantic WORLD factors and two semantic COMMAND factors, but every equivalent
factor is rendered through different raw token bytes. WORLD-equivalent rows
must have identical complete initial packet targets, WORLD factors must differ
in initial packet state, and all four row/column contrasts must change the
terminal target. A WORLD intervention takes a packet compiled from a different
rendering of the counterfactual WORLD factor while holding the COMMAND
semantics fixed; the COMMAND arm performs the orthogonal operation. Every
source row differs from its target row, and every terminal/transaction label
is gathered from an immutable factual corner. The rejected token-identical
version is not retained because deterministic zero-dropout execution reduced
it to a factual permutation.

The data boundary now replays every labeled generic transaction from the
initial target packet and requires exact agreement with the terminal target,
including value/type codes, relations, activity, root, edge capacity,
commit/halt status, and terminal disposition. Initial status is bound to the
compiler's open reset. Right-padded trajectories are legal only when their
last valid step commits or halts, ensuring later fixed-width runtime steps are
frozen. Factual and both intervention terminal states pass the deployed
discrete-state validator under `eval()` and `hard=True`. WORLD and COMMAND
losses, support receipts, and isolated gradient paths are separate.

## Cross-Ontology Hybrid Receipt

The frozen hybrid board contains exactly three couplings:

- arithmetic result selects a rewrite location;
- Horn relation selects a guarded resource operator; and
- resource state selects a Horn query.

Each coupling has 16 factual/counterfactual cases. Independent executors agree
on **96/96** executions, and interventions change both the coupling signal and
final output on **48/48** cases. Candidate payloads expose no ontology label.
Payload SHA-256:
`d155f868494f9379b214028c8d7475cc2cde08192c9b3a5bbdea5a73b29f98e2`.

## Remaining Architecture Gate

- Immutable architecture commit:
  `29d294f53085a254e1bf056abd7c388a5fe7ca95`.
- Job `705188` failed closed on `evc33` before checkpoint/model execution
  because CUDA was unavailable despite allocation. The node is excluded.
- Job `705192` predates the factorial objective and cannot qualify current
  source. It completed cleanly on `evc29` in 6m50s and is retained only as a
  stale systems receipt.
- Exact documentation descendant
  `4ca7366eb5102b1f51e16c2166717ec5e02448cb` is checked out in a detached,
  clean Newton worktree. Exactly one fresh isolated BF16 H100
  eager-versus-compiled memory/throughput profile, job `705213`, completed on
  `evc25` in 10m54s. Report SHA-256 is
  `374edd8e41d143274fab645ec15923ec9a078ddef301442b982b37f7ac9dd408`.
- The schema-v3 profiler must execute factual episodes, both intervention
  arms, the complete composite objective, backward, and Muon/AdamW update
  from matched eager/compiled initial parameter hashes.
- Record strict checkpoint load, nonzero architecture gradients, peak memory,
  measured throughput, compiled-arm status, and unchanged checkpoint hash.
- If profiling finds an OOM or systems defect, revise only the architecture
  implementation and repeat the same synthetic gate.

The resource geometry passed: eager peak allocation was 3.207 GB at 3,916.16
encoded tok/s and compiled peak allocation was 2.763 GB at 6,629.96 encoded
tok/s, a 1.693x throughput ratio. Both arms had finite losses, finite nonzero
architecture gradients, frozen-base zero gradients, matched batches,
matched initial parameters, and an unchanged checkpoint hash. The overall
gate remains closed because the query reader's sampled parameter delta was
zero despite more than 7.94M nonzero gradient elements. The 4,096-value probe
currently exhausts its budget on the first component tensor and must be
replaced by deterministic cross-tensor sampling before reprofile.

Hostile review also identified the remaining causal gap: WORLD/COMMAND
intervention terminal states are supervised but are not yet required to
produce the corresponding late-query answer. Final architecture qualification
therefore requires a matched-prefix, factual-corner query-binding objective
and negative controls in addition to the corrected update receipt. Only after
those gates can the architecture be called technically ready for the user's
later pretraining decision. Capability still requires future pretraining,
matched causal controls, unseen-ontology qualification, and post-training.
Only the user may lift the continuation-pretraining hold.

## Matched-Prefix Causal Query Binding

Commits `f263616` and `19b74f2` close the missing consumer-side path without
adding parameters. Each factual row declares one query read position, but no
answer label field. Within every immutable 2x2 rectangle:

- the read position is identical across all four corners;
- query tokens and masks are exactly identical through that position;
- each WORLD edge changes the factual next-token target; and
- each COMMAND edge changes the factual next-token target.

The intervention runner receives only target row indices. It reads each
intervention terminal state with the target corner's query sequence and
returns logits gathered at the pre-answer read position. Labels are gathered
later, inside the data contract, from immutable factual shifted-token targets.
The foil is the corresponding factual row with the intervened factor
unchanged. Because correct and foil predictions receive the exact same prefix
but require different factual labels, the frozen query-only LM path cannot
satisfy the objective.

WORLD and COMMAND query-binding losses are separate. Each combines
correct/foil classification with a directional difference-in-differences
margin, with explicit pair support and margin-satisfied receipts. The runner
signature contains no target or answer tensor. This proves a categorical
consumer-side causal mechanism only; multi-token autonomous reasoning remains
unproven.

The parameter update probe is also corrected. It now samples deterministic
coordinates across every trainable tensor rather than spending the entire
budget on the first tensor. The integrated ETTR/cross-ontology inventory is
**193/193** passing. A fresh exact-source H100 profile and the frozen
qualification/control matrix remain required.

## Sealed Continuation and Optimizer Custody

The final architecture source is
`cf568182b75e865ddce2bb739fd42ff8d450c317`. It replaces mutable, partially
bound packet admission with a canonical continuation manifest and immutable
packet-sufficiency index. Separate train and validation context entries,
full-batch payload digests, payload hashes, row/context cardinalities, and the
combined dataset digest are bound into the manifest. Verification uses sealed
independent admission sets; changing visible frozen-dataclass fields after
construction cannot alter the admitted train or validation population.

The training step accepts only train-split batches admitted by that exact
manifest. Optimizer ownership is recomputed from live parameter groups,
survives serialization of the complete trainer, and is permanently fail-stop
after a partial update. Compiler, reactor core, command projection, and query
reader are disjoint causal-attribution groups.

Profiling schema `shohin-ettr-h100-profile-v5` isolates causal gradient probes
from full-objective timing and memory, uses eager BF16 for attribution, and
counts exact encoded work as `WORLD + 2*COMMAND + 3*QUERY`. The exact source
passes **209/209** integrated tests plus Ruff, byte compilation, shell syntax,
and diff checks. The first exact-source hardware attempt, `705281`, failed
before model execution because CUDA was busy/unavailable on `evc43`; this is
a node defect and provides no architecture evidence.

Replacement `705285` completed on `evc30` in 11m42s. The schema-v5 report
SHA-256 is
`ea16f5b2c4da382edc288cbcfeb9a0e14590ddcf10debe013f5f5834d928d75f`.
The exact report is preserved at
`artifacts/r12/ettr_profile_cf56818_schema5_sealed/report.json`.
Both H100 BF16 arms completed the full factorial objective, backward, and one
architecture optimizer update with finite losses and all gates true. They
match batch, initial-parameter, and parameter receipts. The frozen base has
zero gradients and zero sampled delta. Every architecture group has finite
nonzero gradients and nonzero sampled delta, and the separate causal
attribution receipt confirms the intended WORLD and COMMAND paths while
detached-state controls cut upstream gradients to exact zero.

Eager full-objective execution measures 5,108.80 encoded tok/s at
3,750,596,608 peak allocated bytes. Compiled execution measures 8,771.94
encoded tok/s at 3,143,077,888 bytes: 1.7170x eager throughput and 0.8380x
eager peak allocation. The checkpoint remains byte-identical before and after,
no shards were read, and no model state was written.

This closes the architecture implementation and hardware gate. It does not
establish learned reasoning. The frozen control matrix must be executed after
user-authorized training, and continuation pretraining remains under the
user's explicit hold.

## Frozen Qualification Harness Completion

The learned-capability control matrix now has an executable assessor boundary
in `train/ettr_qualification.py`. The harness accepts only hard deployed
terminal packets and factorially matched query groups. It binds exact state,
query, target, factor, paraphrase, and control-index bytes into a batch
SHA-256, physically deletes the answer and all suffix tokens before forward,
and never gives a target tensor to the candidate model.

One invocation produces sealed treatment, query-only, zero-reader,
shuffled-state, wrong-WORLD, wrong-COMMAND, and query-twin readouts. Detached
post-forward scoring reports correct-query and deliberately wrong-query
exactness, target-derangement exactness, complete-state query-twin exactness,
packet-group exactness, packet-effect rows, and query-sensitivity rows.
Reassociation with another same-shape batch and mutation of copied labels
fail closed.

The hostile harness suite passes 19/19. The expanded complete ETTR,
checkpoint, optimizer, source-deletion, and cross-ontology inventory passes
240/240 in 156.18 seconds with clean Ruff, byte compilation, and diff checks.
Physical source deletion remains independently enforced by the four-process
custody suite. Packet-field sufficiency ablations are intentionally deferred
to equal-budget training arms because an evaluation-time zeroing intervention
would test a different claim.

Independent hostile review found and closed three P1 bypasses before final
handoff. Readout logits are cloned, model-bound, and tensor-receipt sealed;
mutating or substituting them invalidates scoring. Every candidate input is
cloned and hash-checked around its forward, while exact batch and model-state
receipts must remain unchanged across all arms. Finally, every shuffled,
wrong-WORLD, and wrong-COMMAND donor must change the factual target, and the
scorer requires the correct donor counterfactual output rather than treating
arbitrary prediction change as evidence.

The final trust-root repair makes public evaluation atomic and withholds all
logits. It requires preregistered model and semantic-manifest hashes, includes
every named child module's class implementation in model identity, rejects
child method overrides and hooks, and randomizes all seven arm calls with an
execution-order receipt. Final independent re-review found no remaining
P0/P1 through the supported public API.

## Direct WORLD-COMMAND-QUERY Qualification Completion

The earlier hybrid and factorial receipts established useful geometry, but
they did not by themselves constitute a direct three-stage learned-
qualification board. That final gap is closed by
`pipeline/ettr_factorial_qualification_board.py`.

The new frozen board defines direct Horn, typed-rewrite, and guarded-resource
programs. WORLD establishes an initial state, a separately packaged COMMAND
arrives after sealing and transforms that state, and a separately packaged
QUERY arrives only after execution. Each ontology contains an exact 2x2 WORLD
x COMMAND rectangle, two semantic queries, and two paraphrases. This yields 12
terminal packets and 48 query rows. Primary and independently implemented
oracles agree 12/12. All 24 WORLD and all 24 COMMAND semantic edges change the
answer, and all 12 packets expose different targets for the two query
semantics.

The board payload SHA-256 is
`18686ff7f0476b5a4432830f2a301f693833cf867656d3997a010cf17bb0149a`.
Its WORLD, COMMAND, QUERY, and assessor packages are separately receipt-bound.
Candidate surfaces contain no targets, oracle outputs, or ontology labels,
and hostile tests simulate physical deletion of prior-stage packages before
each successor runs.

`train/ettr_factorial_qualification.py` converts externally produced hard
terminal states into the existing sealed learned-qualification manifest and
batch. It binds every packet to the exact board and model receipts and freezes
matched wrong-WORLD, wrong-COMMAND, shuffled-state, query-twin, and target-
derangement controls. The fresh executor was repaired to consume an immutable
post-seal COMMAND through the hash-bound protected base while remaining unable
to access WORLD, QUERY, tokenizer, or assessor inputs.

Hostile review caught and rejected an earlier self-attestation gap before
publication: a caller could bind handcrafted valid packets to an arbitrary
model hash, and the package-deletion test was not the same execution that
produced the admitted packets. The supported path now requires externally
preregistered complete-model, execution-manifest, compiler-receipt, and
executor-receipt hashes. That chain binds exact board/package, token,
configuration, checkpoint/step, compiler/reactor weight, parent-state,
terminal-state-file, and canonical-state-tensor identities. The integrated
four-process test derives the actual compiler/command/query inputs from frozen
board rows and admits the terminal state only through this receipt chain.
Checkpoint loads are `weights_only=True`.

This closes the architecture-side geometry and hash chain, but not the final
qualification trust root. Independent rereview requires a custody signature
whose private key is unavailable to candidate processes, a canonical
raw-package-to-token transformation receipt, a recomputable receipt proving
that the exact base/compiler/reactor/query-reader components constitute the
complete evaluated model, and inclusion of the late-query process in that
signed chain. It is not a neural score and does not imply that the currently
untrained ETTR additions reason.

The trainable architecture is complete; signed qualification custody remains
open as described above. This does not show that the untrained ETTR additions
reason. Learned promotion still
requires user-authorized training followed by this frozen matrix on unseen
ontologies, depths, compositions, and renderers.

Current decision:
`ettr_architecture_and_control_mechanics_complete_untrained_pretraining_held_capability_unproven`.
