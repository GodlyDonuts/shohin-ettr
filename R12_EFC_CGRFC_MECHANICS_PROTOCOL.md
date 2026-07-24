# R12 EFC Conflict-Gated Reentrant Functor Compiler Mechanics Protocol

## Status

Architecture-mechanics development only. This document does not authorize a
neural fit, development or confirmation access, a native-reasoning claim, or
continuation pretraining. The user pretraining hold remains absolute.

The implementation lives on the isolated `codex/cgrfc-mechanics` worktree
until the sealed QFCR experiment is consumed. Nothing in this branch changes
the public QFCR source/authorization chain.

## Evidence-constrained target

Shohin's bounded components already establish that:

- unseen categorical laws can be composed exactly when the representation is
  supplied;
- source-deleted categorical machines can be executed exactly;
- opaque operations can be induced on a narrow controlled board; and
- a readable late residual exists.

The unrestricted model remains weak, and a large residual motor did not turn
readable state into a reliable autonomous loop. The narrowest unresolved
transaction is therefore:

> Bind physical source occurrences to episode-local state, action, and
> observer identities, and jointly commit their noncommuting transition
> semantics into one query-blind source-deleted machine.

CGRFC targets this transaction. Here, "conflict" means categorical
disagreement among source-derived sections and the provisional global
machine. The current implementation does not claim a mathematical cocycle or
triple-overlap holonomy. It is not an external executor, a prompt scratchpad,
a chain-of-thought renderer, or a claim that adding parameters alone creates
reasoning.

## Architecture

### Attached source phase

1. The frozen 125,081,664-parameter Shohin trunk provides read-only local
   residual features.
2. The maximum 512-wide witness encoder parses generic records and opaque key
   occurrences without a source grammar oracle. JASEC uses a two-view source
   boundary: every opaque-key span in the model view is replaced by the fixed
   two-byte anonymous token `d1`, while the raw key bytes and equality
   partition remain available only to custody and final copying. Frozen Shohin
   residuals are computed from this anonymous view. Literal key spelling,
   codec, and byte width therefore have no trainable path.
3. The custody-bound witness API remains unchanged. A 32-dimensional
   recoding-invariant record diagnostic is assembled from type evidence and
   symmetric confidence/entropy summaries of answer, key, and slot transport.
4. Direct row incidence is emitted from the witness transport. A
   5,386,721-parameter record-local calibration branch applies a learned
   category-shared nonlinear map to each record's destination and answer
   distribution before aggregation. The shared scalar basis preserves
   state/answer recoding equivariance, while the nonlinear pre-aggregation
   path makes local claims algebraically distinct from the provisional
   machine. This is learned self-calibration of the same evidence, not an
   independent source measurement. The raw wrapper supplies zero closure
   incidence until a genuinely independent behavioral measurement exists.
5. A provisional anonymous machine is represented as 40 categorical rows:
   - `3 * 8 = 24` transition rows with eight destinations; and
   - `2 * 8 = 16` observer rows with four supported answers.
6. `ConflictGatedReentrantRevision` runs four tied correction cycles.

No query is present during these steps.

### Two active contradiction channels and one reserved channel

For record `r`, machine row `l`, and supported category `c`:

1. **Section-consensus disagreement**

   The weighted variance among records assigned to the same row:

   `V_lc = E_r[(q_rlc - E_r[q_rlc])^2]`.

   This detects incompatible local sections that claim the same anonymous
   semantic object.

2. **Local/global disagreement**

   `D_lc = E_r[q_rlc] - p_lc`.

   Rows with zero direct incidence contribute exactly zero, not a phantom
   negative claim. Unlike the rejected first draft, `q` is now produced by a
   separately parameterized record-local nonlinear calibration path rather
   than copied from the same distributions used to construct `p`. This is a
   calibration residual, not independent contradiction evidence, and must
   beat the graph-connected identity-calibration control to be attributed.

3. **Behavioral closure disagreement**

   `H_lc = E_r[h_rlc] - p_lc`,

   This channel is reserved and is bit-zero in the current implementation.
   It may be enabled only after a genuinely separate source-derived
   measurement exists; it must not be reconstructed from the provisional
   machine itself.

The deployed candidate never receives labels, oracle machines, verifier
feedback, late queries, or runtime autograd.

### Tied categorical revision

Every supported `(row, category)` cell receives a 16-scalar, recoding-
equivariant feature vector containing its gauge-fixed logit, probability,
direct claim, signed and absolute residual, record variance, closure claim and
residual, evidence mass, row entropy and maximum, row type, cycle index,
previous update, and contradiction magnitude.

A shared 960-wide controller uses:

- a cell feature encoder;
- a record-feature encoder;
- row, category, machine-type, and source-record contexts;
- one tied `GRUCell`;
- one tied correction head; and
- one tied positive bounded step head.

The same parameters are replayed for all four cycles. No state/action/
observer/category coordinate embedding is permitted. State, action, observer,
answer, and record permutations must commute with every soft cycle and the
hard seal.

The categorical cotangent is raised by one of two parameter-free metrics:

- **Euclidean quotient:** subtract the supported row mean; or
- **quotient Fisher:** divide by supported probability, subtract the row
  gauge, and bound the row maximum.

QFCR decides whether Fisher geometry may be the named treatment. If QFCR does
not attribute a causal advantage, Euclidean quotient revision is the primary
arm and Fisher remains a diagnostic.

### Tied source-perception reentry

Signed terminal first-pass probability corrections, their magnitudes, and signed
squared magnitudes are encoded by one shared bias-free cell map and pooled
symmetrically over categories. Zero terminal correction therefore produces
exactly zero source feedback. Final direct and closure incidences route those
row contexts back to generic source records. A rank-128 bias-free decoder maps
each record context into a bounded correction of the detached frozen
1,728-dimensional trunk feature channel and broadcasts it only over that
record's physical source span.

The exact same witness encoder, claim adapter, and revision core are then
replayed. No encoder layer or controller is cloned. Only the second pass may
be sealed. This changes the traditionally fixed one-pass transformer
interface: global categorical contradictions can causally revise local source
perception before commitment, while the frozen Shohin trunk and detached
runtime remain unchanged.

### Detached query phase

The compiler hardens only:

- active anonymous state/action/observer fields;
- action-next rows;
- observer-answer rows; and
- exact copied opaque key bindings.

All source bytes, tokens, residuals, record features, local claims, incidence,
recurrent hidden state, contradiction tensors, optimizer state, and training
targets must be destroyed before a late query is generated in a new process.
The detached query parser and fixed categorical runtime may read only the
sealed machine and copied key object.

The isolated standard-library worker
`pipeline/episode_functor_sealed_worker.py` consumes only the exact
1,536-byte deployed machine and an already structured late opaque-key query.
A fresh-process test deletes the source file, mutates every parent machine/key
tensor after wire serialization, launches Python with `-I` from an empty
temporary working directory, and recovers the unchanged expected final state
and answer. The worker imports neither PyTorch nor any compiler/witness module.
This proves the fixed wire execution boundary.

The neural late-query boundary is now implemented separately in
`pipeline/episode_functor_detached_query_worker.py`. The default parser is a
standalone raw-byte `width=160`, two-layer, five-head transformer with no
frozen-feature input. This corrects the previous paper-only allocation, whose
parameter count included a projection from frozen residual features that the
deployed process could not obtain under its stated input contract. Parser
weights use pickle-free `safetensors`; a canonical manifest binds the complete
constructor geometry, parameter count, tensor names/shapes/dtypes, weight byte
count, canonical named-tensor state SHA-256, and weight-file SHA-256. The
source-attached system constructor requires this package receipt, recomputes
the parser state digest and actual parameter storage, and stores
the exact manifest/state/weight commitments without retaining the parser
module. A separate canonical execution authorization binds that preregistered
parser, the exact machine-wire SHA-256, source-compiler parameter count,
protected Shohin count, and complete under-200M count. The worker requires the
external authorization SHA-256.

JASEC additionally loads the source tokenizer from an inode-bound,
non-symlink regular artifact, verifies its artifact and canonical runtime
SHA-256 values, and retains that fixed tokenizer in the source-attached
process. After literal anonymization the system tokenizes the anonymous
payload internally. Callers cannot provide token IDs or byte offsets to the
joint compiler. This prevents raw-source tokens or unrelated token IDs from
supplying the frozen Shohin residual channel.

Manifest, authorization, and weight reads use one `O_NOFOLLOW` descriptor and
deserialize the same immutable bytes that were hashed; weights are mode 0600,
bounded before construction, pickle-free, and finite-checked. The worker
reconstructs only the hard machine/key object from the sealed wire, requires
exact `8/3/2` primary geometry, round-trips the wire byte-for-byte to reject
nonzero reserved channels, scans the late raw query, runs the verified parser,
hardens its categorical parse, and executes the fixed machine. The parser now
imports only source-independent runtime constants; a fresh-process denylist
rejects any transitive compiler, witness, pointer-source, or constrained-
transport import. Mutation, substitution, reserved-byte, cap, and no-clobber
tests fail closed.

The source-attached
`ConflictReentrantEFCSystem` now retains no query-parser child and fails
closed on `parse_query`, `forward`, and `execute_sealed`; it may only export a
deployed wire. The detached parser's parameter count remains included in the
aggregate architecture receipt. A fresh process now instantiates and executes
the exact bound parser after source deletion. A qualification orchestration
test must still prove that the compiler process has exited, not merely deleted
one source path, and must confine the worker away from the repository before
end-to-end results become promotion-admissible.
The qualification orchestrator must additionally prove that package export
and system construction preceded source ingestion.

## Joint Assignment-Semantics Equilibrium Compiler (JASEC)

### Why the sequential architecture is insufficient

PK-NPAS followed by CGRFC leaves the two hardest latent variables only
indirectly coupled. The path controller commits a binding before the
categorical revision core commits semantics. A wrong binding can therefore
produce a self-consistent wrong machine, while a semantically informative
provisional machine cannot directly revise the binding that generated it.
The 5.39M claim-calibration branch further recalibrates the same record
evidence rather than introducing an independent contradiction measurement.

JASEC is a separately named replacement hypothesis. It removes the path
controller, claim calibrator, source-reentry branch, and categorical revision
core from the candidate source compiler. It retains the maximum witness
encoder and alternates binding and machine revision inside one tied
four-cycle equilibrium.

### Machine-to-assignment edge

The parameter-free physical nerve exposes one signature for every physical
key:

- states: direct observations (8), one-step transition observations (24),
  and ordered two-step destinations (72), total 104;
- actions: direct transitions (64), transition observations (64), left
  ordered paths (192), and right ordered paths (192), total 512; and
- observers: state-answer behavior (32).

`machine_behavior_signatures` constructs exactly the same 104/512/32
signatures from the current soft categorical machine. Negative
Jensen-Shannon divergence produces an anonymous slot-to-physical-key
compatibility tensor. Physical-key recoding permutes only its final axis.
The action signature keeps `a;c` and `c;a` in different channels. The
one-step-only control excludes both ordered-path blocks from its path mass
while retaining one-transition observer evidence, so it removes composition
without erasing the one-step relation it is intended to preserve.

The assignment half-step appends machine compatibility, its masked
row-centered residual, and its absolute residual to the 24 PK-NPAS cell
features. A tied `27->600->600` encoder and symmetric slot/type/key contexts
emit a bounded gauge-fixed correction before the existing balanced Sinkhorn
projection.

### Assignment-to-machine edge

The revised assignment reconstructs record-to-machine relation evidence
inside each cycle. Every supported machine cell receives 16 anonymous
features containing current and reconstructed probabilities, quotient log
coordinates, signed/absolute residual, entropies, maxima, support/type/cycle
indicators, previous direction, assignment confidence, and semantic
agreement.

A tied `16->960->960` encoder, symmetric row/type/global/record contexts, and
an exactly contractive recurrent cell emit a supported gauge-fixed direction
and bounded step. The candidate machine is updated by a convex mixture of its
current distribution and the revised supported softmax, preserving every
categorical row invariant.

The recurrent path uses a dense input transform and a bounded diagonal
recurrent term `0.9*tanh(d)`. The hidden update
`h' = 0.75 h + 0.25 tanh(Wx + 0.9*tanh(d)*h)` has a recurrent hidden-state
Jacobian contribution strictly below 0.975 before downstream feedback.
The complete closed-loop Jacobian still requires measurement; this local
bound is not a convergence claim.

### Directed causal controls

- **M->A cut:** machine compatibility is replaced by direct path
  compatibility plus a graph-connected zero residual.
- **A->M cut:** relation reconstruction, assignment-confidence context, and
  physical signatures used by the machine branch are all fixed to the
  initial projected assignment plus graph-connected zero differences.
- **Both cut:** both directed edges are removed with identical module
  execution.
- **Broken glue:** the existing side-marginal-preserving off-diagonal
  contraction replaces physical intermediate identity.
- **One-step only:** direct relations remain while ordered two-step and
  commutator channels are graph-zero.
- **Sign reversed:** both learned update directions are negated without
  retraining.
- **Open loop:** every module executes while both corrections are
  graph-connected zero.

The implementation is in:

- `train/episode_functor_joint_assignment_semantics.py`;
- `train/episode_functor_joint_equilibrium.py`; and
- `train/episode_functor_joint_compiler.py`; and
- `train/episode_functor_joint_system.py`.

The integrated source compiler seals a hard machine plus copied keys. A
separate exact-type system constructor now binds it to the adapter-free frozen
Shohin trunk and preregistered detached parser package, recomputes the live
aggregate parameter receipt, and retains no parser child. Trunk receipts are
invoked through the exact class implementation rather than instance dispatch.
Each attached forward emits a canonical compilation receipt binding a random
compiler-instance nonce, exact compiler tensor state, mutable execution
configuration, exact parameter count, ordered source hashes, physical-key
inventory, and all final equilibrium tensors. Sealing recomputes those
commitments and requires a nonserializable in-process object-identity
capability. A second private issuance registry binds each emitted seal to the
exact compiler deployment digest, wire hashes, and seal-receipt hash. The
sealed object itself retains neither the compilation receipt nor source
hashes. This rejects outputs from another or deep-copied compiler instance,
mutated configuration/state/output, and altered machines even when an
attacker recomputes the public wire hash.
The system rejects raw-wire authorization entirely: authorization consumes
only a reverified `SealedJointMachine`, rechecks the exact non-subclassed
`FrozenShohinTrunk`, protected checkpoint hash, zero adapters/trainable parent
parameters, current compiler state, row, wire hash, and aggregate cap. The
compiler parameter count is recomputed from live modules for every parameter
receipt and authorization; it is never trusted from the constructor cache, and
the strict aggregate cap is reapplied after any module change. A
qualification orchestrator has not yet proven process-A death and process-B
repository confinement. No neural fit is authorized.

### JASEC parameter receipt

| Component | Parameters |
|---|---:|
| Frozen Shohin checkpoint | 125,081,664 |
| Maximum gauge-invariant witness encoder | 44,658,064 |
| Tied JASEC equilibrium | 19,013,524 |
| Standalone detached query parser | 748,033 |
| **Complete JASEC mechanics** | **189,501,285** |
| **Headroom below 200M** | **10,498,715** |

The 19,013,524-parameter equilibrium replaces 29,376,430 parameters of
sequential path control, same-evidence calibration, source reentry, and
categorical revision. This is an architecture correction, not parameter
minimization: the saved 10.50M remains reserved until a causal qualification
result identifies a specific under-capacity component.

Current focused JASEC mechanics are 54/54 passing. They cover exact
104/512/32 behavior-signature geometry, diagonal machine/key identification,
an independently loop-constructed exact physical/machine match with zero
objective gradient, an independent 40-record match through the actual
physical-key nerve, invalid/zero-evidence exclusion, physical-key
equivariance, complete state/action/observer/answer gauge equivariance through
every tied recurrent cycle, noncommuting order sensitivity, explicit
assignment-controlled nerve signatures, balanced assignment transport,
supported machine normalization, finite nonzero complete backward flow,
directed cut gradients, open-loop zero updates, provenance-bound hard sealing,
cross-compiler/deep-copied-compiler/mutated-state/mutated-output/configuration
rejection, forged-machine rehash rejection, source-hash removal at the sealed
boundary, forged raw-source hash/key-custody rejection, exact trunk-type
enforcement, rejection of raw-source frozen-trunk payloads, raw-wire
authorization rejection, internal anonymous tokenization, exact tokenizer
artifact/runtime loading, post-construction parameter-cap enforcement, and the
exact parameter receipt. Full bijective key renaming is tested across all
trainable tensors, gradients, recurrent cycles, and the hard machine. A
separate cross-codec test changes hexadecimal keys to decimal keys of differing
byte widths and obtains the same model view and machine while preserving
different copied keys. An equality-partition negative control changes the
assignment, proving that anonymity does not erase physical identity.

The combined CGRFC/JASEC/rejected-CMRL/SLRA/detached-runtime focused suite is
170/170 passing, but the CMRL tests are tensor-mechanics evidence only and do
not admit that architecture.
The broader isolated-worktree EFC suite is 633 passed and 26 failed. Twenty-
two failures are the known runtime-source custody checks that deliberately
reject execution from `/private/tmp` rather than canonical main. The other
four require large artifact/checkpoint files that are intentionally absent
from the isolated worktree. No expanded-suite failure enters a new JASEC
module or changed physical-nerve path.

### Counterfactual Machine-Repair Lattice negative result

The 9,618,567-parameter CMRL prototype is rejected before fit. Hostile review
showed that its machine-shaped evidence tensors were exact target machines in
disguise: one candidate feature copied the target probability directly. Its
leave-one-out control retained that target through an invertible normalized
encoding, while its observational twin made every candidate identical and
removed the choice channel. Fixed-cycle mode failed to disable row gates,
unsupported-cell custody was absent, and the terminal halt output did not
participate in the mixture.

The 16 focused tests still establish finite mechanics, exact parameter
accounting, and categorical equivariance, but they do not establish a valid
causal architecture. CMRL is not integrated, its 9,618,567 parameters are not
part of the JASEC receipt, and no neural run is admitted. JASEC therefore
remains 189,501,285 complete parameters with 10,498,715 genuine headroom.

The only surviving design direction is a newly named source-sealed
intervention mechanism whose compiler-owned constraints are incomplete by
construction, withhold the row under repair, and contain only observed
trajectory consequences rather than a complete target machine. Its matched
control must preserve both compute and candidate choice while breaking only
the intervention/consequence pairing. The complete rejection record is in
`R12_EFC_COUNTERFACTUAL_MACHINE_REPAIR_LATTICE.md`.

That successor is preregistered, but not implemented or admitted, in
`R12_EFC_SOURCE_SEALED_INTERVENTION_CONSEQUENCE_PROTOCOL.md`. It replaces the
invalid leave-one-out control with a precommitted random derangement whose
permutation tensor is conjugated under category recoding. Direct source
compatibility remains identical in treatment and control; only the association
between a candidate and its downstream consequence packet changes. Its
provisional 9,641,096-parameter receipt would place JASEC at 199,142,381
complete parameters with 857,619 headroom, but no parameter is counted until
the structural no-leak and zero-parameter oracle gates pass.
The current EFC board admits only the Source-Law Residual Alignment (SLRA)
successor: it hides one cell
per action permutation and balanced observer and states those completion laws
in the source. Candidate interventions may be scored against those public laws
without a target table. The board does not admit behavioral-consequence
features because it lacks independent source-visible trajectories; rolling the
complete target table forward would recreate the rejected leak. A separate
partial-declaration plus independently observed behavior board would need its
own source freeze and custody protocol.
SLRA has two claim levels: a fixed tensor implementation is a valid
architectural law executor, while native law interpretation requires JASEC to
compile an unseen source law into the sealed constraint operator. The
derangement control can isolate use of the supplied aligned residual channel,
but it cannot rule out recomputation from the shared visible table and law.
The zero-parameter SLRA information gate is now implemented and passes 7/7
focused tests. It recovers every hidden transition/observer cell across eight
generated worlds, is exactly gauge equivariant, preserves residual multisets
under a conjugated hard derangement, rejects soft/identity transport and
ambiguous visibility, and has finite gradients on visible cells with zero
gradient through hidden placeholders. Four successive hostile audits found and
closed ten P1 defects. The repaired public boundary accepts only issuer-owned capabilities
constructed from raw source bytes and binds source hashes, issuer nonce,
tensor values, shapes, and dtypes. Copied, cross-issued, mutated, or stale
capabilities fail closed. Controls are independently derived for every
source/row from a precommitted seed and are also source-bound capabilities;
callers cannot supply a fixed permutation matrix. The 9,641,096-parameter controller
remains unimplemented and unadmitted.
Four successive hostile reviews are closed with no remaining P0/P1 in the
zero-parameter mechanic. The final hardening embeds the committed seed in
issuance bytecode, rehashes stored raw sources on every use, revalidates
capability schemas, and derives raw-key recoding bijections from
occurrence-aligned sources before conjugating the already-realized control.

## Parameter receipt

Current implemented large revision core:

| Component | Parameters |
|---|---:|
| Frozen Shohin checkpoint | 125,081,664 |
| Maximum witness encoder with zero-parameter direct provisional projector | 44,690,832 |
| Equivariant record-local claim calibration | 5,386,721 |
| Physical-key ordered-path assignment controller | 3,978,602 |
| Low-rank source-perception reentry | 352,641 |
| CGRFC tied revision core (`D=960`) | 19,658,466 |
| Instantiated standalone late-query parser | 748,033 |
| **Total added reasoning system** | **74,815,295** |
| **Current complete mechanics** | **199,896,959** |
| **Current headroom below 200M** | **103,041** |

Final integration must fit inside the 103,041 residual budget. The integrated
system computes this receipt from its live compiler and the detached parser
instance supplied at construction; no maximum-parser constant is added off to
the side. A fit is forbidden if the complete receipt is at least 200,000,000,
and the constructor rejects any oversized custom compiler/parser pairing.
The trunk must be the exact 125,081,664-parameter protected parent with zero
adapter parameters; integrated trunk parameters, not merely a parent-count
field, are included in the cap.

## Isoparametric controls

All controls share source bytes, source-feature tensors, parameters,
initialization lineage, update count, precision, persistent machine size, and
late-query parser.

1. **Causal incidence:** true record-to-machine routing.
2. **Deranged incidence:** route each record's preserved total mass through
   the leave-one-record-out complement distribution. This is record-order
   equivariant, excludes padding, and preserves per-record mass while
   destroying its own row ownership. Inputs with fewer than two positive
   records fail closed.
3. **Open-loop shadow:** identical recurrent compute with source claims,
   incidence, contradiction channels, record contexts, and source feature
   correction multiplied by graph-connected zero, so every parameter receives
   a finite zero gradient and identical optimizer/weight-decay treatment.
4. **Sign-reversed feedback:** a frozen-weight intervention that negates every
   signed direct, closure, path, and source correction while preserving
   magnitudes without coordinate-dependent masks. It is not a separately
   trained arm because a flexible controller could learn the inverse sign.
5. **Parameter-matched direct compiler:** a nonreentrant MLP/attention arm with
   equal parameters and FLOPs.
6. **Identity-calibration control:** the complete 5,386,721-parameter local
   claim branch executes, but its correction is multiplied by
   graph-connected zero before aggregation.
7. **Source-retained diagnostic:** never capability-admissible.
8. **Gold-machine/gold-query, predicted-machine/gold-query, and
   gold-machine/predicted-query localization ceilings.**

## Mechanics gates

Before any H100 fit:

- finite forward/backward in FP32 and with FP32 master parameters under BF16
  autocast;
- exact parameter receipt and tied-cycle parameter sharing;
- full state/action/observer/answer/record recoding equivariance;
- row-gauge invariance;
- unsupported observer categories remain unreachable;
- zero-incidence rows receive no phantom claim;
- open-loop output is bit-invariant to every source claim and record feature;
- malformed, nonfinite, negative-incidence, tied-hardening, or wrong-device
  inputs fail closed;
- sealing returns only `HardFunctorMachine`;
- source mutation after sealing leaves detached execution bit-identical; and
- no module invokes a host solver or runtime autograd.

The conflict compiler forbids solver-backed straight-through key assignment.
Qualification uses soft Sinkhorn transport; the final seal uses only unique,
untied, finite per-slot argmaxes and fails closed otherwise.

## Repaired reduced oracle-mechanics gate

The reproducible CPU audit
`pipeline/audit_episode_functor_cgrfc_oracle_mechanics.py` consumed 16
training and 64 unseen development machines. Each record carried one
row-local oracle section; every machine began with one fault per action and
observer relation at a wrong-versus-correct margin of 0.5. A
358,146-parameter reduced controller used four tied cycles and 101 AdamW
updates.

The original artifact used a padding-sensitive `flip(1)` derangement and its
attribution was withdrawn. The audit was rerun after replacing that control
with leave-one-record-out complement routing. Repaired result:

- decision: `oracle_mechanics_pass`;
- causal: `64/64` exact machines;
- deranged incidence: `0/64`;
- open loop: `0/64`;
- sign-reversed feedback: `0/64`;
- strongest-control causal advantage: `100` percentage points;
- finite complete first gradient: pass;
- report SHA-256:
  `9cf19fb7eac0674f5885af3ac26e7c84b495bebcb97bf65f05e37b58ccdb5b2c`;
- canonical payload SHA-256:
  `12673aa5f761182ca4016d67038e9c4b000e8d406e62122cbbafedfa6568b590`.

This restores only the narrow result that tied conflict feedback can learn
row-local oracle categorical repair and that true routing beats the
mass-matched derangement. It does not prove raw-source compilation,
unseen-family transfer, or reasoning.

### Exploratory raw-source smoke test

A non-gating CPU smoke test used a deliberately reduced 574,803-parameter
compiler (`width=48`, one encoder and decoder layer, `D=128`, two cycles),
four affine training worlds in grammar `(0,0,0)`, eight unseen affine worlds
in grammar `(1,1,1)`, zero external features, and 51 machine-loss-only AdamW
updates. Final unseen exact-machine accuracy was `0/8`; causal cell accuracy
was `17.1875%` and the zero-feedback two-pass control was `17.5%`.

This negative is not a large-lane architecture verdict. It demonstrates that
four machine-level labels do not identify the latent key/role binding or
support random-world generalization. Any qualification fit must use the
already frozen ordered binding supervisors and adequate world/family
diversity; machine-only toy loss is not an admissible training protocol.

`ConflictQualificationLoss` now joins the existing source-hash-bound
supervisor only after the candidate forward. It supervises second-pass key
assignment, record type, occurrence role, record answer, transition, and
observer tensors without exposing labels to the compiler. A reduced
four-renderer backward test gives finite gradients to every trainable compiler
parameter, including the correction-only source-reentry path.

## Qualification design

The first eventual fit must be a causal machine interchange board, not the
existing train-only HSC board:

- train families: random transitive `S8`, affine `F2^3`, and dihedral vertex
  actions;
- unseen development: regular `D4`;
- sealed confirmation: `Q8` and cube rotations;
- train depth 0--4, development 5--8, confirmation 9--12;
- repeated actions, equivalent words, same-bag order twins, and
  noncommuting reversals;
- five independent seeds;
- strict source deletion before independently generated late queries.

Primary gates:

- exact sealed machine at least 95%;
- transition and observer cells at least 99.5%;
- end-to-end at least 98% aggregate and 95% in every
  family-by-renderer-by-depth cell;
- every unseen family passes independently;
- compensated recoding and unaffected-query invariance 100%;
- donor-consistent transition/observer transplants at least 99%;
- equivalent words at least 99.5%;
- noncommuting twins at least 95%;
- causal incidence beats the strongest matched control by at least ten
  percentage points; and
- paired 99% confidence lower bound exceeds five points.

Passing establishes bounded, source-deleted G0--G3 systematic reasoning. It
does not establish unrestricted natural-language G4 reasoning.

## Implemented PK-NPAS architecture lane

The current compiler compresses physical-key occurrence evidence into one
global slot assignment before conflict revision. That discards a source-given
fact: a transition destination and a later transition source with the same
physical key are exactly composable. A mathematical review therefore rejects
learned holonomy at this stage and reserves **Physical-Key Nerve Path-Algebra
Synchronization (PK-NPAS)** as the next architecture treatment.

For record-to-physical-key role posteriors `Q`, type probabilities `tau`, and
answer probabilities `Y`, the source defines:

`E_T[u,v,w] = sum_r tau_T[r] Q_src[r,u] Q_act[r,v] Q_dst[r,w]`

and

`E_O[u,v,y] = sum_r tau_O[r] Q_state[r,u] Q_obs[r,v] Y[r,y]`.

Exact shared intermediate keys then define ordered path products:

`E_TT[u,v1,v2,w] = sum_m E_T[u,v1,m] E_T[m,v2,w] / d[m]`

and

`E_TO[u,v,o,y] = sum_m E_T[u,v,m] E_O[m,o,y] / d[m]`.

These contractions preserve intermediate identity, direction, action order,
and noncommutativity before anonymous slot projection. Here `d[m]` is the
mean incoming/outgoing transition evidence degree of physical state key `m`;
zero-degree and padded keys contribute exactly zero. The broken-glue control
replaces exact intermediate identity with a permutation-equivariant
off-diagonal transport coupling whose row and column marginals equal the
causal diagonal path masses. It therefore preserves both complete path-side
marginals, not merely one global scalar. Geometries without a feasible
positive off-diagonal coupling fail closed. The implemented shared
24-to-600 cell controller uses direct compatibility, explicit candidate-action
left and right path compatibility, transition-observation composition, and a
signed commutator channel to revise only slot-to-key assignment logits. The
same weights replay after both source-perception passes. It adds 3,978,602
parameters:

| PK-NPAS component | Parameters |
|---|---:|
| Shared `24 -> 600 -> 600` cell encoder | 375,600 |
| Shared `2400 -> 1200 -> 600` context mixer | 3,601,800 |
| Correction head | 601 |
| Gate head | 601 |
| **PK-NPAS total** | **3,978,602** |
| **Complete implemented mechanics** | **199,896,959** |
| **Remaining headroom** | **103,041** |

The parameter-free nerve/path-signature mechanics are implemented in
`train/episode_functor_physical_key_nerve.py`, and the learned controller is in
`train/episode_functor_physical_key_controller.py`. CPU gates prove
record-order invariance, physical-key permutation equivariance, active-slot
recoding equivariance, bit-zero correction under zero path mass, and
graph-connected zero gradient under open loop. The broken-glue control
preserves every direct transition and observation relation while changing only
the side-marginal-matched shared intermediate-key pairing. Both action keys and
the observer key remain physical through `E_TT`/`E_TO` contraction and are
projected only afterward. The one-step-only control still
computes ordered paths and zeros their signed channels only immediately before
the MLP. Noncommuting order channels do not collapse on the mechanics fixture.

The integrated lane is still mechanics-only and must prove exact
record/key/state/action/observer/answer equivariance, bit-zero correction with
zero composable path mass, at least a 20-point advantage over broken-glue and
one-step-only matched controls, ordered/noncommuting twins above 95%, and a
fresh-process source-deleted gain. Holonomy remains rejected until genuinely
distinct local frames make a nontrivial cycle defect identifiable.

## Kill rules

Kill or redesign CGRFC if:

- oracle claims cannot be sealed exactly;
- deranged incidence matches causal incidence;
- open-loop matches feedback within five points;
- contradiction energy falls without exact-machine improvement;
- a source-retained or answer-cache diagnostic explains the gain;
- recoding, transplant, equivalent-word, or noncommuting gates fail;
- the learned dynamics have an unstable local Jacobian around correct
  machines;
- unseen families collapse despite train interpolation; or
- the claim heads retain source/question information across the seal.

The architecture may use nearly the full parameter budget, but parameter
count is not a promotion criterion.

## Current verification

- focused changed-path JASEC/CGRFC/SLRA package:
  `170 passed`, `55` known warnings;
- broader stable EFC suite in the isolated `/private/tmp` worktree:
  `633 passed`, `26 failed`, `116` known warnings;
- 22 failures are runtime-semantic custody checks that deliberately reject
  `/private/tmp` instead of the canonical repository path; four require
  protected checkpoint/bundle artifacts that are intentionally absent from
  the isolated worktree;
- four successive hostile SLRA reviews found and closed ten P1 mechanics
  defects; the final mechanics review reports no remaining P0/P1;
- a separate claim-level audit nevertheless forbids SLRA residuals from neural
  confirmation because the unique zero residual is label-equivalent on the
  current one-hole board;
- `git diff --check`: pass;
- no neural weights, protected checkpoint bytes, pretraining state, or main
  QFCR custody files were changed.
