# DIVERGE-v0 Source-Sealed Factorized CSDC

Status: CPU mechanics and the token-role/source-copy compiler pass. The final
resource-corrected A--G gate fails its frozen broad-promotion rule on five of
five seeds. DIVERGE is retained only as a delayed-recovery mechanism for
ambiguity widths where factorization amortizes its runtime state overhead.

Date: 2026-08-05

## 1. Decision

DIVERGE is the ordered successor to the protected CSDC result. It is not a
restart of PCSD, FCPT, or dormant R10 ACAW/VSPT. PCSD and FCPT remain closed.
R10 contributes only audited mechanics: canonical commitments, exact
deduplication, monotone refinement, sticky fail-closed overflow, query-set
agreement, and disjoint accounting stores. Its affine tree, external source
references, evaluator, and unread score chain are not inherited.

The v0 claim is narrow:

> A source-sealed factorized packet can represent complete coherent worlds,
> execute shared typed consequences without fieldwise averaging, revise
> support only through verified evidence conflicts, merge only certified
> equivalents, and answer exactly when the late query is invariant.

The first gate is CPU-only. No learned compiler, neural scorer, long
continuation pretraining, or public benchmark is authorized until these
semantics pass exactly.

## 2. Causal boundary

The only allowed path is:

```text
WORLD
  -> source-only compiler
  -> sealed epistemic packet
  -> guarded execution and evidence refinement
  -> late QUERY
  -> exact answer certificate or abstention
```

After sealing, runtime state may retain source and evidence commitments but
not raw WORLD bytes, source token IDs, source residuals, source KV cache,
query fields, answers, gold assignments, or host repair state. Exact
enumerators and verifiers exist only as independent supervisors/assessors or
explicit counted controls. The candidate packet runtime may not call them.

## 3. Canonical packet schema

All fields are immutable, finite, length-bounded, canonically ordered, and
serializable as canonical JSON. Integer fields use exact integers; weights use
positive integer masses. Floating-point semantic equality is forbidden.

### 3.1 Shared typed state

`SharedState` contains:

- a fixed-size tuple of anonymous typed cells `(slot, type, value, live)`;
- a sorted tuple of directed typed edges `(source, relation, target)`; and
- a transaction index.

Slots and relation labels are episode-local integers. There are no permanent
semantic slots. State validation rejects duplicate slots or edges, invalid
references, unknown types/relations, and out-of-range values.

### 3.2 Fault-line variables

Each `FaultLine` contains:

- an episode-local integer ID;
- an ordered finite domain of at least two option commitments; and
- a compiler-provenance commitment.

The v0 cap is six variables, each with two through four options, and at most
64 represented complete worlds after hard constraints. Variable IDs and
option labels carry no cross-episode semantics.

### 3.3 Literals, guards, and hard factors

A literal is exact equality `(variable = option)`. A guard is a sorted
conjunction of noncontradictory literals; the empty guard is true. General
negation and arbitrary code are forbidden. A `HardFactor` contains an ordered
scope and the exact set of allowed tuples over that scope. The conjunction of
all factors defines initial support.

A nogood is a nonempty guard interpreted as a forbidden partial assignment.
It is inactive until an independent verifier accepts its evidence-bound
certificate. Nogoods can only remove support; no refinement operation may add
a previously absent world.

### 3.4 Guarded ordered patches

A `GuardedPatch` contains:

- one guard;
- one complete typed transaction;
- an evidence or command provenance commitment; and
- a chronological transaction index.

The first v0 transaction vocabulary is deliberately finite:

- `SET_VALUE(slot, value)`;
- `ADD_VALUE(slot, delta)`;
- `COPY_VALUE(source, target)`;
- `SWAP_VALUE(left, right)`;
- `SET_TYPE(slot, type)`;
- `LINK(source, relation, target)`; and
- `UNLINK(source, relation, target)`.

Transactions execute in chronological order. Noncommuting order is part of
the semantics. A transaction that violates state typing or bounds yields a
hard contradiction for exactly the assignments satisfying its guard; it is
never silently skipped or repaired.

### 3.5 Calibrated support and provenance

`SupportFactor` uses the same finite scope/tuple geometry as a hard factor and
assigns a positive integer mass to each tuple. A complete world's unnormalized
mass is the exact product of all matching factor masses. Zero and negative
masses are invalid. The packet stores source, evidence, transaction, factor,
and nogood commitments in separate canonical provenance records. No raw
source payload is embedded in these records.

### 3.6 Overflow

Caps apply to variables, domain width, represented worlds, factors, factor
tuples, patches, guard literals, nogoods, cells, edges, and exact integer bit
growth. Crossing any cap sets a sticky `overflow` disposition. An overflowed
packet exposes no incomplete world sample, accepts no certificate, and may
only abstain. Overflow propagates through every refinement and execution.

## 4. Exact semantics

For every assignment satisfying all hard factors and no accepted nogood:

1. start from the same validated shared state;
2. apply every guarded patch whose guard is true, in chronological order;
3. retain the complete resulting state or exact contradiction receipt; and
4. multiply exact support-factor masses.

The packet represents only complete assignments and complete states. It never
averages values, types, edges, programs, or transaction fields across worlds.

### 4.1 Independent reference

The assessor receives the board's complete gold world specifications and
executes each world separately with an independently implemented transaction
interpreter. It must not call packet enumeration, guard evaluation, patch
execution, refinement, merging, or query code. Exact extensional parity means
identical assignment support, contradiction disposition, terminal state,
query answers, and integer mass for every represented world.

### 4.2 Verified conflict cores and nogoods

A candidate conflict core is accepted only when the independent verifier
proves both:

1. every complete assignment matching the core contradicts the cited sealed
   evidence or a hard state invariant; and
2. no evidence-consistent valid world matches the core.

The report also tests deletion-minimality, but v0 safety requires validity,
not minimality. An invalid, stale, source-unbound, or unverifiable core is
rejected without modifying support. Gold-support recall is measured
immediately after compilation and after every accepted nogood.

### 4.3 Conservative merging

Structural merging is allowed only for assignments with bit-identical
canonical terminal states and identical remaining factor/provenance behavior.
Behavioral merging additionally requires an assessor certificate that the
worlds agree under every command continuation and query in the board's finite
declared future universe. Merging combines integer mass and provenance but
never state fields. An incomplete future universe cannot certify a behavioral
merge.

### 4.4 Query marginalization

The late reader returns the exact answer-to-mass map over all surviving
worlds. It emits `ANSWER(value)` only when every positive-mass world returns
the same value. If credible worlds disagree, it emits `ABSTAIN` plus the exact
marginal. It emits `REJECT` for invalid packets and `OVERFLOW` for overflowed
packets. Learned concentration thresholds are outside v0.

## 5. Delayed Disambiguation/Recovery board

Every episode contains two through 64 coherent worlds and a deliberately
wrong initial top-1 mass assignment while retaining the gold world. Delayed
evidence arrives only after sealing. Commands include at least two
noncommuting transactions. Each episode is paired with three late queries:

- `sensitive`: evidence should resolve a world distinction that changes the
  answer;
- `invariant`: multiple worlds remain, but every survivor gives one answer;
- `underdetermined`: credible surviving worlds disagree, so abstention is
  required.

The initial CPU board uses three typed ontologies over the same transaction
contract: register workshops, parcel-relation graphs, and signal-routing
systems. Train/development/confirmation splits hold out, independently:

- lexical renderer templates;
- alias and entity pools;
- ambiguity widths (`2--8`, `16--32`, `64` represented worlds);
- command depths (`2--4`, `5--7`, `8--10`);
- fault-line compositions; and
- the complete signal-routing ontology for confirmation.

The generator may expose gold fault lines, support, assignments, and conflict
cores only to the supervisor and independent assessor. Candidate runtime input
is the sealed packet, delayed evidence commitments, commands, and late query.
Query-only and source-retained views are explicit controls, never candidate
inputs.

## 6. Matched neural arms

These arms are frozen in name and meaning now, but are not launched by the CPU
gate:

- **A SINGLE:** one-state CSDC/ETTR with immediate top-1 commitment;
- **B FULL:** up to `K` complete whole-state particles with shared weights and
  no fieldwise averaging;
- **C INDEPENDENT:** `K` independent recurrent trajectories;
- **D RECURRENT:** one state with total recurrent proposal FLOPs matched to
  all `K` alternatives;
- **E SOFT:** a soft latent aggregate permitted to blend fields;
- **F FACTORIZED:** the DIVERGE packet without conflict learning;
- **G DIVERGE:** factorized packet plus verified conflict refinement, safe
  merging, and query-invariant commitment.

Required interventions are shuffled guard/evidence provenance, forced
premature top-1, packet swaps between matched episodes, conflict-disabled,
state reset, query-only, and shuffled labels. Full particles are the principal
capacity control and scaffold, not the contribution.

## 7. CPU mechanics gates

All must pass on calibration, development, and held-out ontology splits:

1. 100% extensional parity with the independent enumerator;
2. 100% gold-support recall immediately after compilation;
3. zero verifier-accepted nogoods that remove a valid world;
4. zero false query commitments or certificates;
5. zero unsafe structural or behavioral merges;
6. exact monotone support under every refinement;
7. deterministic canonical bytes and commitments under insertion, variable-
   ID, factor-order, and guard-order permutations;
8. sticky fail-closed behavior for every overflow cap;
9. exact equality between factorized and full-particle marginals; and
10. complete receipts for represented worlds, canonical bytes, unique versus
    duplicated transactions, verifier calls, peak packet objects, and integer
    bit growth.

CPU calibration must report worlds/byte, worlds/unique-transaction, and
duplicated-versus-shared transaction ratios before numerical neural promotion
thresholds are frozen. Those thresholds must be committed before any learned
arm is trained. They must require materially higher initially-wrong-top-1
recovery for G than the best matched control, positive gain on every board
family, causal collapse under relevant interventions, and a real sharing
advantage. First-pass support loss is fatal and cannot be averaged away.

## 8. Frozen CPU result

The deterministic seed-`20260805` board contains twelve episodes and 252
complete worlds: two lexical renderers at each ambiguity width
`2/4/8/16/32/64`, command depths `2/3/4/5/6/9`, and a completely held-out
signal-routing ontology at width 64. Every episode begins with the gold world
in support but a wrong top-1 support assignment. Delayed evidence eliminates
exactly half of the worlds.

All exact semantic gates pass:

- candidate/reference execution parity: 252/252 before refinement and 126/126
  after refinement;
- compile-time gold-support losses: zero;
- valid-support losses from twelve independently verified nogoods: zero;
- false invariant commitments, false sensitive answers, or missed required
  abstentions: zero;
- unsafe merge receipts: zero;
- source text retained in canonical packet bytes: zero; and
- all sealing-time and runtime overflow tests are sticky and fail closed.

The complete factorized packets use 37,930 canonical bytes versus 640,960
bytes for complete materialized whole-particle controls, a `16.90x` aggregate
storage advantage. The advantage rises monotonically from `1.18x` at two
worlds to `38.34x` at 64 worlds. Execution records 1,792 duplicated whole-world
transactions versus 320 unique state/transaction applications (`5.60x`), of
which 1,472 are shared. These are representation/mechanics results, not learned
reasoning results.

The first pre-threshold calibration report correctly failed its aggregate
storage gate because it undercharged the control by serializing only terminal
states. That report is retained read-only with SHA-256
`da555952246730a1df854aec39b9256cb69a6d3624a23ced2b4478fb72496da9`.
The corrected control materializes each particle's initial and terminal state,
selected fault-line semantics, applicable program, support mass, and
provenance. No packet semantics or board episode changed. The passing report
has board commitment
`c8c371293b7d023d8bea9d7e9defa31c87da9d046c08a58177d496765ce88a17`
and SHA-256
`b3562654524d773901a5ed4aebf91d0c1408883d4786451ae1053d6766daddec`.

The learned gate is now frozen in
`DIVERGE_V0_NEURAL_PROMOTION_GATE.json`. Neural promotion requires 100% source
support recall and zero false certificates, at least 90% wrong-top-1 recovery,
a ten-point OOD gain over every matched control, gains on every family and four
of five seeds, and the specified causal intervention collapses. A merely
working factorized executor does not pass.

## 9. Invalid provisional V3 matched A--G result

The first two aggregate attempts are retained but invalid for promotion. The
first board accidentally gave one sensitive label two-thirds prevalence. The
balanced successor still re-encoded source aliases when delayed evidence
arrived, so source deletion occurred too late. Neither result is used below.

The corrected runtime separates the phases. Source-time inference predicts
only token roles, ordered actions, support priors, and complete fault-line
records. It then deletes the source view. Delayed evidence supplies an alias;
the runtime hashes that alias and matches it against option commitments already
inside the sealed packet. It creates the binary nogood from that match alone.
The independent assessor checks the candidate certificate afterward but does
not gate, repair, or change runtime execution.

Five compiler checkpoints each evaluate the same balanced 144-episode board:
three ontologies, two held renderers, widths 2--64, and sensitive, invariant,
and underdetermined late queries. Across 720 episodes / 2,160 queries:

| Arm | Mean exact |
|---|---:|
| A single immediate top-1 | 33.333% |
| B memory/FLOP-capped whole particles | 33.333% |
| C independent coherent trajectories | 40.602% |
| D equal-transaction recurrence | 33.333% |
| E soft field aggregation | 66.667% |
| F factorized packet without conflict | 66.667% |
| **G full DIVERGE** | **100.000%** |

Every seed has 100% exact packets, compile-time gold-support recall, valid-
support preservation, sensitive recovery, invariant answers, required
abstentions, and zero false G certificates. G is 100% in every ontology. The
strongest A--F control is 66.667%, a 33.333-point margin. C ranges from 39.352%
to 42.130% across seeds.

Causal drops from G are 33.333 points without conflict, 66.667 under forced
top-1, 33.333 with shuffled guard/provenance, 34.722 under packet swap, and 100
under state reset or shuffled labels. Query-only equals its balanced empirical
majority at 83.333%. Source deletion is bit-identical.

At 64 worlds the packet uses 4,745 canonical bytes versus 176,177 complete-
particle bytes (`37.129x`) and 52 unique versus 448 duplicated transaction
applications (`8.615x`). The width-eight transaction ratio is exactly the
frozen `2.0x` floor after a semantics-preserving schedule moves only provably
disjoint slot transactions; all overlapping/noncommuting pairs retain source
order and exact execution parity.

The immutable five-seed aggregate SHA-256 is
`0d78b271cd4bf4761dde5aa80b929e24ad50fe5057ae5f9cd2c1ea763a8b918d`.
Per-seed report hashes are `039015b0...8aa8`, `c25a449a...9fa2`,
`b3a5a0f8...9154`, `9834f958...aca`, and `da1e5ef5...d8f`. Runtime commit is
`8e846c3`; minimal archive SHA-256 is
`778d33c37241f27c502ba274bc1b35ce70f3456778e24c46d2fbd4bb8bd3fadc`.

This is a real synthetic mechanism win, not unrestricted language reasoning.
The finite grammar exposes candidate records and a parseable delayed key. The
remaining CUDA run supplies profiled compiler FLOPs and peak device memory;
subsequent promotion must then broaden the language compiler and execute a
sealed distribution shift without changing this gate.

### Resource-corrected final matched result

The result above is V3 and is **not claim-eligible**. Its executor called the
enumerative world path internally while its receipt charged only static packet
bytes. That underfunded the complete-particle control and omitted DIVERGE's
active state-group memory. Pending CUDA jobs `742054--742058` were canceled
before allocation once this was found.

V4 replaces that path with an exact bitset/state-group executor. It represents
each distinct terminal state once, associates it with a disjoint support mask,
merges only bit-identical states, and refines delayed evidence by intersecting
already-executed masks without replaying transactions. It matches independent
enumeration on an additional 360-episode / 7,560-world audit with 2,160 exact
pre/post-evidence query comparisons. The focused suite has 21 passing tests.
Whole-particle B receives the maximum of DIVERGE's pre/post-evidence canonical
packet plus measured peak state-group bytes and the same unique transaction
budget.

Five Stokes seeds (`766191--766195`) cover 720 episodes / 2,160 late queries:

| Arm | V4 mean exact | Five-seed range |
|---|---:|---:|
| A single immediate top-1 | 33.333% | 33.333% |
| B resource-matched whole particles | 72.222% | 72.222% |
| C independent coherent trajectories | 65.278% | 62.500--68.750% |
| D equal-transaction recurrence | 33.333% | 33.333% |
| E soft field aggregation | 66.667% | 66.667% |
| F factorized packet without conflict | 66.667% | 66.667% |
| **G full DIVERGE** | **100.000%** | **100.000%** |

Packet/support/refinement minima remain 100%, with zero false G certificates
and the same required causal collapses. The corrected effective storage ratios
(static packet plus peak active groups versus complete particles) are `1.010x,
1.893x, 3.412x, 6.412x, 13.359x, 27.365x` at 2, 4, 8, 16, 32, and 64 worlds.
The frozen four-world threshold is `>=2.0x`; every seed therefore fails the
formal broad gate. The result is narrowed to the observed `>=8`-world regime,
not promoted, and no CUDA receipt or continuation pretraining is authorized.

Aggregate SHA-256 is
`8e4405920379b7c0a2f4a0c9acc463839a3816b4a87e84f78fcb9d656e17aaab`.
Per-seed hashes are `9a931017...5a7`, `ea017710...77a`, `d181845a...ac6`,
`de78c491...a6b`, and `1b4a06a2...c5f`. Exact source runtime is private
commit `e56a37f`; archive SHA-256 is
`841e901c849ee8bee805067aff54367f5e7f7c31dd8f658fbd683ba45a7fb11f`.

## 10. Kill conditions

Kill or sharply narrow DIVERGE if any of the following occurs:

- full particles match G at equal activation memory and FLOPs;
- the soft aggregate matches G;
- equal-FLOP recurrence matches G;
- gold support is absent immediately after compilation;
- a verified nogood removes a valid world;
- packet/guard/conflict interventions do not cause the predicted collapse;
- factorization exceeds its caps on modest held-out ambiguity;
- canonical sharing gives no worlds/byte or worlds/FLOP advantage; or
- answers depend on query statistics, raw source retention, hidden host
  repair, or assessor access.

## 11. Evidence and novelty boundary

Protected CSDC has already established
`plastic local evidence -> complete hypotheses -> source counterexample
falsification -> one coherent commit -> tied late execution` at
99.593%/99.723% controlled exact answers. DIVERGE tests whether that principle
can be extended compactly without deleting a minority world before delayed
evidence arrives.

Do not claim that particles, version spaces, BDDs, factor graphs, variational
execution, lifted inference, conflict clauses, typed state, or source deletion
are individually new. A future bounded contribution, if earned, is their
learned source-sealed conjunction with language-derived fault lines, shared
guarded execution, verified conflict revision, conservative merging, and
query-invariant commitment.
