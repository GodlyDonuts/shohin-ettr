# R12 ETTR Isolated Learnability Implementation Audit

## Scope

This audit compares `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md` with the
existing ETTR data contract, objective structures, cross-ontology generators,
and structural-variant generators. It addresses deterministic CPU
materialization only. It does not authorize training, evaluation, confirmation
opening, or Newton execution.

## Decision

**NO-GO for claim-bearing CPU materialization.**

The frozen theory-index pools and existing score-board commitments are
reconstructible, but the preregistration does not yet determine a unique
dataset. Several required objects have no canonical schema or constructor,
some preregistered rectangles are incompatible with
`ETTRCausalRectangle`, and the resource depth/command requirements cannot be
satisfied by the current generator. Materializing now would require the
implementer to make unregistered scientific choices.

## Hard Blockers

### 1. The split commitments have no reconstructible preimage

**Evidence**

- The preregistration publishes master, split, and fold hashes at
  `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md:170-203`.
- It ranks candidates with
  `SHA256(master_commit || canonical_tuple_bytes)` at
  `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md:180-187`.
- It never defines the split-spec bytes, fold-spec bytes,
  `canonical_tuple_bytes`, candidate fields, field encodings, candidate
  domains, or enumeration order.
- No existing generator contains those commitments or an ETTR isolated-board
  constructor.

**Impact**

A hash authenticates known bytes; it cannot reconstruct an unspecified
preimage. Two conforming implementers can produce different folds, strata, and
rows while retaining the same prose interpretation.

**Required repair**

Freeze literal canonical split/fold specification files or an exact constructor
for them. Define a versioned tuple schema and byte encoding containing, at
minimum: fold, split, ontology, stratum, theory identity, two worlds, two
commands, depth, renderer, presentation, query semantics, paraphrases, and
opaque-name seed. Define admissibility, complete candidate enumeration, stable
sorting, tie handling, and fold-hash derivation.

### 2. Command depth 4-6 is unsupported, and depth semantics are incomplete

**Evidence**

- The preregistered composition split requires operation-sequence depths
  `4, 5, 6` at `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md:112-125`.
- Resource programs are capped at three operations by
  `MAX_SEQUENCE_LENGTH = 3` in
  `pipeline/cross_ontology_resource_board.py:175-180`.
- `execute_sequence` rejects sequences outside lengths 1-3 at
  `pipeline/cross_ontology_resource_board.py:233-243`.
- Existing resource held-out programs contain only lengths 2 and 3 at
  `pipeline/cross_ontology_resource_board.py:296-302`.
- Existing factorial commands encode one Horn atom, one rewrite constructor,
  or one resource sequence at
  `pipeline/ettr_factorial_qualification_board.py:397-404`; execution applies those
  single Horn/rewrite actions at
  `pipeline/ettr_factorial_qualification_board.py:436-466`.

**Impact**

Resource depth 4-6 is rejected by the current oracle. For Horn and rewrite,
there is no frozen definition of a dependent multi-operation command, its
intermediate states, failure behavior, or transaction trace.

**Required repair**

Freeze a typed command AST and sequential execution oracle for every ontology,
including intermediate packet/transaction targets. Either extend and validate
resource execution through depth 6 or preregister a different composition
regime. Define Horn and rewrite composition rather than inferring it from
single-operation code.

### 3. The 16-row semantic rectangle is incompatible with the ETTR rectangle contract

**Evidence**

- One preregistered semantic rectangle has 16 rows: four world-command cells,
  two query semantics, and two paraphrases at
  `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md:108-122`.
- Its admission rule requires each WORLD and COMMAND edge to change at least
  one of the two query answers at
  `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md:115-122`.
- `ETTRCausalRectangle` is exactly `[R, 2, 2]` at
  `train/ettr_data_contract.py:649-669`, partitions every row once at
  `train/ettr_data_contract.py:670-682`, requires one identical query prefix
  across all four cells at `train/ettr_data_contract.py:743-770`, and requires
  every WORLD and COMMAND edge to change that prefix's label at
  `train/ettr_data_contract.py:771-786`.

**Impact**

A rectangle admitted because query A changes an edge while query B does not is
valid under the preregistration but invalid under the ETTR contract. The 16
rows also cannot be represented as one current `ETTRCausalRectangle`.

**Required repair**

Choose and freeze one rule. The least invasive repair is to require every
selected query-semantic/paraphrase group to satisfy all four ETTR edge
contrasts, then expand one semantic rectangle into four explicitly ordered
four-row causal rectangles. Otherwise, version and change the data/objective
contract under a new protocol.

### 4. “Rectangle” denotes two different optimization units

**Evidence**

- Dataset sizes count 16-row semantic rectangles at
  `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md:143-168`.
- Optimization uses eight “rectangles” per update and two per microbatch at
  `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md:225-240`.
- The implemented rectangle is the four-row object described in Finding 3.
- Six thousand updates times eight rectangles exceeds the 2,304 fitting
  rectangles, but no epoch, wrap, repetition, or reshuffle schedule is frozen.

**Impact**

The two readings yield either 128 or 32 rows per update and expose different
numbers of query formulations. They therefore change tokens, FLOPs, objective
weighting, and the exact training stream.

**Required repair**

Rename the units `semantic_rectangle` and `causal_rectangle`; define their
exact expansion and row order. Freeze rows per microbatch/update, accumulation,
query grouping, the 6,000-update repeat schedule, end-of-epoch behavior, and
whether ranking is reused or rerun.

### 5. The anonymous typed AST and four renderers do not exist

**Evidence**

- The preregistration requires canonical JSON, prefix S-expression,
  record-delimited infix, and reverse/postfix renderings of one anonymous typed
  AST at `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md:127-134`.
- Horn currently uses four ontology-specific evidence styles at
  `pipeline/cross_ontology_horn_board.py:298-343`.
- Rewrite renderings are ontology-specific at
  `pipeline/cross_ontology_rewrite_board.py:611-668`.
- Resource renderings are ontology-specific at
  `pipeline/cross_ontology_resource_board.py:443-536`.
- The factorial board only supplies canonical JSON world/command bytes at
  `pipeline/ettr_factorial_qualification_board.py:328-404` and two JSON query
  forms at `pipeline/ettr_factorial_qualification_board.py:407-433`.

**Impact**

There is no shared AST schema, total renderer, parser, round-trip check, or
independent semantic equivalence oracle. Renderer assignment and exact bytes
cannot be derived.

**Required repair**

Freeze an ontology-neutral WORLD/COMMAND/QUERY AST with canonical field order
and type tags. Implement four total renderers plus independent parsers and
round-trip semantic checks. Pin delimiters, escaping, whitespace, integer
encoding, templates, and tokenizer identity.

### 6. Current candidate bytes leak the held-out ontology

**Evidence**

- Stable family/ontology labels are forbidden at
  `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md:81-85`.
- Existing qualification payloads explicitly use numeric domain codes at
  `pipeline/ettr_factorial_qualification_board.py:88-96`.
- The same stable `"d"` code is embedded in world, command, and query bytes at
  `pipeline/ettr_factorial_qualification_board.py:328-433`.

**Impact**

The unnamed integer is a perfect ontology identifier. In leave-one-ontology-out
scoring it permits family routing and violates the stated isolation rule even
though no human-readable family name appears.

**Required repair**

Remove stable domain codes from candidate-visible bytes and infer types from
anonymous syntax, or explicitly allow anonymous domain tags and preregister
them as candidate features. In either case, include the exact bytes in leakage
controls and metadata-classifier inputs.

### 7. Rewrite fit presentations cannot be generated

**Evidence**

- Every fit ontology must use `base`, `alpha_reorder`, and `alias_split` at
  `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md:136-140`.
- Rewrite fitting uses the frozen fit theory indices at
  `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md:89-95`.
- `build_rewrite_variant_family` rejects any theory outside
  `HELDOUT_THEORY_INDICES` at
  `pipeline/cross_ontology_rewrite_variants.py:1317-1323`.
- The emitted variant family is constructed only after that rejection at
  `pipeline/cross_ontology_rewrite_variants.py:1324-1495`.

**Impact**

The required rewrite fit rows cannot be produced with the existing public
variant constructor.

**Required repair**

Implement and audit a fit-safe rewrite presentation constructor for exactly
the three permitted fit presentations, or generalize the builder while keeping
score-only twins/challenges inaccessible to fitting.

### 8. Score presentation allocation is not specified

**Evidence**

- The `all_axes` stratum “additionally uses” four score-only presentations at
  `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md:136-141`.
- The exact stratum counts are stated at
  `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md:154-165`, but no count is allocated
  to each presentation and no rule says whether presentations are separate or
  composed with renderer, theory, and depth axes.
- Existing Horn variant kinds and expectations are separate cases at
  `pipeline/cross_ontology_horn_variants.py:31-97`.
- Resource pair directives are separate cases at
  `pipeline/cross_ontology_resource_variants.py:61-87` and
  `pipeline/cross_ontology_resource_variants.py:772-868`.
- Rewrite variants are likewise emitted as distinct cases at
  `pipeline/cross_ontology_rewrite_variants.py:1324-1495`.

**Impact**

There is no deterministic way to fill the 96 `all_axes` rectangles, pair twins
to bases, allocate invariant versus changed-answer cases, or decide legal
renderer/presentation composition.

**Required repair**

Freeze a split-by-stratum factor table with exact quotas summing to every
published count. For each cell specify theory pool, depth, renderer,
presentation, base-pair identity, expected disposition, answer derivation, and
allowed compositions.

### 9. Query semantics and paraphrases are not generative specifications

**Evidence**

- The preregistration requires two semantic queries and two paraphrases per
  semantic rectangle at `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md:117-121`,
  but provides no query grammar, templates, evaluator, or selection rule.
- The existing factorial board has only two fixed `SemanticProbe` classes at
  `pipeline/ettr_factorial_qualification_board.py:99-104`, a fixed evaluator at
  `pipeline/ettr_factorial_qualification_board.py:504-526`, hand-selected
  definitions at `pipeline/ettr_factorial_qualification_board.py:529-573`, and two
  fixed query byte forms at
  `pipeline/ettr_factorial_qualification_board.py:407-433`.

**Impact**

Those probes do not generate valid contrastive queries for arbitrary theories,
worlds, commands, and depths. Query choice would become an implementer degree
of freedom.

**Required repair**

Freeze per-ontology query candidate grammars, exact and independent evaluators,
the two paraphrase encodings, answer vocabulary, and SHA-ranked selection after
the finalized edge-contrast test.

### 10. Target balancing and ABSTAIN semantics are undefined

**Evidence**

- The preregistration requires exact 50/50 target balance per
  ontology/stratum/query/paraphrase while balancing ABSTAIN separately at
  `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md:289-304`.
- Factorial probe targets are Boolean at
  `pipeline/ettr_factorial_qualification_board.py:108-123` and
  `pipeline/ettr_factorial_qualification_board.py:504-526`.
- Matrix rows instead carry directives and terminal expectations at
  `pipeline/cross_ontology_qualification_matrix.py:64-79` and
  `pipeline/cross_ontology_qualification_matrix.py:201-315`.
- Resource variant cases carry disposition directives and full outcomes at
  `pipeline/cross_ontology_resource_variants.py:193-203` and
  `pipeline/cross_ontology_resource_variants.py:806-865`.
- Rewrite uses a structured `VariantOracle` at
  `pipeline/cross_ontology_rewrite_variants.py:219-228`.

**Impact**

There is no canonical projection from structured terminal outcomes to the
balanced query label, nor a denominator for 50/50 strata containing ABSTAIN or
REJECT.

**Required repair**

Define the complete label alphabet and projection from each oracle outcome.
Publish an exact balance table per answer class, state whether dispositions are
excluded from binary balance, and freeze deterministic query selection when a
cell cannot meet the requested label.

### 11. No materializer maps ontology cases to ETTR continuation targets

**Evidence**

- `ETTRContinuationBatch` requires dataset/manifest commitments, episodes,
  initial and terminal packet targets, causal rectangles, transaction targets,
  and disposition flags at `train/ettr_data_contract.py:1004-1018`; validation
  replays and checks the batch at `train/ettr_data_contract.py:876-1136`.
- Equivariance training additionally requires slot, type, relation, and value
  permutations and masks at `train/ettr_objectives.py:567-650`.
- Existing qualification matrix rows store hashes, directives, and
  expectations, not ETTR packet/transaction tensors, at
  `pipeline/cross_ontology_qualification_matrix.py:64-79`.
- The concurrent fail-closed freeze gate can audit a future canonical JSONL
  (`pipeline/freeze_ettr_isolated_learnability.py:601-634`) but its
  `materialize` command explicitly reports that no admitted production row
  generator exists (`pipeline/freeze_ettr_isolated_learnability.py:1065-1077`).

**Impact**

Even after selecting semantic cases, there is no canonical CPU path that emits
the tensors consumed by the preregistered arms. Packet slots, transaction
steps, dispositions, query-read positions, and alignment permutations would
all be invented during implementation.

**Required repair**

Implement a versioned CPU materializer that maps each ontology terminal state
to `ETTRPacketTargets`, each operation to complete
`ETTRTransactionTargets`, each query to token targets/read indices, and each
structural variant to `ETTRVariantAlignment`. Require independent oracle
replay and `ETTRContinuationBatch` validation before hashing artifacts.

### 12. The binding-deranged control contradicts the anti-leakage donor gate

**Evidence**

- Arm 3 reassociates supervision within the same
  ontology/depth/renderer/**answer** stratum at
  `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md:260-273`.
- The anti-leakage gate requires every wrong-WORLD, wrong-COMMAND, and
  shuffled-state donor to change the assessor target at
  `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md:297-301`.

**Impact**

Reassociation within the same answer stratum preserves the answer by
construction, while the gate requires the donor to change it. No exact
derangement or component-level interpretation resolves the contradiction.

**Required repair**

State which target components must be preserved and which must change. Remove
`answer` from the donor stratum if the query target must change, or redefine
the gate as a packet/transaction-binding test with an unchanged query label.
Freeze a total seeded derangement algorithm and its no-fixed-point checks.

### 13. Confirmation encryption and open-once custody are not implementable

**Evidence**

- The preregistration requires encrypted immutable confirmation artifacts and a
  root-signed manifest at
  `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md:205-212`.
- It requires exactly one authorized confirmation opening at
  `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md:308-321`.
- Existing authority code signs factorial-board/execution commitments in
  `train/ettr_factorial_authority.py:115-218`; it does not define this
  dataset's encryption, key custody, or open-once state machine.

**Impact**

No encryption algorithm, key authority, nonce/AAD derivation, ciphertext
schema, signature schema, or access ledger is specified. Secure randomized
encryption also cannot have a predetermined ciphertext hash unless its
randomness is frozen or the post-encryption artifact is separately committed.

**Required repair**

Freeze the dataset-manifest and signature schemas, authority public key,
authenticated-encryption algorithm, key custodian, nonce/AAD policy, and
append-only opening receipt. Distinguish deterministic semantic plaintext
generation from custodian-generated ciphertext, then hash and sign the latter
after encryption.

## Additional Determinism and Feasibility Defects

### 14. Zero command overlap may be combinatorially impossible at resource depth 1

**Evidence**

- Zero command overlap across train/development/confirmation is required at
  `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md:289-296`.
- A rectangle needs two commands at
  `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md:108-116`.
- The resource command alphabet has only three operator symbols at
  `pipeline/cross_ontology_resource_board.py:175-180`.

**Impact**

If “command overlap” means semantic operation sequence and more than one split
contains depth-1 resource rectangles, each split needs at least two commands
from a universe of three. Even train versus one score split cannot be
disjoint. The preregistration also does not assign depths to each score
stratum, so feasibility cannot be decided.

**Required repair**

Define command identity precisely: semantic AST, rendered bytes, or
world-bound instance. Publish the complete split/stratum/depth table and a
pre-materialization cardinality proof. If semantic identity is intended,
expand the command grammar or relax the overlap gate under a new commitment.

### 15. Leakage and overlap gates have no canonical algorithms

**Evidence**

- The preregistration requires semantic-world, theory, command, opaque-name,
  graph-isomorphism, token-sequence, normalized 13-gram, and metadata-classifier
  gates at `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md:289-304`.
- It does not define text normalization, tokenizer/version, graph canonical
  labeling, opaque-name extraction, pairwise split scope, classifier features,
  solver, hyperparameters, seed, folds, or the meaning of chance for the
  classifier threshold.

**Impact**

Gate outcomes are implementation-dependent and therefore cannot certify a
deterministically frozen dataset.

**Required repair**

Freeze every fingerprint function and comparison scope. Pin normalization and
tokenizer hashes, graph canonicalization, classifier implementation/version,
features, solver, seed, train/test folds, confidence rule, and multiclass chance
calculation.

### 16. Frozen source labels are path-ambiguous and current HEAD has drifted

**Evidence**

- The preregistration pins commit `854d3d4...` and source hashes at
  `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md:31-51`.
- The published “Qualification source” hash corresponds to
  `train/ettr_qualification.py`, but the path is not stated.
- The current supervisor source differs from the preregistered hash; the file
  at the pinned commit matches it.

**Impact**

The commitment is recoverable only if the implementer guesses the intended
paths and builds from the detached frozen commit rather than the current
checkout.

**Required repair**

Add a canonical source inventory of `{path, sha256, commit}` and require a
clean detached checkout of the pinned commit for source-frozen construction.
Any materializer added after the preregistration needs its own protocol version
and source commitment.

The untracked freeze gate that appeared during this audit improves failure
handling but is not a preregistered source repair: it records several of the
same unresolved clauses at
`pipeline/freeze_ettr_isolated_learnability.py:806-921`.

## Verified Foundations

The following parts are not blockers:

- The master commitment at
  `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md:170-179` reproduces from its stated
  literal preimage.
- The existing matrix, factorial, and hybrid score-board payload hashes at
  `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md:214-223` reproduce.
- The frozen Horn, rewrite, and resource fit/score theory-index pools at
  `R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md:86-95` are valid, internally
  unique by behavior in the current generators, and behaviorally disjoint
  between fit and score pools.
- Existing ETTR batch validation is strict enough to serve as a downstream
  materialization gate once the missing mapping and rectangle semantics are
  frozen.

## Minimum Repair Order

1. Freeze the canonical split schema, candidate tuple, axis/quota table, query
   grammar, and exact command identity.
2. Resolve command depth semantics and the resource depth-1 overlap
   cardinality.
3. Reconcile 16-row semantic rectangles with four-row
   `ETTRCausalRectangle` objects and freeze the update/repetition schedule.
4. Implement the shared typed AST/renderers and fit-safe rewrite presentations.
5. Define outcome projection, ABSTAIN balancing, all score-presentation
   allocation, and the binding-control derangement.
6. Implement and independently validate the ontology-to-ETTR CPU materializer.
7. Freeze overlap algorithms and external confirmation custody.
8. Produce a dry-run cardinality/feasibility report before writing any
   claim-bearing train, development, or confirmation artifact.

Until all eight repairs are committed under a new explicit protocol version,
CPU materialization must remain **NO-GO**.
