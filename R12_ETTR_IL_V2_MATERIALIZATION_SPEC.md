# R12 ETTR IL v2 CPU Materialization Specification

**Protocol:** `R12-ETTR-IL-v2-materialization-v1`  
**Status:** Phase-1 CPU mapping implementation complete; no production
population, fitting, confirmation opening, or job is authorized
**Scope:** closes the design ambiguity in implementation-audit blocker 11 and
reconciles a 16-row semantic rectangle with the existing four-row
`ETTRCausalRectangle` contract. All other blockers in
`R12_ETTR_ISOLATED_LEARNABILITY_IMPLEMENTATION_AUDIT.md` remain closed until
separately repaired.

## 1. Normative implementation surface

This document specifies the implemented CPU-only mapper from an already
admitted v2 semantic case into the existing, unmodified:

- `ETTRContinuationBatch`;
- `ETTRPacketTargets`;
- `ETTRTransactionTargets`;
- `ETTRVariantAlignment`;
- `ETTREpisodeBatch`;
- `ETTREpisodeSegment`; and
- `ETTRCausalRectangle`.

The mapper is assessor-side. It may call ontology oracles while constructing
and validating labels. No ontology parser, family identifier, oracle object,
alignment table, semantic evaluator, or target may enter a model forward
input.

The following sources are normative for the receiving contract:

- `train/ettr_data_contract.py`;
- `train/ettr_objectives.py`;
- `train/ettr_episode.py`;
- `train/ettr_train_step.py`;
- `train/endogenous_typed_theory_reactor.py`;
- `pipeline/cross_ontology_schema.py`;
- `pipeline/cross_ontology_horn_board.py`;
- `pipeline/cross_ontology_rewrite_board.py`;
- `pipeline/cross_ontology_resource_board.py`;
- the three `cross_ontology_*_variants.py` modules; and
- `pipeline/ettr_factorial_qualification_board.py`.

If a future implementation changes any receiving dataclass, validation rule,
opcode meaning, model geometry, tokenizer, or source hash, it is not an
implementation of this protocol.

## 2. Frozen production geometry

The production dimensions are:

| Symbol | Meaning | Value |
|---|---|---:|
| `S` | packet slots | 64 |
| `T` | type classes | 8 |
| `R` | relation classes | 16 |
| `V` | value classes | 256 |
| `K` | reactor steps in every materialized batch | 64 |
| `E_max` | active relation edges | 256 |
| `A` | answer codes | 4 |

The only tokenizer payload has 2,309,567 bytes and SHA-256
`87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4`.
Its exact runtime identity and conformance vectors are defined by the custody
specification.

All categorical tensors are CPU `torch.int64` (`torch.long`). All masks and
status tensors are CPU `torch.bool`. A materialized batch is moved to an
accelerator only by a later loader. Every tensor in one
`ETTRContinuationBatch` must share one device when validated.

## 3. Required v2 semantic-case input

The mapper consumes canonical records with schema
`r12-ettr-il-v2-semantic-case-v1`. Upstream generation must have fixed their
literal bytes, enumeration, split membership, renderer, presentation, theory,
and query grammar. This specification does not choose those objects.

One semantic case contains:

1. one `semantic_rectangle_id`;
2. exactly two semantic WORLD objects, `W0` and `W1`;
3. exactly two typed COMMAND ASTs, `C0` and `C1`;
4. exactly four independently replayed corner executions `(Wi,Cj)`;
5. exactly two query semantics, `Q0` and `Q1`;
6. exactly two meaning-preserving query encodings per semantic, `P0` and `P1`;
7. one candidate-visible ASCII WORLD byte string per `(world,command-cell)`,
   where the two strings for a fixed semantic world are independently
   opaque-renamed meaning-preserving renderings;
8. one candidate-visible ASCII COMMAND byte string per
   `(world-cell,command)`, where the two strings for a fixed semantic command
   are independently opaque-renamed meaning-preserving renderings;
9. one candidate-visible answer-free ASCII query-prefix byte string per
   `(query_semantic, paraphrase)`;
10. assessor-only initial, per-operation, and terminal oracle structures for
    each corner;
11. an assessor-only disposition in `answer`, `abstain`, or `reject`;
12. an assessor-only Boolean target when disposition is `answer`, and `null`
    otherwise;
13. a canonical ordered list of one through six command operations;
14. a primary-oracle receipt and a structurally independent replay receipt;
15. an invariant-orbit identity, or `null` for a semantically changed twin;
    and
16. the canonical alignments supplied by the ontology variant oracle.

Assessor-only `ontology`, theory indices, canonical names, split names,
variant names, behavior classes, possible-outcome sets, and oracle products
may exist in the semantic-case record. They are not fields of
`ETTRContinuationBatch` and must not be serialized into any token segment.

Canonical JSON is ASCII, sorted-key, compact-separator JSON followed by one
newline. Bytes, tensors, and hashes are represented in a separate versioned
envelope; no Python pickle is a canonical semantic-case encoding.

## 4. Ontology oracle projection

### 4.1 Shared static theory graph

For all three ontologies, the static theory portion is the canonical
`ReactorState` returned by the applicable `reference_theory_state(theory_index)`.
Only these fields are consumed:

- `cells`;
- `edges`;
- `root`;
- `type_count`; and
- `relation_specs`.

`ReactorState.committed_steps` and `ReactorState.halted` are construction
history, not learned packet disposition, and are ignored. Every learned
initial packet is open.

The existing reference states fit the static packet region:

| Ontology | Static cells | Static edges | Largest raw value | Construction steps |
|---|---:|---:|---:|---:|
| Horn | 19-23 | 23-34 | 258 | 55-71 |
| Rewrite | 18-25 | 15-27 | 272 | 54-80 |
| Resource | 18-21 | 23-29 | 769 | 62-74 |

The construction-step column is an audit fact, not a transaction target. Some
values exceed the ETTR value-class range and some construction traces exceed
`K=64`; Sections 5 and 8 define the required projection.

### 4.2 Horn dynamic state

The primary Horn result is the sorted tuple returned by `execute_closure`.
The independent result must be returned by an implementation that does not
call `execute_closure`.

- Slots `32..37` represent the six canonical objects.
- Each slot is active, has type code `4`, and has value
  `LOCAL_ID(object_index)`.
- Predicate `p` is relation code `8+p`, for `p in 0..4`.
- Unary fact `(p,(a,))` is the self-edge `[8+p,a_slot,a_slot]`.
- Binary fact `(p,(a,b))` is `[8+p,a_slot,b_slot]`.
- The initial fact set is encoded in `packet_targets`.
- Each operation snapshot's exact fact set is encoded during trace
  construction.
- The final fact set is encoded in `terminal_packet_targets`.

Persistent closure may only add facts. A semantic twin that removes facts is
mapped with `UNLINK` operations and remains admissible only if its complete
trace fits `K`.

### 4.3 Rewrite dynamic state

The primary rewrite result is the unique `GroundTerm` selected from
`execute_normal_forms`. The independent result must come from the independent
normal-form implementation.

- Slots `32..35` are four fixed preorder term registers.
- All four registers are active and have type code `4`.
- A present node stores `LOCAL_ID(constructor_index)`.
- An unused register stores value code `0`.
- Child topology is recovered from the frozen constructor arities and preorder
  sequence; no dynamic relation edge is emitted.
- A term with more than four nodes is inadmissible.
- A non-unique normal form has no arbitrary selected result. It follows the
  abstention path in Section 8.5.

Each operation snapshot is a four-value register vector. This representation
is intentional: a direct graph-edit trace can exceed 64 operations at depth
six, whereas a complete register rewrite needs at most four `WRITE`s per
semantic operation.

### 4.4 Resource dynamic state

The primary resource result is `ProcessOutcome(marking,cursor,status)` from
`execute_sequence` or the v2 depth-six extension. The independent result must
come from the independent resource executor.

- Slots `32..35` represent places `0..3`.
- Each slot is active and has type code `4`.
- Multiplicity `n` is stored as `SMALL_UINT(n)`.
- Cursor is stored in slot `54`.
- `ProcessStatus.HALT` is stored in slot `55` as `PROCESS_HALT`.
- `ProcessStatus.DEADLOCK` is stored in slot `55` as
  `PROCESS_DEADLOCK`.

A resource deadlock is an answerable semantic outcome unless the case oracle
separately directs epistemic abstention or rejection. It therefore normally
ends with ETTR disposition `answer`, not `reject`.

The current public resource executor rejects depths four through six. Such a
case cannot be materialized until the upstream v2 executor independently
defines and validates those depths.

### 4.5 Ambiguity and contradiction

An ambiguous oracle result is never collapsed to one member of a possible
outcome set. A contradictory world is never repaired.

- `abstain`: leave the ontology runtime registers at their last uniquely
  justified state, record the disclosed command and cursor, write
  `ABSTAIN` to slot `55`, and terminate with opcode `HALT`.
- `reject`: leave the ontology runtime registers at their last valid state,
  record the disclosed command and cursor, write `REJECT` to slot `55`, and
  terminate with opcode `REJECT`.

The possible-outcome set and contradiction witness remain assessor-only.

## 5. Canonical packet allocation

### 5.1 Slot allocation

| Slots | Meaning |
|---|---|
| `0..31` | canonical static `reference_theory_state` cells |
| `32..47` | ontology runtime registers |
| `48..53` | command-operation registers for depths one through six |
| `54` | operation cursor |
| `55` | outcome/reason register |
| `56..63` | reserved and inactive |

Static object slots retain their `ReactorState` slot numbers. A static state
using a slot above 31 is inadmissible. Dynamic slots are fixed and are not
allocated by source-name order.

### 5.2 Type allocation

| Code | Meaning |
|---:|---|
| `0..3` | canonical static oracle types, unchanged |
| `4` | ontology runtime register |
| `5` | command-operation register |
| `6` | cursor or outcome control register |
| `7` | reserved for a future already-reified static node |

Unused slots have type code `0`. A static oracle type outside `0..3` is
inadmissible under this version. Presentation-local type names are first
mapped through the assessor's canonical alignment; they never allocate new
ETTR type codes.

### 5.3 Relation allocation

| Code | Meaning |
|---:|---|
| `0..7` | canonical static oracle relations, unchanged |
| `8..12` | Horn predicate facts `0..4` |
| `13..15` | reserved |

Static unary relations are represented as self-edges. Static binary relations
are represented directly. A static relation of arity greater than two is
inadmissible unless the upstream semantic case already supplies a
deterministic reified binary graph that fits the fixed slots and relation
codes. The materializer does not invent a reification.

### 5.4 Value allocation

The global value codebook is identical in every split, fold, batch, ontology,
renderer, and presentation:

| Codes | Symbol |
|---|---|
| `0` | `EMPTY` |
| `1..32` | `STATIC_VALUE_RANK(0..31)` |
| `33..64` | `LOCAL_ID(0..31)` |
| `65..80` | `SMALL_UINT(0..15)` |
| `81..144` | `COMMAND_ATOM(0..63)` |
| `145` | `EXECUTE` |
| `146` | `ABSTAIN` |
| `147` | `REJECT` |
| `148` | `PROCESS_HALT` |
| `149` | `PROCESS_DEADLOCK` |
| `150..255` | reserved and never targeted |

For one static theory, sort its distinct signed raw `ObjectCell.value` integers
in ascending numeric order. The zero-based position is
`STATIC_VALUE_RANK`. More than 32 distinct static values is inadmissible.
This removes the existing values `258`, `272`, and `769` from the fixed
256-class interface without truncation or modulo arithmetic.

`COMMAND_ATOM` is ontology-specific but allocation-stable:

- Horn: position in `all_ground_atoms()`'s canonical sorted tuple;
- rewrite: canonical constructor index;
- resource: operator-symbol index.

No command atom may exceed 63. Opaque source spelling, declaration order,
renderer order, and alias choice do not affect any value code.

### 5.5 `ETTRPacketTargets`

For `B` rows, both initial and terminal targets have:

| Field | Shape | Dtype |
|---|---|---|
| `value_code` | `[B,64]` | `long` |
| `type_index` | `[B,64]` | `long` |
| `relations` | `[B,16,64,64]` | `bool` |
| `active` | `[B,64]` | `bool` |
| `root` | `[B,64]` | `bool` |
| `committed` | `[B]` | `bool` |
| `halted` | `[B]` | `bool` |
| `slot_mask` | `[B,64]` | `bool` |
| `relation_mask` | `[B,16,64,64]` | `bool` |

Inactive slots have `value_code=0`, `type_index=0`, and `root=false`.
Relations touching an inactive slot are false. Static roots are retained and
there is at most one root.

Both support masks are all true. This is required because
`terminal_packet_sufficiency_receipt` rejects anything short of full deployed
state supervision. Reserved cells are supervised false, not masked away.

Initial `committed` and `halted` are both false. Terminal status is assigned
only by Section 8.4.

## 6. Episode and token materialization

### 6.1 Answer allocation

The autonomous answer alphabet is:

| Semantic outcome | ASCII answer code |
|---|---|
| Boolean false | `0` |
| Boolean true | `1` |
| abstain | `2` |
| reject | `3` |

The frozen tokenizer must encode each answer code as exactly one next token in
the context of every admitted query prefix. A case requiring a multi-token
answer is inadmissible under the current one-read-index causal binding
objective.

### 6.2 Segments

For each row:

- WORLD tokens are the tokenizer encoding of the candidate-visible WORLD
  bytes.
- COMMAND tokens are the tokenizer encoding of the candidate-visible COMMAND
  bytes.
- Let `prefix_ids` be the encoding of the answer-free query prefix.
- Let `full_ids` be the encoding of `query_prefix || answer_code || "\n"`.
- `prefix_ids` must be a literal prefix of `full_ids`.
- `full_ids[len(prefix_ids)]` must be the single answer token.
- `query_read_index = len(prefix_ids)-1`.

Each segment must contain at least two tokens. WORLD, COMMAND, and QUERY are
right-padded to the protocol-wide exact widths `192`, `96`, and `48`
respectively, using token ID zero and a false attention mask. Valid token IDs
are unchanged, and any unpadded segment that exceeds its width makes the board
inadmissible before serialization. Construct each segment with
`ETTREpisodeSegment.from_tokens`, which creates the required one-token causal
targets and `-1` outside valid causal pairs.

For `B` rows:

| Field | Shape |
|---|---|
| `world.tokens/targets/attention_mask` | `[B,Lw]` |
| `command.tokens/targets/attention_mask` | `[B,Lc]` |
| `query.tokens/targets/attention_mask` | `[B,Lq]` |
| `reset_mask` | `[B]`, all true |
| `query_read_index` | `[B]` |

Each `episode_id` is:

```text
SHA256(canonical_json({
  "protocol": "R12-ETTR-IL-v2-materialization-v1",
  "semantic_rectangle_id": semantic_rectangle_id,
  "presentation_id": presentation_id,
  "semantic_index": q,
  "paraphrase_index": p,
  "world_index": w,
  "command_index": c
}))
```

It is lowercase hexadecimal and unique within the batch.

The answer suffix is training supervision, not part of the source QUERY
package. It is joined only in the assessor/training collation phase. At
autonomous evaluation, the query reader receives the prefix only and no
teacher-forced answer token.

## 7. Rectangle expansion and exact row order

### 7.1 Strict v2 admission rule

The v1 rule "each edge changes at least one of two query answers" is replaced
for materialization by the receiving contract's stronger rule:

> For every `(query_semantic, paraphrase)` group independently, both WORLD
> edges and both COMMAND edges must change the one-token answer code.

Every group must also have:

- one identical query prefix and read index at all four corners;
- identical initial packet targets across `W0,C0` and `W0,C1`;
- identical initial packet targets across `W1,C0` and `W1,C1`;
- different initial packets between `W0` and `W1`;
- identical packet support masks at all four corners;
- a different terminal packet on all four WORLD/COMMAND edges; and
- distinct raw WORLD and COMMAND renderings where
  `ETTRCausalRectangle.validate` compares them.

A semantic case that only satisfies the old existential two-query edge rule
is rejected, not repaired.

The cell-local source construction is normative. For corner `(w,c)`, render
the WORLD semantics `Ww` with opaque-name `cell_salt="world-<c>"`, and render
the COMMAND semantics `Cc` with `cell_salt="command-<w>"`. Thus the two WORLD
sources at fixed `w` differ only by a meaning-preserving opaque renaming tied
to the opposite COMMAND cell, and the two COMMAND sources at fixed `c` differ
only by a meaning-preserving opaque renaming tied to the opposite WORLD cell.
The assessor must independently parse and canonicalize both variants to the
same factor semantics before materialization. A byte-distinct pair that fails
that semantic identity check is inadmissible.

### 7.2 Expansion

Let a batch contain `M` semantic rectangles. Its row count is `B=16M` and its
causal-rectangle count is `C=4M`.

Rows are ordered by:

```text
row(m,q,p,w,c) = 16*m + 8*q + 4*p + 2*w + c
```

where each of `q,p,w,c` is in `{0,1}` and `c` is fastest varying.

The causal rectangle for `(m,q,p)` is:

```text
[
  [row(m,q,p,0,0), row(m,q,p,0,1)],
  [row(m,q,p,1,0), row(m,q,p,1,1)]
]
```

`ETTRCausalRectangle.rows` therefore has shape `[4M,2,2]`, dtype `long`, and
partitions `0..B-1` exactly once.

One 16-row semantic rectangle is not one `ETTRCausalRectangle`. It expands to
four explicitly ordered causal rectangles. Under the preregistered batching
language, "two rectangles per microbatch" means two semantic rectangles,
32 rows, and eight `ETTRCausalRectangle` entries. One update contains four
such microbatches: eight semantic rectangles, 128 rows, and 32 causal
rectangles.

## 8. Transaction trace materialization

### 8.1 Opcode allocation

The existing nine ETTR opcodes are:

| Code | Meaning |
|---:|---|
| `0` | `ALLOC` |
| `1` | `WRITE` |
| `2` | `CLEAR` |
| `3` | `LINK` |
| `4` | `UNLINK` |
| `5` | `SET_ROOT` |
| `6` | `COMMIT` |
| `7` | `HALT` |
| `8` | `REJECT` |

The v2 dynamic adapters preallocate all runtime and control registers.
Consequently their normal command traces use only `WRITE`, `LINK`, `UNLINK`,
and one final disposition opcode. Static theory-construction transactions are
never copied into command targets.

### 8.2 Operand columns

Every target step has six categorical columns. Unused operands are canonical
zero:

| Opcode | `source` | `target` | `relation` | `type_index` | `value_code` |
|---|---|---|---|---|---|
| `ALLOC` | slot | `0` | `0` | allocated type | initial value |
| `WRITE` | slot | `0` | `0` | `0` | new value |
| `CLEAR` | slot | `0` | `0` | `0` | `0` |
| `LINK`/`UNLINK` | source slot | target slot | relation | `0` | `0` |
| `SET_ROOT` | root slot | `0` | `0` | `0` | `0` |
| `COMMIT`/`HALT`/`REJECT` | `0` | `0` | `0` | `0` | `0` |

This matches `_operand_masks`: opcode is supervised on every valid step;
source on opcodes `0..5`; target and relation only on `3..4`; type only on
`0`; and value only on `0..1`.

### 8.3 Per-operation trace order

For operation position `j` in `0..depth-1`:

1. `WRITE` slot `48+j` with the canonical `COMMAND_ATOM`.
2. Apply ontology state differences:
   - Horn: emit removed facts as `UNLINK`, sorted by
     `(relation,source,target)`, then added facts as `LINK` in the same order.
   - Rewrite: for changed term registers, emit `WRITE` in ascending slot
     order.
   - Resource: for changed place registers, emit `WRITE` in ascending slot
     order.
3. `WRITE` slot `54` with `SMALL_UINT(j+1)`, except that a resource deadlock
   writes the oracle cursor.

No semantic operation may be omitted merely because it changes no ontology
register; its command-register and cursor writes remain in the trace.

After the final justified operation:

1. `WRITE` slot `55` with the exact outcome/reason code; then
2. append exactly one final disposition opcode.

### 8.4 Disposition semantics

| Semantic disposition | Final opcode | Terminal `committed` | Terminal `halted` |
|---|---:|---:|---:|
| `answer` | `COMMIT` (`6`) | true | false |
| `abstain` | `HALT` (`7`) | false | true |
| `reject` | `REJECT` (`8`) | true | true |

There is no final open state. `COMMIT` followed by `HALT` is forbidden:
deployed ETTR freezes mutations after either status is set, so the second
opcode cannot change the state. Resource process status is carried in slot
`55`, independently of this epistemic disposition.

### 8.5 Ambiguous and rejected traces

When no unique semantic operation snapshot exists, steps still record each
disclosed command atom and advance the cursor, but do not mutate ontology
runtime registers. The outcome write and final `HALT` or `REJECT` distinguish
the terminal packet and freeze padding.

This does not waive the strict rectangle rule. A four-corner query group whose
labels are all `2` or all `3` is not representable as a current
`ETTRCausalRectangle` and must be excluded or placed under a future versioned
objective contract.

### 8.6 `ETTRTransactionTargets`

All nine fields have shape `[B,64]`.

- `step_mask` is true through and including the final disposition opcode and
  false afterward.
- Padded categorical fields are zero.
- Padded `committed` and `halted` repeat the terminal status.
- At each valid position, `committed` and `halted` are the post-step statuses.
- Every row has at least one valid step.
- The final valid step is terminal.
- A trace needing more than 64 steps is inadmissible.

`initial_committed` and `initial_halted` are `[B]` all-false Boolean tensors
and exactly equal the corresponding initial packet fields.

The mapper must replay every target step with the exact recurrence in
`_validate_target_trajectory` and reproduce every supervised terminal field.

## 9. Factorial interventions

No intervention target is independently generated. The artifact stores only
factual packet and transaction targets. `objective_batch()` gathers all
counterfactual targets from factual rows.

For causal-rectangle vectors `r00`, `r01`, `r10`, and `r11`, in the exact
concatenation order used by `intervention_indices()`:

```text
world_packet   = cat(r11, r10, r01, r00)
world_command  = cat(r00, r01, r10, r11)
world_target   = cat(r10, r11, r00, r01)

command_packet  = cat(r00, r01, r10, r11)
command_command = cat(r11, r10, r01, r00)
command_target  = cat(r01, r00, r11, r10)
```

The WORLD arm takes the compiled packet from `world_packet`, command tokens
from `world_command`, and query row and factual packet/trace targets from
`world_target`. The COMMAND arm analogously uses `command_packet`,
`command_command`, and `command_target`.

Because every query group passed Section 7.1, the correct and foil next-token
targets are distinct for every generated pair. Packet source, command source,
and target provenance remain row indices; no answer is passed to
`CausalETTREpisodeRunner.intervene`.

## 10. Equivariance alignment

### 10.1 Admission

`ETTRVariantAlignment` is used only for variants whose independent oracle
proves all of the following:

- same semantic world and command;
- same disposition and query answer;
- identical canonical initial and terminal packets;
- identical canonical transaction opcode sequence;
- identical trace length and operation boundaries; and
- a bijective coordinate correspondence after semantic quotienting.

Semantically changed execution twins, ambiguity-deleted twins, and
noncongruent type twins receive no pair.

The expected invariant sets are:

- Horn: alpha/reorder, alias split, relation reification, and the Horn type
  twin only when its canonical oracle expectation is invariant;
- rewrite: alpha/reorder, alias split, and relation reification;
- resource: alpha/reorder, alias split, and relation reification.

### 10.2 Canonical quotient

Opaque names, declaration order, aliases, and relation reification are
canonicalized before packet and trace construction. Therefore all admitted v2
target coordinate maps are identity maps:

```text
slot_permutation[p]     = arange(64)
type_permutation[p]     = arange(8)
relation_permutation[p] = arange(16)
value_permutation[p]    = arange(256)
```

This is not a vacuous self-pair. Left and right are distinct rows with
different source bytes. The loss requires their learned packets and policies
to agree after compiling those different sources.

Alias splitting and relation reification are generally many-to-one or
topology-changing before quotienting. The current alignment class can express
only permutations. Direct pre-quotient alignment for those variants is
architecturally impossible and is forbidden.

### 10.3 Pair tensors

Every training microbatch contains exactly one immutable invariant pair and no
unpaired semantic rectangle. For a left rectangle at row base `16a` and right
invariant mate at `16b`, add 16 pairs in local row order:

```text
left_index  = [16a + i for i in 0..15]
right_index = [16b + i for i in 0..15]
```

For `P` pairs:

| Field | Shape |
|---|---|
| `left_index`, `right_index` | `[P]` |
| `slot_permutation` | `[P,64]` |
| `type_permutation` | `[P,8]` |
| `relation_permutation` | `[P,16]` |
| `value_permutation` | `[P,256]` |
| `slot_mask` | `[P,64]`, all true |
| `relation_mask` | `[P,16,64,64]`, all true |
| `step_mask` | `[P,64]`, true exactly on valid aligned trace steps |

Pairs must have distinct indices, must not cross split or semantic-orbit
boundaries, and must have identical factual packet and trace targets after
the declared maps. If `ETTRObjectiveConfig.require_equivariance_pairs` is
true, every objective batch must contain at least one admitted pair. Under
v2 training it is true, and each 32-row microbatch has exactly 16 alignment
pairs covering all rows. Score-only inference batches need no objective batch
and therefore do not manufacture alignments for changed twins.

## 11. Complete `ETTRContinuationBatch`

For a batch of `M` semantic rectangles:

```text
ETTRContinuationBatch(
  manifest_sha256 = final manifest SHA-256,
  dataset_sha256 = final combined dataset SHA-256,
  episodes = ETTREpisodeBatch[B=16M],
  packet_targets = initial ETTRPacketTargets[B=16M],
  terminal_packet_targets = terminal ETTRPacketTargets[B=16M],
  causal_rectangles = ETTRCausalRectangle[C=4M,2,2],
  transaction_targets = ETTRTransactionTargets[B=16M,K=64],
  initial_committed = false[B],
  initial_halted = false[B],
  equivariance = ETTRVariantAlignment[P] or null
)
```

Rows duplicated across query semantics and paraphrases have identical initial
packet, terminal packet, and transaction targets. They differ only in query
tokens, read index when the paraphrase token length differs, episode ID, and
possibly answer token. Rows within one four-corner causal rectangle always
have one common read index.

For training, `M=2` and the two rectangles are the left and right members of
one source-frozen invariant pair from the same semantic core. An optimizer
update accumulates four such independently validated batches. This is the only
training collation admitted by v2.

The runtime constructs `ETTRTokenTargets`; factual and intervention
`ETTRObjectiveBatch` fields are not serialized as additional dataset labels.

## 12. CPU replay and freeze validation

The future materializer must perform these checks in order:

1. Validate the canonical semantic-case schema and all upstream receipts.
2. Recompute primary and independent ontology outcomes.
3. Require byte-independent oracle agreement for every answerable corner.
4. Build the canonical static packet and ontology runtime registers.
5. Build every per-operation snapshot and the ETTR-native trace.
6. Replay the trace with an independent CPU implementation of the nine ETTR
   opcodes.
7. Compare replayed and target initial/terminal packet fields bit for bit.
8. Verify slot, type, relation, value, edge, and step capacities.
9. Encode the source-frozen token-native structural transport with the
   hash-bound tokenizer; prove exact WORLD/COMMAND widths, exact AST inverse,
   deterministic cover validity, and the QUERY prefix/one-token answer
   boundary.
10. Expand every semantic rectangle according to Section 7.
11. Independently check all four packet and answer edges for every causal
    rectangle.
12. Verify each invariant pair and all four full permutations.
13. Instantiate the receiving dataclasses.
14. Call `ETTRContinuationBatch.validate(reactor_config, objective_config)`.
15. Build one global `ETTRPacketSufficiencyIndex.from_splits` over the complete
    train and validation batch populations.
16. Re-run `verify_train` and `verify_validation` from freshly deserialized
    immutable batches.
17. Recompute all batch payload digests with
    `continuation_batch_payload_sha256`.
18. Hash and sign artifacts only after every prior check passes.

The batch payload digest excludes `manifest_sha256` and `dataset_sha256`, so
the freeze is two-pass and non-circular:

1. hash validated receipt-free batch payloads;
2. derive train and validation payload hashes;
3. derive the combined dataset hash;
4. construct and hash the manifest; then
5. bind those final hashes into every batch and revalidate.

Any difference between primary replay, independent replay, serialized replay,
or `ETTRContinuationBatch.validate` is a hard materialization failure. A
failed case is not repaired by masking fields, clipping values, truncating a
trace, choosing one ambiguous output, or changing row order.

## 13. Source-deletion boundary

### 13.1 Assessor-to-candidate boundary

Oracle objects, canonical alignments, ontology labels, possible outcomes,
operation snapshots, and semantic evaluators exist only in the CPU assessor
workspace. The frozen candidate artifact contains token segments, generic
packet/trace labels, row indices, masks, and hash receipts.

Packet and transaction targets are objective labels. They are never inputs to
the compiler, reactor, or query reader. Intervention execution receives
compiled states, token rows, masks, and provenance indices only.

### 13.2 Model dataflow boundary

The existing runner enforces the logical dataflow:

```text
WORLD tokens
  -> compiler
  -> TypedTheoryState
  -> independent COMMAND encoding and reactor
  -> terminal TypedTheoryState
  -> independent QUERY encoding and reader
```

Transformer context is restarted at each segment, and no WORLD or COMMAND KV
cache is passed to the next stage. Exactly the seven categorical state tensors
plus status and bound metadata carry semantics across the model interfaces.

### 13.3 Fitting versus autonomous-evaluation deletion

`CausalETTREpisodeRunner.forward` is an in-process differentiable training
call. The batch retains WORLD, COMMAND, and QUERY tensors, and autograd retains
the compiler graph needed to train the compiler. It therefore proves logical
non-consumption by later modules, not physical erasure of source bytes or
residual memory.

V2 explicitly permits that trusted, source-inventoried fitting implementation.
The model-call graph remains fail-closed: compiler receives WORLD only, reactor
receives the sealed packet plus COMMAND only, reader receives the sealed
terminal packet plus QUERY only, transformer context resets at every boundary,
and no base residual or KV state crosses a boundary. Assessor labels coexist
only in the trusted objective process and are never candidate inputs.

Every autonomous development, confirmation, and frozen-board evaluation uses
the process-isolated supervisor and public admission path instead. Those runs
must physically close and remove each source package, residual/KV state, and
runtime stage before launching the next stage, and must pass the signed
`WORLD -> COMMAND -> QUERY` receipt chain. A positive result may therefore
claim physically source-deleted inference, but MUST NOT claim physical erasure
during differentiable fitting. A future manual adjoint/recompute trainer would
be a separate protocol and is not required by v2.

## 14. Fail-closed architecture limits

The following are impossibilities under the existing receiving architecture,
not implementation choices:

1. **Rectangle label geometry.** Any query/paraphrase group with one unchanged
   WORLD or COMMAND answer edge cannot pass `ETTRCausalRectangle.validate`.
   Uniform ABSTAIN or REJECT groups are therefore inadmissible.
2. **Rectangle cardinality.** A 16-row semantic rectangle cannot be represented
   as one current causal rectangle. It must become four four-row rectangles.
3. **Partitioning.** Every batch row must occur exactly once in
   `causal_rectangles.rows`; auxiliary unpartitioned rows are impossible.
4. **Query width.** Causal binding supervises one next token at one read index.
   A multi-token atomic answer is not representable.
5. **Step capacity.** A complete command trace, including final disposition,
   must use at most 64 steps. Truncation is forbidden.
6. **Static construction traces.** Existing reference-theory construction
   traces reach 71 Horn, 80 rewrite, and 74 resource steps. They cannot be used
   as `ETTRTransactionTargets`; only their completed states can be compiler
   packet labels.
7. **State capacity.** More than 64 slots, 8 types, 16 relations, 256 value
   classes, or 256 active edges is not representable.
8. **Relation arity.** The learned relation ledger is binary. Unspecified
   ternary projection is not representable.
9. **Disposition capacity.** The two status bits encode only open, committed,
   halted, and rejected. Additional semantic statuses must live in packet
   registers.
10. **Alignment capacity.** `ETTRVariantAlignment` supports only row-wise
    bijections and does not permute opcodes or step order. Many-to-one aliases,
    graph reification, different trace lengths, or reordered microtransactions
    cannot be aligned before canonical quotienting.
11. **Value capacity.** Raw oracle integers are categorical labels, not bytes.
    Direct use of `258`, `272`, or `769` is out of range. Modulo, truncation,
    hashing into collisions, and implementer-selected recoding are forbidden.
12. **Full packet support.** The packet-sufficiency gate requires every slot and
    relation cell to be supervised. Partial support masks cannot admit a case.
13. **Deletion claim boundary.** The current differentiable in-process runner
    proves only interface-level non-consumption during fitting. Physical
    source deletion is claim-bearing only in the separately supervised
    autonomous evaluation path.
14. **Depth ownership.** The mapper cannot invent depth semantics. The
    executable v2 semantic module now owns dependent depth-one-through-six
    execution and independent replay for all three ontologies; the mapper must
    consume those admitted snapshots exactly and still fails closed if they
    are absent or disagree.

Any population quota requiring an inadmissible case is itself infeasible. The
materializer must report the exact failed limit and stop before creating a
claim-bearing output.

## 15. Disposition

This specification removes the implementer's freedom over ETTR packet fields,
trace columns, status bits, query targets, read indices, rectangle expansion,
intervention provenance, and invariant alignment. It therefore supplies the
machine-complete mapping required by implementation-audit blocker 11 and
resolves the rectangle mismatch by strengthening admission and expanding each
semantic rectangle into four current causal rectangles.

The Phase-1 implementation now supplies the semantic executors, shared AST,
four parsers, six presentations, token-native transport, quota-validation
sidecar, native batch construction, and CPU replay contracts. This remains a
mechanics freeze rather than a learned-capability result. Production fitting
is still gated on a literal frozen population, complete leakage report, and
the user explicitly opening Phase 2.
