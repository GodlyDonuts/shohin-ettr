# R12 ETTR Isolated Learnability v2 Semantic Generator Specification

**Protocol:** `R12-ETTR-IL-v2-semantic-generator`  
**Status:** machine-complete generator specification only; no materialization,
training, job submission, scored opening, or capability claim is authorized  
**Scope:** this document replaces only the underspecified semantic-generation
clauses implicated by audit blockers 1-10 and 14. It does not repair the
ontology-to-tensor materializer, control-arm derangement, confirmation custody,
overlap-audit implementations, or source inventory.

## 1. Normative language and hostile decisions

`MUST`, `MUST NOT`, `SHALL`, and `SHALL NOT` are requirements. An implementation
that cannot satisfy a quota or invariant MUST stop without emitting a split.
There is no fallback sampling, quota relaxation, retry with a new seed, or
post-hoc replacement.

The following v1 requirements are impossible or scientifically indefensible and
are replaced:

1. The v1 split and fold hashes have no specified preimages. They are retired.
   V2 derives all identities from the literal master seed and schemas below.
   Dataset hashes may be published only after literal files exist.
2. A 16-row object cannot be one `ETTRCausalRectangle`, whose geometry is
   exactly four rows. V2 defines three distinct units: `semantic_core`,
   `semantic_rectangle`, and `causal_rectangle`.
3. `ABSTAIN` and `REJECT` cannot satisfy ETTR's requirement that every WORLD
   and COMMAND edge change the matched query label. V2 causal rectangles contain
   only `ANSWER(false)` or `ANSWER(true)`. `ambiguity_deleted_twin` remains
   score-only on the already frozen seven-variant boards and is not inserted
   into the v2 causal population.
4. Zero semantic command-template overlap at resource depth 1 is impossible:
   there are three primitive symbols, while two disjoint commands are needed
   per split rectangle. V2 requires disjoint world-bound command instances,
   not disjoint primitive templates. Primitive-template overlap is measured and
   reported, never called leakage.
5. Literal-byte disjointness between all renderer alphabets is impossible for
   ASCII integers, opaque symbols, and line termination. V2 requires disjoint
   renderer-reserved structural delimiters and exact byte inequality, not
   disjoint ordinary characters.
6. The resource production helper's depth-3 guard is not semantic law. V2
   defines a separate bounded assessor oracle through depth 6 without changing
   the existing helper.
7. Rewrite challenge terms of at most four nodes cannot support six dependent
   decreasing rewrites. V2 uses the existing factorial command action: each
   operation wraps the current term and then normalizes it. It does not pretend
   that six decreasing steps exist in the old four-node challenge space.

These replacements are protocol changes, not interpretations of v1.

## 2. Frozen roots and canonical bytes

The master seed preimage is the ASCII byte string:

```text
R12_ETTR_ISOLATED_LEARNABILITY_V2|2026-07-26|semantic-generator
```

`master_seed` is the raw 32-byte `SHA256(preimage)` digest. Its lowercase
hexadecimal display is:

```text
f6edaccd75ba80763540b990fcd0d1c85016e2d62a79cc3bbe328a206db925dd
```

Implementations MUST recompute it and require this equality. Every `||` in this
document means raw byte concatenation. A displayed integer concatenated into a
hash preimage is its canonical unsigned ASCII decimal representation. `uint64`
means the unsigned big-endian interpretation of the first eight digest bytes;
`first_bit` is the most significant bit of the first digest byte.

Selection and split-local naming use the `split_key` and exact
`PRF(K,label,context)` construction in
`R12_ETTR_IL_V2_CUSTODY_SPEC.md`, not `master_seed`. Train and development
keys are publicly reconstructible; each confirmation key remains with the
independent custodian until the one authorized opening. The public
`master_seed` determines only the finite candidate universe, canonical
ordering, and split-disjoint public world ownership. It never ranks selected
confirmation cases.

The tokenizer is the immutable 2,309,567-byte JSON artifact with SHA-256:

```text
87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4
```

Tokenization is strict UTF-8 loading, candidate bytes are strict ASCII,
`add_special_tokens=false`, right padding, and pad token ID 0. The answer byte
strings are exactly `0` and `1`; under the frozen tokenizer they MUST encode as
the singleton token IDs 28 and 29. A mismatch is fatal.

Canonical JSON is RFC 8259 JSON constrained as follows:

- strict ASCII output;
- keys sorted by Unicode code point, which equals byte order for admitted keys;
- separators `,` and `:` with no whitespace;
- no duplicate object keys;
- integers only, no floats, exponent notation, `NaN`, or infinity;
- booleans and null use lowercase JSON literals;
- one trailing LF byte (`0a`);
- SHA-256 is over literal bytes including that LF.

For a value `x`, `CJ(x)` denotes those bytes. For byte strings, metadata JSON
uses lowercase hex and never base64.

## 3. Ontology-neutral surface AST

Candidate-visible WORLD, COMMAND, and QUERY objects are instances of one generic
tree. No node contains an ontology, family, theory, split, stratum, renderer,
presentation, oracle, target, or disposition field.

The exact JSON Schema is:

```json
{
  "$id":"r12-ettr-il-v2-surface-ast.schema.json",
  "$schema":"https://json-schema.org/draft/2020-12/schema",
  "$defs":{
    "node":{
      "oneOf":[
        {
          "additionalProperties":false,
          "properties":{"i":{"maximum":2147483647,"minimum":0,"type":"integer"}},
          "required":["i"],
          "type":"object"
        },
        {
          "additionalProperties":false,
          "properties":{"s":{"pattern":"^x[0-9a-f]{16}$","type":"string"}},
          "required":["s"],
          "type":"object"
        },
        {
          "additionalProperties":false,
          "properties":{
            "a":{"items":{"$ref":"#/$defs/node"},"maxItems":256,"type":"array"},
            "h":{"maximum":15,"minimum":0,"type":"integer"}
          },
          "required":["a","h"],
          "type":"object"
        }
      ]
    }
  },
  "$ref":"#/$defs/node"
}
```

Head codes have renderer-independent meanings:

| `h` | Meaning |
|---:|---|
| 0 | ordered tuple |
| 1 | mathematical set; semantic children are canonical-sort ordered |
| 2 | bag of `(key,value)` pairs |
| 3 | declaration `(surface_symbol, local_ordinal, payload)` |
| 4 | application `(surface_symbol, arguments...)` |
| 5 | typed variable `(local_ordinal, local_type)` |
| 6 | law `(left,right)` |
| 7 | state collection |
| 8 | execution-policy declaration |
| 9 | direct Boolean query |
| 10 | Boolean-equals-one query |
| 11 | alias-equivalence declaration |
| 12 | reified incidence `(relation_node, role, endpoint)` |
| 13 | command sequence |
| 14 | stage document `(version, declarations, payload)` |
| 15 | demonstration `(before,after)` |

`h=1` and the pair collection under `h=2` are semantically unordered.
Everything else is ordered. Semantic canonicalization sorts unordered children
by `CJ(child)` after resolving aliases. Renderers preserve surface order;
parsers do not silently sort.

Opaque symbols are assigned in canonical semantic-symbol order. For symbol
ordinal `j`, try counters `k=0,1,...` and choose the first unused:

```text
x || first16hex(
  PRF(
    split_key,
    "opaque-name",
    CJ({
      "cell_salt": cell_salt,
      "counter": k,
      "fold": fold,
      "presentation": presentation,
      "semantic_core_id": semantic_core_id,
      "split": split,
      "symbol_ordinal": j
    })
  )
)
```

`cell_salt` is `world-0`, `world-1`, `command-0`, `command-1`, or `shared-query`.
This collision-resolution rule is total.

## 4. Assessor-side semantic schemas

The surface tree has no domain tag. The assessor record does, and is never
candidate-visible. The following schema is the canonical union for WORLD and
COMMAND semantics:

```json
{
  "$id":"r12-ettr-il-v2-semantics.schema.json",
  "$schema":"https://json-schema.org/draft/2020-12/schema",
  "$defs":{
    "atom":{
      "additionalProperties":false,
      "properties":{
        "arguments":{"items":{"maximum":5,"minimum":0,"type":"integer"},"maxItems":2,"minItems":1,"type":"array"},
        "predicate":{"maximum":4,"minimum":0,"type":"integer"}
      },
      "required":["arguments","predicate"],
      "type":"object"
    },
    "term":{
      "additionalProperties":false,
      "properties":{
        "children":{"items":{"$ref":"#/$defs/term"},"maxItems":2,"type":"array"},
        "constructor":{"maximum":7,"minimum":0,"type":"integer"},
        "type":{"maximum":2,"minimum":0,"type":"integer"}
      },
      "required":["children","constructor","type"],
      "type":"object"
    },
    "world":{
      "oneOf":[
        {
          "additionalProperties":false,
          "properties":{
            "evidence_id":{"pattern":"^[0-9a-f]{64}$","type":"string"},
            "initial":{"items":{"$ref":"#/$defs/atom"},"type":"array"},
            "ontology":{"const":"horn"},
            "policy":{"enum":["persistent","derived_only"]},
            "theory_index":{"maximum":19,"minimum":0,"type":"integer"}
          },
          "required":["evidence_id","initial","ontology","policy","theory_index"],
          "type":"object"
        },
        {
          "additionalProperties":false,
          "properties":{
            "evidence_id":{"pattern":"^[0-9a-f]{64}$","type":"string"},
            "initial":{"$ref":"#/$defs/term"},
            "ontology":{"const":"rewrite"},
            "policy":{"enum":["contextual","root_only"]},
            "theory_index":{"maximum":14,"minimum":0,"type":"integer"}
          },
          "required":["evidence_id","initial","ontology","policy","theory_index"],
          "type":"object"
        },
        {
          "additionalProperties":false,
          "properties":{
            "evidence_id":{"pattern":"^[0-9a-f]{64}$","type":"string"},
            "initial":{"items":{"maximum":3,"minimum":0,"type":"integer"},"maxItems":4,"minItems":4,"type":"array"},
            "ontology":{"const":"resource"},
            "policy":{"enum":["atomic_deadlock","skip_blocked"]},
            "theory_index":{"maximum":59,"minimum":0,"type":"integer"}
          },
          "required":["evidence_id","initial","ontology","policy","theory_index"],
          "type":"object"
        }
      ]
    },
    "command":{
      "additionalProperties":false,
      "properties":{
        "depth":{"maximum":6,"minimum":1,"type":"integer"},
        "operations":{"items":{},"maxItems":6,"minItems":1,"type":"array"}
      },
      "required":["depth","operations"],
      "type":"object"
    }
  },
  "additionalProperties":false,
  "properties":{
    "command":{"$ref":"#/$defs/command"},
    "world":{"$ref":"#/$defs/world"}
  },
  "required":["command","world"],
  "type":"object"
}
```

The `operations` item type is selected assessor-side:

- Horn: an `atom`.
- Rewrite: integer 0 or 1, naming nullary constructor 0 or 1.
- Resource: integer 0, 1, or 2, naming the theory-local operator symbol.

`evidence_id` is `SHA256(CJ(canonical evidence))`. Candidate WORLD bytes contain
the evidence, not this digest or the theory index.

## 5. Exact command sequence semantics, depths 1-6

All command sequences have exactly `depth` operations. `S[0]` is the initial
state. `S[i]` is the state after operation `i`, one-indexed. The execution record
contains all `S[0..depth]`, the per-operation outcome, and terminal disposition.

### 5.1 Horn

The rule library, object types, predicates, theory ordering, grounding, and
least-fixed-point closure are exactly those in
`pipeline/cross_ontology_horn_board.py`.

For operation atom `a[i]`:

```text
S[i] = least_fixed_point(theory, sorted_unique(S[i-1] union {a[i]}))
```

Admission requires:

- `a[i]` is well typed;
- `a[i]` is absent from `S[i-1]`;
- `S[i] != S[i-1]`;
- for `i > 1`,
  `least_fixed_point(theory,S[0] union {a[i]}) != S[i]`.

The last clause makes each later operation causally dependent on its prefix,
not merely concatenated.

### 5.2 Rewrite

Constructors, rules, matching, contextual occurrence order, and normal-form
set equality are exactly those in
`pipeline/cross_ontology_rewrite_board.py`. Let `c[i]` be 0 or 1 and let
`wrap(t,c)` be constructor 5 with children `(t, GroundTerm(type=0,
constructor=c, children=()))`.

For `policy=contextual`, `NF` is exhaustive contextual normal forms. For
`policy=root_only`, only root redexes may be traversed. Then:

```text
N[i] = NF(theory, policy, wrap(S[i-1], c[i]))
```

If `|N[i]|=0`, disposition is `REJECT`. If `|N[i]|>1`, disposition is
`ABSTAIN`. Otherwise `S[i]` is the singleton member. Causal-rectangle admission
requires singleton `N[i]`, `S[i] != S[i-1]`, and for `i>1`:

```text
NF(theory,policy,wrap(S[0],c[i])) != (S[i],)
```

This defines all depths 1-6 without enlarging or misdescribing the old
four-node decreasing-rewrite challenge set.

### 5.3 Resource

Places, capacities, resource kinds, operator library, and theory ordering are
exactly those in `pipeline/cross_ontology_resource_board.py`. V2 removes only
the helper's length-3 validation guard. For each symbol, guards and consumption
are checked against the pre-step marking; production is applied atomically.

Under `atomic_deadlock`, a blocked operation returns the unchanged marking,
cursor `i-1`, status `deadlock`, and stops. Under `skip_blocked`, it returns the
unchanged marking for that step and continues. Successful completion returns
cursor `depth` and status `halt`.

For base, alpha, alias, reification, and type presentations, rectangle admission
requires every operation enabled, every successful operation changes the
marking, and for `i>1` applying operation `i` to `S[0]` does not produce
`S[i]`. For an `execution_semantics_twin` bundle, at least one operation MUST be
prefix-dependently blocked, the base must deadlock, the twin must skip it, and a
later operation must change the twin state. This is the only admitted exception
to "all operations succeed"; the disclosed depth still counts operations
actually evaluated by the twin policy.

### 5.4 Snapshot and ETTR transaction bound

Semantic operations are not ETTR transaction steps. This generator emits the
canonical assessor-side `S[0..depth]` snapshots and command atoms only.
`R12_ETTR_IL_V2_MATERIALIZATION_SPEC.md` is the sole authority for fixed slot,
type, relation, value-code, status, and generic transaction projection. The
generator MUST invoke that mapper during feasibility admission and retain its
replay receipt, but MUST NOT implement a second slot allocator or packet
difference compiler.

The materialized trace includes all command-register, ontology-state, cursor,
outcome, and final-disposition edits and has exactly 64 target positions with
a right-padded valid-step mask. Its valid prefix MUST contain 1-64 generic
transactions and replay exactly to every projected boundary and terminal
packet. A candidate whose materialized trace exceeds 64 valid transactions is
inadmissible; truncation or an alternative projection is forbidden.

## 6. Four exact byte renderers and parsers

Each stage renders one surface AST. A parser MUST consume the entire byte
string, produce exactly one AST, validate the schema, and satisfy
`render(parse(bytes)) == bytes`. Noncanonical but parseable input is rejected.

### 6.1 Renderer 0: canonical JSON

Bytes are `CJ(ast)`. The parser is strict JSON with duplicate-key rejection,
then schema validation and canonical re-render equality.

### 6.2 Renderer 1: prefix S-expression

Grammar is ABNF:

```text
document = node LF
node     = integer / symbol / call
integer  = "#" ("0" / DIGIT1-9 *DIGIT)
symbol   = "@" "x" 16HEXDIG-LOWER
call     = "(" head *(" " node) ")"
head     = "0" / "1" / "2" / "3" / "4" / "5" / "6" / "7" /
           "8" / "9" / "10" / "11" / "12" / "13" / "14" / "15"
```

`call` maps to `{"a":[children...],"h":head}`. No tabs, CR, leading/trailing
spaces, uppercase hex, or empty document are legal.

### 6.3 Renderer 2: record-delimited infix

Bytes begin `V2` followed by RS (`1e`). AST nodes are numbered in preorder from
0. Each record ends RS:

```text
I<id>=<decimal>
S<id>=x<16-lower-hex>
N<id>=<head>%<arity>%<child-id-0>%...%<child-id-n-1>
R=0
```

IDs and integers are canonical unsigned decimal. `%` is US-equivalent within a
record and the literal byte is `25`; RS is the actual control byte `1e`.
Records MUST appear in increasing ID order, every child ID MUST exceed its
parent ID, every nonroot node has exactly one parent, and `R=0` is last. The
infix `=` and framed records are normative; no LF is appended.

### 6.4 Renderer 3: reverse-child postfix

Grammar:

```text
document = token *(" " token) " !" LF
token    = integer / symbol / close
integer  = "#" canonical-unsigned-decimal
symbol   = "$x" 16HEXDIG-LOWER
close    = "^" head "/" canonical-unsigned-decimal
```

For a call, render children in reverse semantic order, then `^head/arity`.
The parser uses a stack: an integer or symbol pushes a leaf; a close pops
`arity` nodes, reverses the popped list, and pushes the call. Before `!` the
stack MUST contain exactly one node.

The reserved structural sets are respectively JSON punctuation, `()@#`, RS plus
`=%`, and `$^/!`. Ignoring LF and characters inside opaque symbols or decimal
integers, these sets are pairwise disjoint.

## 7. Semantic adapters and presentations

The canonical evidence constructors and exact fit/score theory pools remain:

| Ontology | Fit | Score-only |
|---|---|---|
| Horn | `1,3,4,6,8,9,11,13,14,16,17,18` | `0,2,5,7,10,12,15,19` |
| Rewrite | `0,4,5,7,9,12,14` | `1,2,3,6,8,10,11,13` |
| Resource | all `0..59` except score-only | `0,7,14,21,30,39,48,59` |

The fit-safe rewrite constructor is defined by generalizing the existing
`_build_parts` transform to any valid rewrite theory only for `base`,
`alpha_reorder`, and `alias_split`. The held-out-only guard remains mandatory
for the four score-only transforms. No challenge or oracle from a score-only
family enters fit.

Presentation transforms are deterministic:

- `base`: canonical declarations and evidence order.
- `alpha_reorder`: reverse every semantically unordered list; alpha-rename all
  local symbols using the presentation namespace; preserve ordered command
  operations.
- `alias_split`: choose the lowest `SHA256(CJ(symbol-definition))` symbol with
  at least two occurrences, create one alias, emit `h=11`, and replace
  occurrences 1,3,5,... in canonical preorder with the alias. Semantic decoding
  quotients the pair.
- `relation_reification`: replace each direct n-ary application by a fresh
  relation node and one `h=12` incidence per ordered role.
- `type_twin`: use the existing ontology-specific type-twin transformation and
  its assessor alignment; it MUST have a separating terminal witness.
- `execution_semantics_twin`: Horn uses derived-only closure, rewrite root-only
  reduction, and resource skip-blocked execution. It MUST have a separating
  terminal witness in the selected core.
- `ambiguity_deleted_twin`: remove the first evidence item in deletion-index
  order whose removal leaves at least two behavioral classes and a separating
  challenge. It is never a causal-rectangle view.

For each causal rectangle, equivalent WORLD factors must have identical initial
packet targets but distinct raw bytes. Let
`b = first_bit(SHA256(semantic_rectangle_id || "|layout"))`. WORLD layout for
command column `c` is `b xor c`; COMMAND declaration layout for world row `w`
is `b xor w`. Layout 0 is canonical order and layout 1 reverses only
semantically unordered declarations. Thus equivalent factors differ in bytes
without changing operation order or inserting inert padding. QUERY bytes are
identical across all four cells through the answer read position.

## 8. Query grammar, evaluator, and paraphrases

Queries are assessor-side Boolean expressions with the exact schema:

```json
{
  "$id":"r12-ettr-il-v2-query.schema.json",
  "$schema":"https://json-schema.org/draft/2020-12/schema",
  "additionalProperties":false,
  "properties":{
    "args":{"items":{"type":"integer"},"maxItems":3,"type":"array"},
    "op":{
      "enum":[
        "horn_has","horn_count_ge",
        "rewrite_root_is","rewrite_contains","rewrite_nodes_ge","rewrite_child_root_is",
        "resource_place_ge","resource_cursor_ge","resource_halt"
      ]
    }
  },
  "required":["args","op"],
  "type":"object"
}
```

Candidate enumeration order is exactly:

1. Horn: `horn_has(predicate,args)` in `all_ground_atoms()` order, then
   `horn_count_ge(k)` for `k=1..27`.
2. Rewrite: `rewrite_root_is(c)` for `c=0..7`;
   `rewrite_contains(c)` for `c=0..7`; `rewrite_nodes_ge(k)` for `k=1..16`;
   `rewrite_child_root_is(position,c)` for position 0 then 1 and `c=0..7`.
3. Resource: `resource_place_ge(place,k)` for place `0..3`, `k=1..3`;
   `resource_cursor_ge(k)` for `k=1..6`; then `resource_halt()`.

Evaluation is literal:

- `horn_has`: membership in sorted terminal closure.
- `horn_count_ge`: closure cardinality at least `k`.
- `rewrite_root_is`: root constructor equals `c`.
- `rewrite_contains`: any node constructor equals `c`.
- `rewrite_nodes_ge`: recursive node count at least `k`.
- `rewrite_child_root_is`: false if the child position is absent; otherwise its
  root constructor equals `c`.
- `resource_place_ge`: terminal multiplicity at place is at least `k`.
- `resource_cursor_ge`: terminal cursor is at least `k`.
- `resource_halt`: terminal process status is halt.

For each presentation bundle, a query is admissible only if every included view
is `ANSWER` and its labels on `(W0C0,W0C1,W1C0,W1C1)` change on all four edges.
Thus each vector is necessarily `0110` or `1001` and is exactly 2/2 balanced.

Select query slot 0 as the admissible query minimizing:

```text
SHA256(master_seed || "|query|0|" || CJ(query))
```

Select slot 1 similarly among queries with a distinct AST and a denotation
signature over the complete bounded ontology terminal universe that is neither
equal to nor the Boolean complement of slot 0's signature. If none exists, the
semantic core is inadmissible.

Paraphrase 0 renders `h=9(query-expression)`. Paraphrase 1 renders
`h=10(query-expression, integer(1))`. Their evaluator meanings are respectively
`q` and `q == true`; bytes MUST differ and labels MUST agree.

The query prefix is the rendered query AST followed by the renderer's ordinary
stage terminator and the ASCII bytes `R=`. The next byte is exactly `0` or `1`.
The token before that answer is the shared `query_read_index`.

## 9. Outcome, target, and disposition projection

Let `V` be the evidence-consistent theory set under the presented execution
policy and let `B` be the number of distinct bounded behavior signatures in
`V`.

```text
V empty                         -> REJECT, target null
V nonempty and B > 1            -> ABSTAIN, target null
V nonempty and B = 1, q false   -> ANSWER, target false
V nonempty and B = 1, q true    -> ANSWER, target true
```

ETTR status bits are:

| Projection | committed | halted | terminal opcode |
|---|---:|---:|---:|
| `OPEN` | 0 | 0 | forbidden terminal |
| `ANSWER` | 1 | 0 | 6 |
| `ABSTAIN` | 0 | 1 | 7 |
| `REJECT` | 1 | 1 | 8 |

`ABSTAIN` and `REJECT` are excluded from the binary denominator. Every admitted
causal rectangle is independently 2 false / 2 true for each query slot and
paraphrase. Therefore balance is exact per
`fold/split/ontology/stratum/query_slot/paraphrase` without target-conditioned
sampling.

Structured terminal outcomes remain assessor records. They are never replaced
by the Boolean target. Primary joint scoring still requires packet,
transactions, disposition, and answer independently.

## 10. Rectangle units and expansion

A `semantic_core` contains one theory/evidence instance, worlds W0/W1, commands
C0/C1, one depth, two query ASTs, and canonical outcomes for four cells. It is
the indivisible split, leakage-audit, and statistical-bootstrap cluster.

A `semantic_rectangle` is one presentation/renderer view of one core. It has
four packet cells and 16 rows.

A `causal_rectangle` is one `(query_slot,paraphrase)` slice of one semantic
rectangle. Its row tensor is:

```text
[[W0C0,W0C1],[W1C0,W1C1]]
```

It has exactly four rows and is one `ETTRCausalRectangle`.

Expansion order is:

```text
query_slot 0 paraphrase 0: W0C0,W0C1,W1C0,W1C1
query_slot 0 paraphrase 1: W0C0,W0C1,W1C0,W1C1
query_slot 1 paraphrase 0: W0C0,W0C1,W1C0,W1C1
query_slot 1 paraphrase 1: W0C0,W0C1,W1C0,W1C1
```

Admission requires all four terminal packets distinct, equal packet support
masks, W0 initial packets equal across commands, W1 initial packets equal
across commands, distinct W0/W1 initial packets, distinct equivalent-factor raw
bytes, identical query prefixes, and all four packet and label edge contrasts.

## 11. Exact split and presentation quotas

Folds retain v1 leave-one-ontology-out membership.

### 11.1 Training, per fit ontology

For each depth 1, 2, 3:

- 96 semantic cores;
- every core emits these four distinct views:

```text
pair 0 left:  base          renderer 0
pair 0 right: alpha_reorder renderer 1
pair 1 left:  base          renderer 1
pair 1 right: alias_split   renderer 0
```

- both pairs must pass the identity-map equivariance admission in the
  materialization specification; and
- therefore each depth has 192 invariant pairs, 384 semantic rectangles,
  1,536 causal rectangles, 1,536 packets, and 6,144 rows.

Per fit ontology this is 288 cores, 1,152 semantic rectangles, 4,608 packets,
4,608 causal rectangles, and 18,432 rows. Per fold totals remain 2,304 semantic
rectangles, 1,152 invariant pairs, 9,216 packets, and 36,864 rows.

### 11.2 Development and confirmation, non-`all_axes`

Each of the seven strata has 32 semantic cores. Every core has `base`,
`alpha_reorder`, and `alias_split`, producing exactly 96 semantic rectangles.

If composition is not shifted, core depth counts are 11, 11, 10 over depths
1,2,3. If shifted they are 11,11,10 over depths 4,5,6. Rotate which depth gets
10 by `uint64(SHA256(CJ(cell_key))) mod 3`.

Non-renderer-shift cells assign 16 cores to renderer 0 and 16 to renderer 1.
Renderer-shift development assigns 16 to renderer 2 and 16 to renderer 3.
Renderer-shift confirmation assigns all 32 to renderer 3.

### 11.3 Development and confirmation, `all_axes`

There are 24 semantic cores. Every core has exactly four views:

```text
base
relation_reification
type_twin
execution_semantics_twin
```

This yields 96 semantic rectangles, with 24 per presentation. Depths 4,5,6
receive eight cores each, hence 32 rectangles each. Development assigns 12
cores to renderer 2 and 12 to renderer 3. Confirmation uses renderer 3 only.

`ambiguity_deleted_twin` receives no row in this population for the impossibility
stated in Section 1. Its `ABSTAIN` evidence remains on the frozen seven-variant
score board.

Consequently each scored ontology/stratum still has 96 semantic rectangles,
384 packets, 384 causal rectangles, and 1,536 rows. Each scored split still has
2,304 semantic rectangles, 9,216 packets, and 36,864 rows.

### 11.4 Theory allocation

For a cell with `Q` cores and ordered theory pool `P` of size `m`, let
`q=floor(Q/m)`, `r=Q mod m`, and
`o=uint64(SHA256(master_seed || "|theory-offset|" || CJ(cell_key))) mod m`.
Every theory gets `q` cores; the first `r` theories in cyclic order starting at
`o` get one more. Core selection is then performed independently per theory
quota. This is the only remainder rule.

## 12. Candidate domain and deterministic selection

Initial state domains are exactly:

- Horn: the 378 `challenge_initials()` values.
- Rewrite: the 64 `challenge_terms()` values.
- Resource: the 81 `input_markings()` values.

World ownership prevents cross-split semantic-world reuse. For canonical world
bytes `w`, compute
`owner = uint64(SHA256(master_seed || "|world-owner|" || fold || ontology ||
CJ(w))) mod 3`; owners 0,1,2 mean train,development,confirmation. A world may
appear repeatedly inside its owner split but nowhere else.

For each theory/depth/owned unordered world pair, construct commands by beam
enumeration:

1. Start with the empty sequence and both initial states.
2. Extend every surviving prefix by the complete operation alphabet in the
   exact ontology order.
3. Reject extensions violating typing, execution, or dependency in either
   world, except the explicit execution-twin rule.
4. Deduplicate by command semantic AST.
5. Rank by
   `SHA256(master_seed || "|beam|" || fold || split || ontology || theory ||
   depth || CJ(world_pair) || CJ(sequence))`.
6. Retain the lowest 64 after each depth; ties use `CJ(sequence)` bytes.

At target depth, enumerate unordered pairs of the surviving commands in
lexicographic `CJ` order. Combine them with the world pair, compute all bundle
views and queries, and apply every admission rule.

The exact candidate tuple schema is:

```json
{
  "$id":"r12-ettr-il-v2-candidate.schema.json",
  "$schema":"https://json-schema.org/draft/2020-12/schema",
  "additionalProperties":false,
  "properties":{
    "commands":{"items":{"pattern":"^[0-9a-f]{64}$","type":"string"},"maxItems":2,"minItems":2,"type":"array"},
    "depth":{"maximum":6,"minimum":1,"type":"integer"},
    "fold":{"maximum":2,"minimum":0,"type":"integer"},
    "generator_ordinal":{"maximum":9223372036854775807,"minimum":0,"type":"integer"},
    "ontology":{"enum":["horn","rewrite","resource"]},
    "opaque_seed":{"maximum":9223372036854775807,"minimum":0,"type":"integer"},
    "presentations":{"items":{"enum":["base","alpha_reorder","alias_split","relation_reification","type_twin","execution_semantics_twin"]},"type":"array","uniqueItems":true},
    "queries":{"items":{"pattern":"^[0-9a-f]{64}$","type":"string"},"maxItems":2,"minItems":2,"type":"array"},
    "renderer":{"maximum":3,"minimum":0,"type":"integer"},
    "schema":{"const":"r12-ettr-il-v2-candidate"},
    "split":{"enum":["train","development","confirmation"]},
    "stratum":{"enum":["seen_id","rule","composition","renderer","rule_composition","rule_renderer","composition_renderer","all_axes"]},
    "theory_instance":{"pattern":"^[0-9a-f]{64}$","type":"string"},
    "theory_pool_index":{"minimum":0,"type":"integer"},
    "worlds":{"items":{"pattern":"^[0-9a-f]{64}$","type":"string"},"maxItems":2,"minItems":2,"type":"array"}
  },
  "required":["commands","depth","fold","generator_ordinal","ontology","opaque_seed","presentations","queries","renderer","schema","split","stratum","theory_instance","theory_pool_index","worlds"],
  "type":"object"
}
```

Hashes in the tuple are SHA-256 of canonical assessor-side semantic ASTs.
Within each exact
`(fold,split,ontology,stratum,theory_pool_index,depth,renderer,presentation_bundle)`
cell, candidates are emitted before admissibility filtering by ascending
canonical WORLD-pair bytes and then ascending canonical COMMAND-pair bytes.
`generator_ordinal` is the zero-based emission counter in that cell and is
never compacted after rejection.

`opaque_seed` is the low 63 bits of:

```text
PRF(
  split_key,
  "opaque-name",
  CJ(candidate_without_opaque_seed)
)
```

An admissible candidate's selection key is:

```text
PRF(split_key, "candidate-rank", CJ(candidate))
```

For each exact theory quota, choose the lowest key interpreted as 32 unsigned
big-endian bytes, then `CJ(candidate)` as tie breaker, while enforcing unique
semantic core IDs. Selection never reruns after presentation expansion.

`theory_instance` includes evidence and presentation seed. Underlying law
identity is separately recorded as `law_signature`. Seen-theory strata
necessarily reuse law signatures; claiming zero law-signature overlap there
would contradict the definition of `seen_id`. Literal theory-instance bytes and
worlds remain disjoint.

Command identities are:

```text
template_id = SHA256(CJ(command semantic AST))
instance_id = SHA256(CJ({law_signature,world_semantic_id,command semantic AST}))
```

`instance_id` MUST be disjoint across splits. `template_id` may overlap and its
overlap matrix MUST be reported. This resolves the depth-1 cardinality
contradiction without mislabeling primitive reuse as novel composition.

## 13. Training schedule unit

The indivisible training schedule unit is one admitted invariant pair. A
microbatch contains exactly one pair = two semantic rectangles = eight causal
rectangles = 32 rows. Four microbatches form one update containing four pairs
= eight semantic rectangles = 32 causal rectangles = 128 rows.

The exact five-seed repetition and remainder schedule is Section 4 of
`R12_ETTR_IL_V2_ARMS_AND_STATISTICS_SPEC.md`. It orders all 1,152 fitting pair
IDs separately for each `(fold,model_seed,epoch)`, consumes 20 complete epochs
plus the first 960 pairs of epoch 20, and exposes 24,000 invariant pairs =
48,000 semantic rectangles over 6,000 updates. All four causal slices of each
semantic rectangle and all 16 row alignments of each invariant pair remain in
one microbatch. No component spec may substitute an unpaired rectangle or
causal-rectangle schedule.

## 14. Cardinality and feasibility certificate

Before any row file is written, the generator MUST emit and validate a
`r12-ettr-il-v2-cardinality-report` in canonical JSON. For every
`fold/split/ontology/stratum/theory/depth/renderer/presentation-bundle` cell it
records:

- world domain count and owned-world count;
- unordered owned-world-pair count;
- operation alphabet size;
- raw template count (`27^d`, `2^d`, or `3^d`);
- beam survivor counts after each prefix depth;
- command-pair count;
- dependency-pass count;
- four-distinct-terminal count;
- packet-item and transaction-bound pass count;
- checkerboard-one-query and checkerboard-two-query counts;
- presentation-bundle pass count;
- admissible unique core count;
- required core quota and surplus.

The report passes only if every surplus is nonnegative, every selected core has
two query semantics, and all selected IDs are unique. It also proves these
static facts:

- initial-state domain sizes are Horn 378, rewrite 64, resource 81;
- operation-template counts at depth `d` are `27^d`, `2^d`, `3^d`;
- resource depth-1 semantic-template disjointness across three splits is
  impossible because each rectangle needs two of only three templates;
- the replacement instance-disjointness test is satisfiable only if owned world
  domains are disjoint and each selected instance ID is unique;
- every transaction trace is at most 64 steps and packet union at most 64 slots.

The generator MUST NOT assert in advance that dynamic candidate counts are
sufficient. Only the exhaustive report over the bounded beam domain proves
feasibility. If any cell is short, the scientifically valid result is
`semantic_generator_infeasible_at_<cell>`. Increasing beam width, changing a
query, or reallocating a quota requires protocol v3.

## 15. Canonical selected-row record

Each assessor JSONL row has this exact top-level shape:

```json
{
  "$id":"r12-ettr-il-v2-row.schema.json",
  "$schema":"https://json-schema.org/draft/2020-12/schema",
  "additionalProperties":false,
  "properties":{
    "assessor":{"type":"object"},
    "candidate":{
      "additionalProperties":false,
      "properties":{
        "command_hex":{"pattern":"^(?:[0-9a-f]{2})+$","type":"string"},
        "query_hex":{"pattern":"^(?:[0-9a-f]{2})+$","type":"string"},
        "world_hex":{"pattern":"^(?:[0-9a-f]{2})+$","type":"string"}
      },
      "required":["command_hex","query_hex","world_hex"],
      "type":"object"
    },
    "causal_rectangle_id":{"pattern":"^[0-9a-f]{64}$","type":"string"},
    "command_index":{"maximum":1,"minimum":0,"type":"integer"},
    "fold":{"maximum":2,"minimum":0,"type":"integer"},
    "paraphrase":{"maximum":1,"minimum":0,"type":"integer"},
    "query_slot":{"maximum":1,"minimum":0,"type":"integer"},
    "row_id":{"pattern":"^[0-9a-f]{64}$","type":"string"},
    "schema":{"const":"r12-ettr-il-v2-row"},
    "semantic_core_id":{"pattern":"^[0-9a-f]{64}$","type":"string"},
    "semantic_rectangle_id":{"pattern":"^[0-9a-f]{64}$","type":"string"},
    "split":{"enum":["train","development","confirmation"]},
    "world_index":{"maximum":1,"minimum":0,"type":"integer"}
  },
  "required":["assessor","candidate","causal_rectangle_id","command_index","fold","paraphrase","query_slot","row_id","schema","semantic_core_id","semantic_rectangle_id","split","world_index"],
  "type":"object"
}
```

The open-ended `assessor` object is not candidate-visible, but its canonical
required schema MUST be versioned by the later materializer. This semantic spec
requires it to contain at least ontology, stratum, depth, renderer,
presentation, theory/law/evidence hashes, semantic world/command/query ASTs,
all intermediate outcomes, packet projections, generic transactions,
disposition, Boolean target or null, alignments, and all parent IDs.

`row_id = SHA256(CJ(row_without_row_id))`. Parent IDs are hashes of their
canonical records excluding their own ID field.

## 16. Required conformance checks

A semantic generator conforms only if all checks pass:

1. Recompute the master seed and tokenizer hash.
2. Validate every internal and emitted object against the schemas above.
3. For all four renderers and all emitted stages, parse, semantic-decode,
   re-render, and require byte equality.
4. Require renderer byte inequality for the same surface AST.
5. Replay every ontology execution with the production oracle and an
   independently implemented oracle; require exact intermediate and terminal
   equality.
6. Replay every generic transaction and require exact packet equality.
7. Validate every causal rectangle with the current `ETTRCausalRectangle`
   invariants.
8. Verify exact quotas, expansion factors, row order, 2/2 label balance, and
   presentation pairing.
9. Verify no withheld ontology or score-only theory/presentation/renderer enters
   fit.
10. Verify split-disjoint world IDs, command instance IDs, opaque names, and raw
    rows; report, but do not prohibit, primitive template overlap.
11. Verify the cardinality report before writing split files.
12. Verify that `ambiguity_deleted_twin`, `ABSTAIN`, and `REJECT` are absent
    from the causal JSONL population.

Failure is terminal and names the first canonical cell and invariant. No partial
split has standing.

## 17. Audit-blocker closure map

| Blocker | V2 resolution |
|---:|---|
| 1 | Sections 2, 11, 12, and 15 define roots, schemas, domains, ordering, ties, and IDs. |
| 2 | Section 5 defines dependent depths 1-6, intermediate states, failure, packet differences, and transaction bounds. |
| 3 | Section 10 expands each 16-row view into four valid four-row causal rectangles. |
| 4 | Section 13 fixes the optimization unit and delegates the exact five-seed repetition and remainder to the arms/statistics specification. |
| 5 | Sections 3 and 6 define one shared AST and four total strict byte codecs. |
| 6 | Candidate bytes contain no domain code; assessor tags remain sealed. |
| 7 | Section 7 defines fit-safe rewrite transforms for exactly three fit presentations. |
| 8 | Section 11 allocates every presentation, renderer, depth, and theory quota. |
| 9 | Section 8 defines finite query grammars, evaluators, paraphrases, and selection. |
| 10 | Section 9 defines the complete disposition/Boolean projection and denominator. |
| 14 | Sections 1, 12, and 14 reject impossible template disjointness and replace it with provable instance disjointness. |

## 18. Remaining blockers outside this specification

This document deliberately does not claim closure of implementation-audit
blockers 11, 12, 13, 15, or 16. In particular:

- no CPU materializer currently maps these semantic records into production
  `ETTRContinuationBatch` tensors and `ETTRVariantAlignment`;
- the binding-deranged control remains contradictory until its preserved and
  changed target components are redefined;
- confirmation encryption, signing, and open-once custody remain unspecified;
- graph-isomorphism, normalized overlap, and metadata-classifier gate
  implementations remain to be frozen;
- source paths and hashes for a future v2 generator/materializer do not exist.

Therefore this specification makes semantic generation reconstructible but does
not authorize data materialization, training, jobs, or a capability claim.
