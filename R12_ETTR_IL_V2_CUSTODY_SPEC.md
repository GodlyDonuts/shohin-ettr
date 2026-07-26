# R12 ETTR Isolated Learnability v2 Custody Specification

**Protocol:** `R12-ETTR-IL-v2`
**Document schema:** `r12-ettr-il-v2-custody-spec-v1`
**Status:** Phase-1 custody and source-deletion mechanics implemented; no
production population, job, training, development opening, confirmation
opening, or capability claim is authorized by this document
**Repairs:** implementation-audit blockers 1, 6, 13, 15, and 16
**Does not repair:** implementation-audit blockers 2-5, 7-12, or 14

## 1. Scope and mandatory interpretation

This document replaces the hash-custody, candidate-metadata, overlap-audit,
source-identity, and confirmation-opening clauses of
`R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md` under the new protocol identifier
`R12-ETTR-IL-v2`. The v1 split and fold commitments are retired because their
preimages were not published. They must not be accepted as aliases for the v2
commitments below.

The semantic generators, typed command ASTs, rectangle expansion, query
grammars, outcome projection, structural presentations, ETTR target
materializer, update schedule, and binding-deranged control now have Phase-1
implementations. A conforming custody instance must still stop before fitting
until their exact source inventory, literal production population, and
population-level audits are committed in Phase 2. Custody completeness is not
learned-capability evidence.

The terms `MUST`, `MUST NOT`, `REQUIRED`, `SHALL`, and `SHALL NOT` are
normative. Any unrecognized field, schema version, enum value, file, mount,
descriptor, signer, seed domain, or state transition is fatal.

## 2. Primitive encodings and identifiers

### 2.1 Canonical JSON

`CJ1(x)` is the repository's existing canonical JSON contract:

```python
(json.dumps(
    x,
    ensure_ascii=True,
    allow_nan=False,
    sort_keys=True,
    separators=(",", ":"),
) + "\n").encode("ascii")
```

Inputs are limited to objects, arrays, strings, integers, booleans, and null.
Floats, duplicate object keys, lone surrogates, non-NFC strings, and integers
outside `[-9223372036854775808, 9223372036854775807]` are rejected. Protocol
keys, enum values, paths, IDs, and domains are ASCII. Readers must parse and
then require `input_bytes == CJ1(parsed_value)`. JSONL is a concatenation of
`CJ1(row)` records; an empty file and a missing final LF are invalid.

The following aliases are used:

- `H(b) = SHA-256(b)`, lowercase 64-character hexadecimal when encoded.
- `HFILE(path) = H(literal file bytes)`.
- `hex32` is exactly 64 lowercase hexadecimal characters representing 32
  bytes.
- `hex64` is exactly 128 lowercase hexadecimal characters representing 64
  bytes.
- `u16be`, `u32be`, and `u64be` are unsigned big-endian integers of the named
  width.
- A relative path is nonempty ASCII, uses `/`, has no empty, `.` or `..`
  component, has no leading `/`, and is already normalized.
- Arrays called `files`, `entries`, `signatures`, or `events` are sorted by
  the key stated for their schema and contain no duplicate key.

### 2.2 Signed objects

All signatures are raw Ed25519 signatures. A detached signature object has
exactly:

```text
schema                 "r12-ettr-il-v2-detached-signature-v1"
payload_sha256          hex32
role                    one authorized role
public_key_hex          hex32
public_key_sha256       H(bytes.fromhex(public_key_hex))
signature_domain        one exact domain from Section 8
signature_hex           hex64
```

Verification message is
`ASCII(signature_domain) || 0x00 || bytes.fromhex(payload_sha256)`.
Signing a JSON object's embedded empty-signature form is forbidden. Every
payload is signed by hash as specified above. Signature objects themselves are
encoded with `CJ1`.

This deliberately reuses the Ed25519, immutable single-link read, write-once,
and `CJ1` patterns in `train/ettr_factorial_authority.py`,
`train/ettr_factorial_signed_custody.py`, and
`train/ettr_deployment_contract.py`. Their factorial schemas and signature
domains are not reusable for v2.

## 3. Literal commitment preimages

For each block below, the preimage is exactly the single ASCII JSON line
between the markers plus one LF. Markers and Markdown fences are excluded.
The parser must reproduce the stated byte count and SHA-256 before using it.

### 3.1 Master commitment

The master preimage is exactly 58 ASCII bytes including its final LF:

```text
R12_ETTR_IL_V2|2026-07-26|custody-complete|source-deleted
```

Its SHA-256 is:

`8d3201be7e2f0a6e047223a67342971df70bd8be533ab50c80dd42e2208432c6`

### 3.2 Split specification

`SPLIT-SPEC-PREIMAGE-BEGIN`

```json
{"candidate_tuple_fields":["schema","fold","split","ontology","stratum","theory_instance","theory_pool_index","worlds","commands","depth","renderer","presentations","queries","opaque_seed","generator_ordinal"],"candidate_tuple_schema":"r12-ettr-il-v2-candidate","counts":{"score_rectangles_per_ontology_stratum":96,"train_rectangles_per_depth_per_fit_ontology":384,"train_rectangles_per_fit_ontology":1152},"folds":[0,1,2],"ontologies":["horn","rewrite","resource"],"presentations":["base","alpha_reorder","alias_split","relation_reification","type_twin","execution_semantics_twin"],"protocol":"R12-ETTR-IL-v2","renderer_ids":[0,1,2,3],"schema":"r12-ettr-il-v2-split-spec-v2","seed_domains":["candidate-rank","opaque-name","renderer-choice","presentation-choice","query-choice","paraphrase-choice","donor-derangement","classifier-fold","classifier-permutation"],"splits":["train","development","confirmation"],"strata":["seen_id","rule","composition","renderer","rule_composition","rule_renderer","composition_renderer","all_axes"]}
```

`SPLIT-SPEC-PREIMAGE-END`

Byte count: `1033`
SHA-256:
`a09f82684c8a118a633b0bb23e244de961166ebdd3593485d897c8c27deb9747`

### 3.3 Fold specifications

`FOLD-0-PREIMAGE-BEGIN`

```json
{"fit_ontologies":["rewrite","resource"],"fold":0,"protocol":"R12-ETTR-IL-v2","schema":"r12-ettr-il-v2-fold-spec-v1","seed_context":"fold/0","withheld_ontology":"horn"}
```

`FOLD-0-PREIMAGE-END`

Byte count: `169`
File SHA-256:
`a4293ae0cf972abfdfb155ad4268dceeafcb7d9ebc4df975002d08896ea65ab8`

`FOLD-1-PREIMAGE-BEGIN`

```json
{"fit_ontologies":["horn","resource"],"fold":1,"protocol":"R12-ETTR-IL-v2","schema":"r12-ettr-il-v2-fold-spec-v1","seed_context":"fold/1","withheld_ontology":"rewrite"}
```

`FOLD-1-PREIMAGE-END`

Byte count: `169`
File SHA-256:
`6e0b45dbdb28af3684db1649767a50bae6dd5ba9d8c7fdfbae6b4edad16af425`

`FOLD-2-PREIMAGE-BEGIN`

```json
{"fit_ontologies":["horn","rewrite"],"fold":2,"protocol":"R12-ETTR-IL-v2","schema":"r12-ettr-il-v2-fold-spec-v1","seed_context":"fold/2","withheld_ontology":"resource"}
```

`FOLD-2-PREIMAGE-END`

Byte count: `169`
File SHA-256:
`ff127ca341c8215fe4d08883d2aef9a29a3ca810adc8cf44fbe7a565c4961f68`

Fold commitment is:

```text
H(
  ASCII("R12-ETTR-IL-v2") || 0x00 ||
  ASCII("fold-commitment") || 0x00 ||
  bytes.fromhex(split_spec_sha256) ||
  bytes.fromhex(fold_spec_file_sha256)
)
```

The resulting commitments are:

| Fold | Fold commitment |
|---:|---|
| 0 | `cd21d2501e57a275267080ceec35089f5d89e8c83c4d7e3a2ac22c2a39f6eb60` |
| 1 | `c8509e61b93cbac341c42a2cd73e5d58cd02edbb0eff0b06173df729d83c7d01` |
| 2 | `8487125d8354be89ff15dceca987a06af2e2dfd457890b387e696002771768b5` |

## 4. Candidate tuples, enumeration, and seed domains

### 4.1 Tuple schema

A candidate tuple has exactly the fields in the split preimage, in the
following types:

| Field | Type and constraint |
|---|---|
| `schema` | `r12-ettr-il-v2-candidate` |
| `fold` | integer in `0,1,2` |
| `split` | `train`, `development`, or `confirmation` |
| `ontology` | `horn`, `rewrite`, or `resource`; assessor-only |
| `stratum` | one split-spec stratum |
| `theory_instance` | hex32 over the canonical assessor-side theory/evidence instance |
| `theory_pool_index` | nonnegative integer index in the frozen fit or score-only pool |
| `worlds` | exactly two distinct hex32 semantic-world identities |
| `commands` | exactly two distinct hex32 world-bound command identities |
| `depth` | integer `1..6` |
| `renderer` | integer `0..3` |
| `presentations` | the exact unique presentation bundle admitted for the cell |
| `queries` | exactly two distinct hex32 semantic query identities |
| `opaque_seed` | low signed-63-bit integer from the `opaque-name` PRF domain |
| `generator_ordinal` | integer `0..9223372036854775807` |

`canonical_tuple_bytes = CJ1(tuple)`. Ontology and custody metadata exist only
in assessor records. They are not rendered or mounted for a candidate.
`ambiguity_deleted_twin` is not a v2 causal-population presentation and is
therefore absent from the split preimage; it remains score-only on the
pre-existing sealed seven-variant board.

### 4.2 Enumeration and selection

The future semantic specification must expose one finite iterator for each
`(fold, split, ontology, stratum)` in ascending `generator_ordinal`. An
iterator must emit every syntactically constructible tuple exactly once before
admissibility filtering. Generator ordinal is assigned before filtering and
must not be compacted after rejection.

The admissibility predicate is a pure function of the canonical semantic ASTs
and the future committed semantic specification. It must not inspect tuple
rank, model output, optimizer state, development result, confirmation result,
or encryption randomness. Until that predicate and iterator are source-frozen,
materialization remains unauthorized.

For each cell:

1. reject malformed tuples and duplicate `canonical_tuple_bytes`;
2. evaluate admissibility;
3. compute `rank = PRF(split_key, "candidate-rank", canonical_tuple_bytes)`;
4. sort by `(rank as 32 unsigned big-endian bytes, canonical_tuple_bytes)`;
5. take the first exact quota;
6. fail if the quota is unavailable; never relax, reroll, or borrow.

SHA-256 rank collisions are resolved by tuple bytes. Two different tuples with
identical tuple bytes are a generator error.

### 4.3 Keys and deterministic domains

The public seed root is:

```text
K_public = H(ASCII("R12-ETTR-IL-v2/public-seed-root\n"))
         = bba84905d8f0d574ddb7e348bde9dc83b19b55a0374984988ac47664c07128a4
```

For train and development:

```text
split_key =
  HMAC-SHA256(
    key=K_public,
    msg=ASCII("R12-ETTR-IL-v2") || 0x00 ||
        ASCII("split-key") || 0x00 ||
        u16be(fold) || 0x00 || ASCII(split)
  )
```

For confirmation, `K_confirmation[fold]` is 32 CSPRNG bytes generated inside
the independent custodian after the implementation source and authority
record are sealed. It is never stored in a repository, manifest, command
line, environment variable, log, or candidate account. Its public commitment
is:

```text
H(
  ASCII("R12-ETTR-IL-v2") || 0x00 ||
  ASCII("confirmation-seed") || 0x00 ||
  u16be(fold) || K_confirmation[fold]
)
```

`split_key = K_confirmation[fold]` for confirmation.

The only PRF is:

```text
PRF(K, label, context) =
  HMAC-SHA256(
    key=K,
    msg=ASCII("R12-ETTR-IL-v2") || 0x00 ||
        ASCII("seed") || 0x00 ||
        u16be(len(ASCII(label))) || ASCII(label) ||
        u32be(len(context)) || context
  )
```

Allowed labels are exactly the nine `seed_domains` in the split preimage.
Mutable PRNG state shared between labels, folds, or splits is forbidden. A
stream is `PRF(K,label,context || u64be(counter))` for counters from zero.
Uniform integer selection from `[0,n)` reads successive unsigned 64-bit words
and rejects values at or above `floor(2^64/n)*n`; result is `word mod n`.

Semantic plaintext and all train/development choices are deterministic.
Confirmation semantic plaintext is deterministic conditional on its secret
seed. Confirmation seeds, encryption keys, nonces, key IDs, attempt IDs, and
timestamps are randomized custody bytes and never affect semantic selection.

## 5. Tokenizer identity

The only admitted tokenizer payload has:

| Field | Frozen value |
|---|---|
| Logical path | `inputs/tokenizer/tokenizer.json` |
| Literal byte count | `2309567` |
| Literal SHA-256 | `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4` |
| Vocabulary size, with added tokens | `32768` |
| Vocabulary size, without added tokens | `32768` |
| Payload encoding | strict UTF-8 |
| API | Hugging Face `tokenizers.Tokenizer.from_str` |
| Reference package | `tokenizers==0.22.2` |
| Special tokens during audit | disabled |
| Truncation | disabled |
| Padding | disabled |

The runtime manifest must additionally pin the CPython executable SHA-256,
platform tag, `tokenizers` wheel or installed-distribution file inventory, and
every shared-library SHA-256. Merely recording version `0.22.2` is
insufficient. The tokenization identity object has schema
`r12-ettr-il-v2-tokenizer-identity-v1` and exactly the table fields plus
`runtime_inventory_sha256`.

Required conformance vectors are:

| Input hex | Token IDs |
|---|---|
| empty | `[]` |
| `7b7d` | `[7965]` |
| `28776f726c64207829` | `[20,7891,601,21]` |
| `617c620a` | `[77,104,78,211]` |
| `7b226b223a317d0a` | `[31736,87,1418,29,105,211]` |
| `415f392d7a` | `[45,75,37,25,102]` |

The four verbose renderer byte strings are strict ASCII decoded one-to-one and
encoded with `add_special_tokens=False` for parser and leakage conformance.
Model-visible WORLD, COMMAND, and QUERY use the separately frozen token-native
structural transport in the semantic-generator specification. Its exact
fixed-width token arrays, parsed logical prefix, deterministic cover suffix,
and codebook identity are all recorded assessor-side. Any
package/runtime/vector/codebook mismatch fails before overlap audits.

## 6. Candidate metadata and ontology isolation

Candidate-visible stage inputs consist only of raw WORLD, COMMAND, or QUERY
bytes and the frozen model/runtime inputs appropriate to that stage. Candidate
mounts, file names, JSON wrappers, batch records, environment variables, row
order sidecars, and descriptors must not contain:

- ontology/family names or numeric domain codes;
- fold, split, stratum, renderer, theory, presentation, or variant IDs;
- tuple, row, packet, donor, oracle, target, answer, or disposition IDs;
- source paths, generator ordinals, seed commitments, or ciphertext metadata.

The existing stable `"d"` field used by factorial qualification payloads is
forbidden. Removing the field name while retaining a stable per-ontology
integer is also forbidden. Candidate syntax must use one ontology-neutral AST
envelope and generic type tags shared across all three ontologies. A parser
must infer structure from the anonymous syntax rather than receive an
ontology selector.

Every materializer field has one classification:

`candidate_semantic`, `candidate_opaque_split_local`, `assessor_only`, or
`custody_only`.

The classification map is source-frozen and exhaustive. Rendering may consume
only the first two classes. A dynamic taint audit reconstructs every candidate
byte from classified leaves and fails if an assessor/custody leaf contributes
to candidate bytes, token IDs, lengths, order, path, or package boundaries.

The constant-token audit extracts every non-opaque lexical atom from candidate
bytes. Each atom must occur in at least one admitted row of every ontology in
the same split. An atom present in one ontology and absent from another is a
hard failure even if the statistical metadata classifier passes.

## 7. Exact leakage audits

All audits operate across the union of all folds. A collision is fatal when
the same fingerprint occurs in two different split names, even if fold,
ontology, renderer, or presentation differs. Duplicate fingerprints within
one split are reported but do not satisfy a quota twice.

### 7.1 Fingerprints

For every row, compute:

| Name | Exact preimage |
|---|---|
| `raw_row` | `CJ1({"command_hex":...,"query_hex":...,"world_hex":...})` |
| `semantic_world` | `CJ1` of alpha-normalized, renderer-free world AST |
| `theory` | `CJ1` of alpha-normalized theory/rule AST |
| `semantic_command` | `CJ1` of alpha-normalized world-free command AST |
| `bound_command` | `CJ1({"command":semantic_command_object,"world_sha256":semantic_world_hash})` |
| `opaque_name` | literal ASCII bytes of each symbol-table value |
| `stage_token_sequence` | `ASCII(stage) || 0x00 ||` concatenated `u32be(token_id)` |
| `package_token_sequence` | three stage sequences in WORLD, COMMAND, QUERY order |

The v2 meaning of the v1 word `command` is `bound_command`. Bound-command
overlap across splits must be zero. World-free semantic-command overlap is
reported by depth and ontology but is not itself fatal; this resolves the
depth-1 cardinality ambiguity without pretending the operation alphabet is
larger than it is.

Opaque names are extracted only from the canonical AST symbol table, sorted by
raw bytes, and must match `[A-Za-z][A-Za-z0-9_]{15,63}`. Substring scanning is
not a substitute. Opaque-name overlap is zero across split names.

### 7.2 Normalized 13-grams

For each stage and the WORLD/COMMAND/QUERY concatenation:

1. require ASCII;
2. map `A-Z` to `a-z`;
3. map each maximal byte run outside `a-z0-9` to one ASCII space;
4. strip leading/trailing spaces;
5. split on one space;
6. require at least 13 tokens;
7. encode each consecutive 13-token window by joining with one space.

Every window is indexed by literal bytes. Cross-split overlap must be zero.
No Unicode, locale, stemming, stop-word removal, or opaque-name replacement
occurs.

### 7.3 Complete graph-isomorphism audit

Each row supplies an assessor-only graph object with exactly:

```text
schema  "r12-ettr-il-v2-audit-graph-v1"
nodes   array of {"id": integer, "color": ASCII string}
edges   array of {"src": integer, "dst": integer, "color": ASCII string}
```

Node IDs are exactly `0..N-1`; edges are a multiset sorted by
`(src,dst,color)` and may be directed self-edges. Colors are from the
source-frozen ontology-neutral audit vocabulary. Opaque names, ontology,
split, renderer, presentation, theory index, and source ordinals are absent.
The graph contains the complete theory, WORLD, COMMAND, and QUERY ASTs with
root-kind, generic type, argument-position, binding, rule-incidence, operation
order, and query-reference edges.

Canonical labeling is exact, not Weisfeiler-Lehman:

1. partition nodes by literal color;
2. enumerate the Cartesian product of every permutation within each color
   class, classes ordered by color bytes;
3. for each complete relabeling, assign new IDs by class order and permutation
   order;
4. build `CJ1({"edges":[...],"node_colors":[...]})`, with relabeled edges
   sorted by `(src,dst,color)`;
5. select the lexicographically least byte string;
6. `graph_iso_sha256 = H(selected_bytes)`.

This exhaustive algorithm is the normative oracle. An optimized
individualization/refinement or nauty/bliss implementation is admissible only
if it returns byte-identical canonical forms to the exhaustive oracle on every
official graph and on all graphs up to eight nodes in the complete
color/directed-edge conformance suite. If exhaustive official evaluation
exceeds the frozen resource bound, the board is infeasible and fails; a
probabilistic or incomplete substitute is forbidden.

Cross-split `graph_iso_sha256` overlap must be zero. Every collision report
includes both row hashes and a color-preserving node bijection independently
verified by a second implementation.

### 7.4 Metadata classifier

The classifier receives no raw bytes or token IDs. Its feature vector, in this
exact order, is:

```text
world_byte_count, command_byte_count, query_byte_count,
package_byte_count,
world_token_count, command_token_count, query_token_count,
package_token_count,
world_mask_count, command_mask_count, query_mask_count,
row_ordinal, package_file_byte_count
```

`package_byte_count` and `package_token_count` are stage sums.
`*_mask_count` is the number of one bits before padding and therefore must
equal the corresponding token count. `row_ordinal` is zero-based position in
the immutable split file. `package_file_byte_count` is the literal size of
that file. No other feature is allowed.

The implementation is scikit-learn `1.9.0`
`LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
fit_intercept=True, class_weight="balanced", max_iter=2000, tol=1e-10)`.
Features are standardized using training-fold means and population standard
deviations; a zero deviation maps to zero. `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` are `1`.
The exact Python, NumPy, SciPy, scikit-learn, BLAS, and shared-library
inventories are hash-pinned. Convergence warnings or failure to converge are
fatal.

Five-fold grouped cross-validation is used on development. Group is semantic
rectangle ID. Fold is the unsigned integer represented by the first eight
bytes of
`H(ASCII("R12-ETTR-IL-v2/classifier-fold") || 0x00 ||
bytes.fromhex(rectangle_sha256)) mod 5`. Four folds train and one tests; each
serves once. Missing classes in any train/test partition are fatal.

Tasks are:

- binary `answer_false` versus `answer_true`, excluding non-answer
  dispositions;
- three-class ontology;
- eight-class stratum;
- four-class renderer;
- disposition over the complete source-frozen disposition alphabet.

Metric is balanced accuracy for binary and macro recall for multiclass.
Required point bounds are `<=0.52` for binary and `<=1/K+0.02` for a K-class
task.

The confidence rule uses 10,000 deterministic group-label permutations per
task. Eligible rows are sorted by row SHA-256 inside each rectangle. A group
label vector is the resulting ordered labels. Groups are partitioned by label
vector length. Within each length bucket, source vectors are ordered by
rectangle SHA-256; destination groups are ordered by
`(PRF(K_public,"classifier-permutation",
ASCII(task)||0x00||u32be(i)||bytes.fromhex(rectangle_sha256)),
rectangle_sha256)`. Source vector `j` is assigned to destination group `j`.
This preserves the complete label multiset and group shape. The complete
cross-validation procedure is rerun. The one-sided empirical p-value is
`(1 + count(permuted_metric >= observed_metric))/10001`. It must be at least
`0.05`. Both the point bound and p-value rule must pass.

The leakage report records package/runtime identity, feature matrix hash,
group/fold hash, predictions, confusion matrices, metrics, permutation metric
hash, and all fingerprint index roots. It contains no confirmation plaintext
and is signed by the audit authority.

## 8. Authorities, keys, and signatures

Required public-key roles are:

| Role | Function |
|---|---|
| `offline_root` | delegates roles and countersigns final dataset root |
| `dataset_publisher` | signs plaintext/public manifest roots |
| `confirmation_custodian` | owns confirmation seeds/DEKs and signs envelopes/ledger |
| `opening_authorizer` | signs one opening authorization after development pass |
| `launch_verifier` | signs measured stage-launch receipts |
| `leakage_auditor` | signs independent leakage report |

All six Ed25519 keys are distinct. Private keys are generated independently
and never placed in the repository, dataset root, environment, command line,
or candidate filesystem. This document contains no key, seed, DEK, nonce, or
other secret.

Signature domains are exactly:

```text
R12-ETTR-IL-v2/authority
R12-ETTR-IL-v2/plaintext-manifest
R12-ETTR-IL-v2/envelope-manifest
R12-ETTR-IL-v2/leakage-report
R12-ETTR-IL-v2/dataset-root
R12-ETTR-IL-v2/open-authorization
R12-ETTR-IL-v2/open-ledger
R12-ETTR-IL-v2/open-result
R12-ETTR-IL-v2/stage-launch
```

The authority record schema is `r12-ettr-il-v2-authority-v1` and has exactly:

```text
schema, protocol, authority_mode, protocol_spec_sha256,
root_public_key_hex, root_public_key_sha256,
delegations, issued_at_utc, root_signature
```

`delegations` is sorted by role and contains exactly
`{role,public_key_hex,public_key_sha256,principal_id,administrative_domain,
storage_domain,valid_from_utc,valid_until_utc}` for the five delegated roles.
Timestamps are RFC 3339 UTC with seconds and `Z`. `root_signature` is the
detached signature over the authority payload with that field omitted. The
root public-key SHA-256 must also be pinned outside the custody storage and
supplied directly to verifiers.

The final dataset root requires valid signatures from dataset publisher,
confirmation custodian, leakage auditor, and offline root. A delegated signer
cannot countersign for another role.

### 8.1 Same-account mechanics versus independent authority

`authority_mode` is either `same_account_rehearsal` or
`independent_authority`.

Modes, `O_EXCL`, `O_NOFOLLOW`, one-link checks, read-only bind mounts,
namespaces, descriptor closure, append-only local files, and Ed25519 signatures
can demonstrate same-account process mechanics. They cannot prevent the same
account owner, host administrator, or holder of all keys from reading a sealed
file, copying a DEK, replacing a ledger, or replaying a snapshot.

`independent_authority` additionally requires:

- confirmation custodian and opening authorizer have different
  `principal_id` and `administrative_domain` from the trainer/candidate;
- custodian seed, DEK, ciphertext master copy, and ledger storage are not
  mounted or credentialed in the trainer administrative domain;
- the WORM/conditional-write ledger is administered outside the trainer
  domain;
- offline root, custodian, opening authorizer, and leakage auditor are not one
  principal;
- each independent role signs from its own security boundary.

Different key bytes in one account do not establish independence. Any
`same_account_rehearsal` artifact may validate mechanics but must set
`claim_authorized=false`, and it cannot authorize a confirmation opening or
scientific claim.

## 9. Source inventory and detached commits

### 9.1 Frozen legacy commit

The legacy reference commit is
`854d3d4d63ee467fc6bed10ffee8fdd2d97a07d1`, with tree
`4dc8d24f989e96867d8f81a4a17a332299597912`. The following path labels and
literal SHA-256 values are exact:

| Path | SHA-256 |
|---|---|
| `train/endogenous_typed_theory_reactor.py` | `7b8a1f98267268240766775c558d9f3e98cd62c680a181993fd5be477ea9cd0a` |
| `train/ettr_episode.py` | `daf47408eb7db53c4a2e2e50d8490d4900ab7014b0dc9bb6721c9e73a058a7d3` |
| `train/ettr_objectives.py` | `94c4112bb6861fa7e4e89889c09bdb55730866ae0c4a73a39bbd5a1e02975bb8` |
| `train/ettr_qualification.py` | `00ad5f07f80bd14008fa68dc85c3646c49b44e4df95cf7b06131e71b47b25920` |
| `train/ettr_stage_supervisor.py` | `4a2511a37be5f24501e26d0aa976e1c2d9f92cbf02bff87c1c7bc65685b63207` |
| `pipeline/cross_ontology_schema.py` | `a6cbd3cffb6cac2a476e96c648ffef0d71dda45e53496a441e47dfa7bbcdced8` |
| `pipeline/cross_ontology_horn_board.py` | `24d5d31b2116cf58cbda41c4a25d0a880bf541ceab425acc4afb8687e88475d7` |
| `pipeline/cross_ontology_rewrite_board.py` | `2d6231aff0fc2479d62bc660241de265232f8c06089c47ce230540ced024a198` |
| `pipeline/cross_ontology_resource_board.py` | `b6ea173148ad0f35989160727438a93e7186a48c7822fd35c9fa887546b45d47` |
| `pipeline/cross_ontology_horn_variants.py` | `25707fe12ccb3e588da9b666e738e377179e749428aed48b74aa264a0bd42b2b` |
| `pipeline/cross_ontology_rewrite_variants.py` | `44aa07314d273a8b01e87d9b3143a4a3434aebb6a28531944e03f821da160383` |
| `pipeline/cross_ontology_resource_variants.py` | `10e1c50a6023cc6be35f37e68cb620c6596453ae17e86312be75bd7e6dc39b99` |
| `pipeline/cross_ontology_hybrid_compositions.py` | `f57c2624643da72d04ee79b0339c69adbe7ea8b6dd47ba852ce38356722a5644` |
| `pipeline/ettr_factorial_qualification_board.py` | `74189d36e3cf77c4af890dd133fb5b0c8453821882e71a2081528f05f497f286` |
| `pipeline/cross_ontology_qualification_matrix.py` | `767e15deb1da59fd030357962029482c7e228263314300417c9342d275a509d5` |
| `train/ettr_data_contract.py` | `dfa4615184f68a1eb4b14b7bb976d0d1f60e857043e358947f6a1fcbe0788065` |
| `train/ettr_state_io.py` | `d849dfa36e33f5fc048394fce9b2b3c22ffec61272505d33228cc3d32112b2c0` |
| `train/ettr_train_step.py` | `9cccedd66df797134f0ff4eb98315bf9b2c548f99bc8cfe37311601c192a581c` |
| `train/ettr_optimization.py` | `960825ca8713867e6752b9a681e57a6be2d3638d65d3f1e416b9d02a9107a1a1` |
| `train/ettr_checkpoint.py` | `44e2a0eca19f78378ae65a9aed965901633a0d55cdcaf6e5e8591ba804753f19` |
| `train/ettr_model_assembly.py` | `39c0c5eb68a245f2d9a79dfcb759acbf318dc59ff17f844d61dd5e4f45564ac6` |
| `train/ettr_factorial_qualification.py` | `38e15ef9d1441e80486adf0f002b7706dc53b15735c3b7413645535b2f98e95d` |
| `train/ettr_factorial_authority.py` | `b350a6a0dfa2fea93fde923a529417293305289f762eba9a3bd719c594d515c1` |
| `train/ettr_factorial_custody.py` | `f02a2e9876811b017f7c011d4c728e6e6e93acf81b245dcc8ca7dc6edde96569` |
| `train/ettr_factorial_signed_custody.py` | `aac38f5515eaa5f1ec08cfaa71a40e6d987ffac165f649a64c1adfe91a32d250` |
| `train/ettr_factorial_tokenization.py` | `d2e1707787683307590a5573eca0bfb9ffd288c6bf9a100726c844b9fe712c81` |
| `train/ettr_deployment_contract.py` | `5c2342ec0c8845a83b76ca8f36476393ae0fa25b877917b68e53bdaf0d9e515c` |
| `train/ettr_claim_runtime.py` | `0e3bbb3525491af53fe056cb71ed563be3e81f3521ed6af41775a36924175166` |
| `train/ettr_runtime_bundle.py` | `47a1378b980bdeb6398ce365138bd9aaf4a68aa610b8b90fa01038b4e7d7012e` |
| `train/run_ettr_verified_stage.py` | `c7adfea945efe9426985f42efc8ebc576a462d3aa2fdcef7360bde4190fc89e0` |

`Qualification source` therefore means
`train/ettr_qualification.py`, and `Episode runner` means
`train/ettr_episode.py`. The current branch or HEAD is never substituted.

### 9.2 v2 implementation inventory

Because no v2 materializer or custody implementation exists, no implementation
hash is invented here. Before any seed exists, a new full 40-hex
`implementation_commit` and tree must be published. Its source inventory
schema is `r12-ettr-il-v2-source-inventory-v1` with exactly:

```text
schema, protocol, protocol_spec_sha256,
legacy_commit, legacy_tree, implementation_commit, implementation_tree,
entries, runtime_entries, inventory_sha256
```

Each `entries` item is exactly
`{commit,path,git_mode,git_blob_oid,bytes,sha256,role}` and is sorted by
`(commit,path)`. It includes this specification, every executable source,
every imported first-party module, every schema/preimage file, every audit,
encrypt/open/verify source, and every test used for admission. `runtime_entries`
contains the executable, extension, shared-library, wheel/distribution, and
root-owned launcher identities, sorted by logical path.

`inventory_sha256` is `H(CJ1(entries))`. Missing and extra transitive
first-party imports fail. Dynamic imports, namespace-package fallback,
`PYTHONPATH` additions, editable installs, user site packages, network imports,
and source outside the inventory fail.

Construction and every open use separate clean worktrees in detached HEAD
state:

```text
git rev-parse HEAD == declared commit
git rev-parse HEAD^{tree} == declared tree
git symbolic-ref -q HEAD exits nonzero
git status --porcelain=v2 --untracked-files=all emits zero bytes
```

For each source, bytes read with `git cat-file blob COMMIT:PATH` must equal
worktree bytes and the inventory SHA-256. Symlinks, submodules, sparse
checkouts, replace refs, grafts, alternate object stores not inventoried, and
Git filters are forbidden. Source is copied into a hash-verified runtime bundle
before import. A later commit, including the v1 freeze-gate commit, is not
admitted unless a new protocol instance inventories and signs it.

## 10. Manifests and roots

All schemas reject extra fields.

### 10.1 File record and roots

A file record is exactly:

```text
path, bytes, sha256, row_count, media_type, confidentiality
```

`confidentiality` is `public`, `candidate`, `assessor`, or `ciphertext`.
Records sort by path. A file-set root is `H(CJ1(sorted_file_records))`.

The plaintext manifest schema
`r12-ettr-il-v2-plaintext-manifest-v1` has exactly:

```text
schema, protocol, fold, split, fold_commitment,
source_inventory_sha256, semantic_spec_sha256, split_spec_sha256,
tokenizer_identity_sha256, seed_commitment, files, file_set_root,
counts, cell_counts, fingerprint_roots, graph_root, created_at_utc
```

For train/development the public seed commitment is `H(split_key)`.
For confirmation it is the Section 4.3 commitment. Confirmation plaintext
hashes and counts are assessor-confidential until the opening result; they
must not be placed in candidate/public pre-opening manifests.

The envelope manifest schema
`r12-ettr-il-v2-envelope-manifest-v1` has exactly:

```text
schema, protocol, fold, split, encryption,
plaintext_manifest_sha256, ciphertext_files, envelope_root,
custodian_public_key_sha256, created_at_utc
```

The dataset-root schema `r12-ettr-il-v2-dataset-root-v1` has exactly:

```text
schema, protocol, authority_sha256, protocol_spec_sha256,
source_inventory_sha256, tokenizer_identity_sha256,
split_spec_sha256, fold_commitments,
train_manifest_sha256s, development_envelope_manifest_sha256s,
confirmation_envelope_manifest_sha256s,
leakage_report_sha256, opening_ledger_genesis_sha256,
required_signature_roles, claim_authorized
```

Lists are fold ordered. `claim_authorized` is true only in
`independent_authority` mode after all signatures verify.
`required_signature_roles` is exactly
`["confirmation_custodian","dataset_publisher","leakage_auditor",
"offline_root"]`, in that order. The four detached signature files are not
members of the signed payload and are sorted by role when presented with it.

## 11. Confirmation encryption

Development and confirmation assessor files are encrypted independently.
Confirmation rules below are mandatory; applying them to development does not
consume confirmation authority.

Each fold uses a fresh random 32-byte AES key `DEK[fold]`, generated by the
confirmation custodian with an operating-system CSPRNG. Cipher is AES-256-GCM
as implemented by
`cryptography.hazmat.primitives.ciphers.aead.AESGCM`; the exact
`cryptography` package and OpenSSL shared-library inventory is source/runtime
pinned. Keys are never derived from semantic seeds.

`key_id` is:

```text
H(ASCII("R12-ETTR-IL-v2") || 0x00 || ASCII("key-id") || 0x00 || DEK)
```

For each plaintext file, the custodian generates a fresh random 12-byte nonce.
Nonce reuse under one `key_id` is fatal across development, confirmation,
tests, rehearsals, and retries. A custodian key registry atomically rejects a
previous `(key_id, nonce_hex)`.

AAD is `CJ1` of an object with exactly:

```text
schema                    "r12-ettr-il-v2-aad-v1"
protocol                  "R12-ETTR-IL-v2"
fold                      0, 1, or 2
split                     "development" or "confirmation"
logical_path              plaintext relative path
file_ordinal              zero-based in sorted plaintext file records
plaintext_sha256          hex32
plaintext_bytes           nonnegative integer
row_count                 nonnegative integer
plaintext_manifest_sha256 hex32
source_inventory_sha256   hex32
split_spec_sha256         frozen v2 hash
tokenizer_identity_sha256 hex32
key_id                    hex32
```

`AESGCM.encrypt(nonce, plaintext, aad)` returns `ciphertext || 16-byte tag`.
Those literal bytes are stored in `<logical_path>.aes256gcm`. No compression,
base64, container header, or deterministic nonce is used.

An envelope record has exactly:

```text
schema "r12-ettr-il-v2-aes256gcm-envelope-v1",
logical_path, ciphertext_path, key_id, nonce_hex, aad_sha256,
plaintext_sha256, plaintext_bytes, ciphertext_sha256, ciphertext_bytes
```

`ciphertext_bytes == plaintext_bytes + 16`. Ciphertext hashes are committed
only after randomized encryption. Predetermining or comparing ciphertext
hashes across fresh encryptions is forbidden.

After encrypting, a second process reopens ciphertext by descriptor, verifies
AAD/tag/plaintext hash in custodian memory, and signs the envelope manifest.
All persistent plaintext copies, temporary files, swap-backed mappings, and
inherited descriptors are then destroyed or invalidated before dataset-root
signature. A plaintext path discovered afterward invalidates the dataset.
This deletion claim assumes the custodian OS/storage trust base; it is not a
cryptographic erasure claim.

## 12. Path and process isolation

Logical roots are disjoint:

```text
/public/spec
/public/manifests
/candidate/train
/sealed/development
/sealed/confirmation
/authority/ledger
/runs/endpoints
/runs/open
```

Candidate/trainer principals have no credential, ACL, mount, parent-directory
search permission, object-store token, or inherited descriptor for
`/sealed`, `/authority`, or assessor products. Custodian principals have no
write permission to endpoint roots. The opening assessor has read-only
endpoint descriptors and one custodian-mediated plaintext stream, never a
candidate/trainer write credential.

Every role is launched through a root-owned, hash-pinned default-deny launcher
using the established ETTR supervisor mechanics:

- descriptor-bound immutable single-link inputs opened with `O_NOFOLLOW` and
  `O_CLOEXEC`;
- exact per-role input/output maps and empty otherwise inaccessible roots;
- read-only runtime/source mounts, one fresh output directory, empty home and
  private tmpfs;
- network namespace with no interfaces;
- no inherited descriptor above stderr except declared launcher channels;
- exact environment allowlist, no user site or dynamic library override;
- child exit before outputs are hashed and the next role starts;
- Ed25519 launch receipt binds parent receipt, runtime, policy, inputs,
  outputs, exit, stdout, and stderr.

During fitting, the trusted differentiable runner may retain source tensors
for autograd, but its source-inventoried model-call interface remains
`WORLD-only compiler -> packet-plus-COMMAND reactor ->
terminal-packet-plus-QUERY reader`; no raw upstream segment, base residual, or
KV state crosses those interfaces. This is logical non-consumption, not a
physical-erasure claim.

WORLD, COMMAND, and QUERY are physically separated under the signed supervisor
for every autonomous development, confirmation, and frozen-board evaluation.
Each stage source package and residual/KV state is closed and removed before
the next stage starts. Assessor labels and confirmation AAD/envelopes are
never mounted into a candidate stage. Path modes alone do not upgrade
same-account mechanics to independent authority.

## 13. Consume-on-attempt confirmation opening

### 13.1 Preconditions

An opening authorization is invalid unless it binds:

- final root-signed dataset root;
- all fold/arm/seed update-6000 endpoint hashes and immutable file root;
- frozen evaluator, decoder, scorer, runtime, and launch-policy roots;
- signed development opening result showing every required development and
  custody gate passed;
- zero prior confirmation reservation;
- exact confirmation envelope roots;
- a single opening panel containing all folds, arms, seeds, and the three
  pre-existing score-only boards;
- `rescore_allowed=false`, `retry_allowed=false`,
  `checkpoint_selection_allowed=false`.

The authorization schema is `r12-ettr-il-v2-open-authorization-v1` and has
exactly:

```text
schema, protocol, dataset_root_sha256, endpoint_root_sha256,
evaluator_root_sha256, development_result_sha256,
confirmation_envelope_roots, panel_sha256,
rescore_allowed, retry_allowed, checkpoint_selection_allowed,
expires_at_utc, authorization_nonce, authorizer_signature
```

`authorization_nonce` is 32 fresh random bytes encoded hex. The opening
authorizer signs it under `R12-ETTR-IL-v2/open-authorization`.

### 13.2 Ledger

The independent custodian ledger is append-only WORM storage with atomic
conditional create/CAS. Its event schema is
`r12-ettr-il-v2-open-ledger-event-v1`:

```text
schema, protocol, dataset_root_sha256, sequence, state,
previous_event_sha256, authorization_sha256, attempt_id,
event_at_utc, evidence_sha256, custodian_signature
```

Allowed chain:

```text
SEALED(sequence=0)
  -> RESERVED(sequence=1)
  -> COMPLETED_CONSUMED(sequence=2)
   | FAILED_CONSUMED(sequence=2)
```

Genesis has null authorization/attempt and binds the envelope roots in
`evidence_sha256`. `RESERVED` has a fresh random hex32 `attempt_id` and the
authorization hash. The conditional write succeeds only if sequence zero is
the current tip and no reservation exists. The reservation is durably
witnessed before a DEK is made available or any decryption begins.

There is no transition from either consumed state and no transition from
`RESERVED` back to `SEALED`. A crash, timeout, tag failure, evaluator failure,
host failure, or partial output after reservation yields
`FAILED_CONSUMED`; absence of its terminal event is treated identically.
Thus open-once means at most one authorized attempt, not one successful score.
No retry is authorized under this dataset root.

Hash chaining plus local files is insufficient against rollback by the same
administrator. Claim-bearing open-once requires the independent WORM/CAS
authority described in Section 8.1.

### 13.3 One opening process

After durable reservation, one broker process:

1. verifies source/runtime/authority/dataset/authorization/ledger roots;
2. receives each DEK through a sealed anonymous descriptor from the custodian;
3. decrypts ciphertext in private non-swap memory and verifies tag, AAD,
   plaintext hash, counts, and canonical bytes;
4. projects only candidate WORLD/COMMAND/QUERY bytes into fresh candidate
   child descriptors;
5. waits for candidate exit and seals complete predictions before exposing
   assessor targets;
6. scores with the frozen assessor and independent replay;
7. writes one aggregate result and an encrypted complete prediction transcript;
8. zeroizes broker buffers and closes all descriptors;
9. appends `COMPLETED_CONSUMED`, or `FAILED_CONSUMED` on any exception.

Plaintext confirmation, labels, per-row oracle state, DEKs, and seeds are never
persisted in the run root. The open result schema
`r12-ettr-il-v2-open-result-v1` binds attempt, authorization, ledger
reservation, envelope/plaintext roots, endpoint/evaluator roots, prediction
root, metric root, launch-receipt chain, terminal ledger event, and
`confirmation_access_count=1`. It is signed by custodian and leakage auditor.

## 14. Fail-closed order

The implementation must execute these gates in order and stop on the first
failure:

1. verify this document hash, literal preimages, schemas, and master/fold
   commitments;
2. verify independent authority record and externally pinned root key;
3. enter clean detached legacy and implementation commits; verify complete
   source/runtime inventory before import;
4. verify tokenizer bytes, runtime closure, and conformance vectors;
5. verify that all unresolved semantic/materializer blockers have separate
   signed repairs; otherwise stop before seed generation;
6. create and commit split-isolated seed roots; keep confirmation seeds solely
   inside the custodian;
7. enumerate and select tuples with exact quotas and no fallback;
8. run field classification, taint, forbidden identifier, and constant-token
   audits;
9. run raw, semantic-world, theory, bound-command, opaque-name, token-sequence,
   and normalized-13-gram overlap audits;
10. run exact graph-isomorphism audit and independent witness verification;
11. run the pinned metadata classifier and confidence rule;
12. construct plaintext manifests and independently replay all counts/hashes;
13. encrypt development/confirmation with randomized DEKs/nonces and exact
    AAD; verify decrypt/tag/hash in a second process;
14. prove persistent plaintext absence and seal envelope manifests;
15. collect publisher, custodian, auditor, and offline-root signatures; publish
    dataset root and ledger genesis;
16. authorize candidate training mounts only if `claim_authorized=true`;
17. freeze every endpoint, evaluator, decoder, scorer, and launch identity;
18. open development under its separate ledger; freeze its signed result;
19. if development passes, issue one confirmation authorization;
20. atomically reserve confirmation before DEK release, run one panel, and
    append a consumed terminal event;
21. verify final receipt chain before any scientific statement.

Failures through step 15 produce no authorized dataset. A failure after any
candidate mount invalidates all runs under that dataset root. A failure after
confirmation reservation consumes confirmation permanently. No deletion,
permission repair, manifest rewrite, rescore, or new signature repairs an
already violated instance.

## 15. Required artifacts and blocker closure

A custody instance is complete only when all of these immutable objects exist:

```text
authority.json
authority.signature.json
source_inventory.json
tokenizer_identity.json
split_spec.json
fold_0.json
fold_1.json
fold_2.json
train_plaintext_manifests[3]
development_envelope_manifests[3]
confirmation_envelope_manifests[3]
leakage_report.json
opening_ledger_genesis.json
dataset_root.json
dataset_root_signatures[4]
```

Later opening adds authorization, reservation, result, and terminal ledger
events without modifying any prior byte.

Closure mapping:

| Audit blocker | v2 closure |
|---:|---|
| 1 | Literal split/fold preimages, tuple schema, enumeration/ranking/tie rules, seed domains, and fold-hash derivation are exact. |
| 6 | Stable ontology codes are forbidden; field taint, neutral syntax, constant-token audit, and metadata classification cover candidate-visible bytes and mechanics. |
| 13 | AES-256-GCM, keys, random nonces, exact AAD/envelopes, post-encryption hashes, root/delegated signatures, and consume-on-attempt ledger are specified. |
| 15 | Every fingerprint, normalization scope, tokenizer, exact graph canonicalization, classifier, folds, solver, seed, metric, threshold, and confidence rule is fixed. |
| 16 | Previously ambiguous labels map to exact paths/hashes at a full commit/tree; future v2 sources require a complete signed inventory and clean detached commits. |

This closure records Phase-1 custody mechanics only. Until Phase 2 freezes and
audits a literal population and the user authorizes fitting, the only valid
decision is:

`r12_ettr_il_v2_phase1_custody_ready_phase2_not_authorized_no_capability_claim`
