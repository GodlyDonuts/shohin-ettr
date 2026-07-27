# R12 ER-TT Structured Opcode-Complement Route Preregistration

**Status:** pre-source-refreeze; train-only factorial qualification; no
development or confirmation access is permitted.

## 1. Question

Ordinal-route v1.2 represents a rule as an ordered list containing `2N`
witnesses and one opcode. Its fresh-board rejection admits three distinct
explanations that the old evidence cannot separate:

1. the learned exclusion-path posterior is correct but independent per-slot
   marginal argmax does not form one coherent path;
2. the path posterior needs the excluded candidate to score as the opcode; or
3. another equal-budget pass of the old marginal objective is sufficient;
4. learned opcode coupling is required; or
5. even opcode coupling leaves a diffuse posterior and requires a structured
   exclusion-path objective.

The same rejection also showed that query-pointer offsets changed under
neutral-name recoding while query semantics did not. A separate parameter-free
query mode therefore canonicalizes only six-byte neutral names before query
pointer encoding. It is not bundled into the route diagnosis.

## 2. Structured route

For cardinality `N`, candidates are `c_0 ... c_2N`. Excluding rank `e` defines
the only legal ordered witness path

```text
pi_e(j) = j + 1[j >= e],  j in 0 ... 2N-1.
```

The path score is

```text
S_e = sum_j witness_logit[j, c_pi_e(j)]
      + lambda * opcode_logit[c_e].
```

The independent decoder argmaxes every witness-slot marginal. It succeeds only
when those `2N` hard pointers are unique and leave exactly one complementary
candidate. The coherent decoder instead selects one `argmax_e S_e`. Both then
use the same complement-as-opcode rule, exact six-byte relation construction,
and event rebinding. Thus the decoder contrast changes path hardening only. It
never consults a target, outcome, executor result, retry, renderer label, or
host repair.

The structured arm replaces, rather than augments, the old per-slot witness
loss with

```text
L_route = cross_entropy(S, unique_gold_exclusion).
```

Gold exclusion is source-only compiler supervision: it is the unique local
candidate occurrence not covered by the `2N` witness spans. Cardinality and
record-role losses remain separate and unchanged.

## 3. Equal-resource arms

All fitted arms reconstruct byte-identical qualified v1.2 trainable state,
use the same 10,000 fit families, family order, optimizer, batches, two epochs,
and 2,500 updates.

| Arm | Fit opcode weight | Witness objective |
|---|---:|---|
| `zero_update` | 0 | no optimization; immutable diagnostic reference |
| `legacy_uncoupled` | 0 | original per-slot marginal pointer loss |
| `opcode_coupled` | 1 | original per-slot marginal pointer loss |
| `structured_route` | 1 | coherent exclusion-path NLL |

No arm adds parameters. The exact complete/trainable/headroom certificate
remains `185,532,296 / 11,129,504 / 14,467,704` under the user-authorized strict
below-200M ceiling. Learned motor and reader parameters remain zero.

## 4. Controlled train-only probe

Two thousand family-disjoint semantics from the already-consumed fresh
training split remain withheld from all fitting. Each is scored in its four
original train renderers and four controlled twins crossing only:

```text
witness style: opcode-first / opcode-middle
query style:   prefix / suffix
```

Declaration and event style are fixed at zero. Storage order, rule
distractors, event distractor, event-noise slot, and query distractor are held
identical within each four-way family twin. Thus witness position and query
position are not confounded with event, declaration, noise, or storage changes.
Rows are rendered from train semantics only, reparsed, and independently
executed before scoring.

Every checkpoint is evaluated under the crossed inference modes:

- score coupling `S0` (`lambda=0`) and `S1` (`lambda=1`);
- independent slot-marginal hardening and coherent path-MAP hardening with
  identical downstream decoding; and
- raw query routing and structural-neutral query routing.

Two independent query-only alpha recodings leave every program byte fixed.
Deterministic within-record rotations of opcode logits and witness logits are
separate causal negative controls.

## 5. Frozen diagnosis and advancement

The assessor must identify exactly one of the following before advancement:

1. **Decoder artifact:** zero-update `S0` coherent route joint is at least 99%,
   zero-update `S0` independent route joint is at most 80%, and the gain is at
   least 20 points.
2. **Acute opcode repair:** zero-update `S1` coherent route joint is at least
   99%, zero-update `S0` coherent route joint is at most 80%, and the gain is
   at least 20 points.
3. **Additional marginal training:** legacy `S0` coherent route joint is at
   least 99%, zero-update `S0` coherent route joint is at most 80%, and the
   gain is at least 20 points.
4. **Learned opcode repair:** coupled `S1` coherent route joint is at least 99%, both
   legacy `S1` and coupled `S0` are at most 80%, and the gain is at least 20
   points.
5. **Structured-learning repair:** structured `S1` coherent route joint is at least
   99%, opcode-coupled `S1` is at most 80%, and the gain is at least 20 points.

All route diagnoses use raw query routing and a route-joint metric that excludes
query/answer fields. Query grounding is diagnosed afterward. Raw query routing
must already clear 99% semantic and pointer gates, or structural routing must
clear them while beating raw routing by at least 20 points with raw at most
80%. Query canonicalization therefore cannot rescue or label a route repair.

For the selected mechanism, all of these conjunctive gates must also pass:

- canonical and controlled-twin relation, complete witness pointer, rule-opcode
  pointer, event-opcode pointer, state, and route joint are each at least 99%;
- every cardinality, renderer, and renderer-by-cardinality intersection has at
  least 99% route joint and pointer exactness;
- all semantic predictions are identical across all eight views for every one
  of 2,000 families;
- program alpha, distractor rotation, and both query-only recodings are exact
  on all 8,000 controlled rows;
- query semantics and query pointer spans pass their separately selected mode;
- source-free route joint is at most 10%;
- rotating opcode logits drops an opcode-dependent selected mechanism by at
  least 20 points and to at most 80%;
- rotating witness logits always drops the selected route by at least 20
  points and to at most 80%;
- all fitted arms execute exactly 2,500 updates from one initialization and
  preserve byte-identical excluded parent state;
- the exact parameter certificate remains strictly below 200M; and
- development and confirmation reads remain zero.

Scores from 80% to 99%, multiple simultaneously passing causal diagnoses, or
failure to identify one mechanism are inconclusive/rejecting outcomes. Gates
will not be relaxed after the run.

## 6. Immutable evidence

For every decisive branch, retain raw per-row coherent and independent
predictions. For every controlled row and semantic rule, also retain candidate
source positions, all conditional path scores and probabilities for `N=3..6`,
selected physical record, target and MAP exclusion rank, correct-path
rank/probability, coherent pointers, relation, opcode pointers, and query
pointer. Retain the model-owned cardinality posterior, candidate-local witness
and opcode logits, and witness marginals so the assessor can independently
recompute path scores, path-to-slot marginalization, and both hard decoders.
Retain raw alpha, distractor, source-free, opcode-rotation,
witness-rotation, and both query-recode predictions plus all fit histories,
initialization hashes, frozen-parent digests, source/data manifests, thresholds,
and crossed-cell summaries. Floating path evidence may be stored as float16
after metrics are computed in float32.

The independent assessor imports no experiment-producer code. It separately
rebuilds the train/probe split and controlled rows from the immutable board and
seed, recomputes every decisive metric from raw predictions, implements its own
gate logic, checks the ordered MAP path and probabilities, derives witness
marginals from path and cardinality probabilities, reconstructs both hard
decoders, verifies each logit rotation, and hashes every checkpoint arm against
its bound fit receipt. Any mismatch exits nonzero and forces
`reject_assessment`; a producer authorization cannot survive a failed
assessment.

## 7. Claim boundary

A pass would establish a renderer-position-stable, source-grounded compiler
route on held-out consumed-training semantics and identify whether the repair
comes from coherent hardening, opcode evidence, or structured learning. It
would authorize only a separately frozen fresh-board source and seed. It would
not establish unseen semantic transfer, a learned executor, autonomous halt,
planning, arithmetic, open-language reasoning, public-benchmark gain, or
general reasoning. The closed v1 development board remains permanently closed
and its confirmation remains unopened.
