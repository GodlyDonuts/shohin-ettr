# R12 EFC Source-Sealed Quotient-Algebra Compiler Protocol

## Status

**REJECTED FOR SOURCE FREEZE / CPU FALSIFIER ONLY.**

No neural fit, scored development access, confirmation generation, reasoning
claim, or pretraining is authorized by this document. The protected step-300k
Shohin checkpoint remains immutable. The user pretraining hold remains active.

The candidate is named **Source-Sealed Quotient-Algebra Compiler (SSQAC)**.
Its first fixed-solver draft was rejected because a host-authored
Macaulay/border-basis circuit would own the substantive reasoning. The revised
primary treatment must compile both constraints and a source-deleted algebra
microprogram executed by only primitive field/register operations. The fixed
exact algebra engine remains a gold ceiling and favorable control. JASEC and
hardcoded SLRA remain additional controls.

## Question

Can a compact model compile raw anonymous source into a finite algebra of
episode-local constraints, preserve correlations among multiple legal machine
completions, and prove a query or machine cell only when it is invariant across
that algebra?

The intended distinction is:

- JASEC predicts one binding and one machine, then iteratively makes them
  self-consistent;
- hardcoded SLRA receives a fixed permutation/balance residual whose unique
  zero already identifies the answer;
- SSQAC predicts source constraints, then a learned source-deleted controller
  constructs and checks their joint solution set through a counted algebra
  microprogram, sealing only consequences that are algebraically forced.

This would still be bounded reasoning. It does not establish unrestricted
general reasoning unless the same mechanism later transfers across natural
task families and direct interaction.

## Prior-Art And Novelty Boundary

Neural prediction of Groebner bases, neural guidance of exact border-basis
algorithms, differentiable theorem proving, and probabilistic algebraic
constraint layers already exist. Relevant primary references include:

- [Learning to Compute Groebner Bases](https://arxiv.org/abs/2311.12904);
- [Computational Algebra with Attention](https://arxiv.org/abs/2505.23696);
- [Learning Reasoning Strategies in End-to-End Differentiable Proving](https://proceedings.mlr.press/v119/minervini20a.html);
  and
- [A Probabilistic Neuro-symbolic Layer for Algebraic Constraint Satisfaction](https://proceedings.mlr.press/v286/kurscheidt25a.html).

Therefore SSQAC may not claim novelty merely for combining a neural emitter
with polynomial constraints, Groebner/border-basis computation, or a symbolic
executor. The only potentially distinct empirical claim is the complete
contract: raw anonymous law text is compiled into exact constraints; the
source and compiler are destroyed; a learned controller constructs a
certificate using only primitive field/register operations; later
consequences are sealed only when forced across all legal completions; and the
same learned mechanism transfers across paired opposite laws, unseen law
topologies, variable ontologies, renderers, and task families under matched
controls. Novelty remains unclaimed until that full contract passes and a
dedicated literature review finds no equivalent system.

## Claim Boundary

A positive SSQAC result may support:

> bounded, architecture-native compilation and composition of unseen
> episode-local laws after irreversible source deletion.

It may not support:

- unrestricted general reasoning;
- a claim that the 125M Shohin trunk alone acquired reasoning;
- reasoning attribution to parameter count;
- natural-language transfer without a separate held-out task-family result; or
- source deletion if the compiler process, source mount, GPU state, or
  temporary workspace remains available to the query process.

## Mathematical Object

Let `z = (z_1, ..., z_d)` contain Boolean physical-key assignment atoms and
categorical transition/observer atoms. A source-attached neural emitter maps
each raw record to a sparse bounded-degree polynomial over `F_257`:

`f_r(z) = sum_alpha c[r, alpha] z^alpha`.

The source-defined ideal is:

`I_S = < z_i^2-z_i, one_hot(z), f_1, ..., f_m >`.

The retained semantic object is the finite coordinate algebra:

`A_S = F_257[z] / I_S`.

The first implementation is bounded:

- at most 128 source generators;
- at most 4,096 admitted monomials;
- maximum **emitted-generator** degree four;
- quotient dimension at most 256;
- exact arithmetic modulo 257;
- no floating rank decision;
- overflow, incomplete closure, ambiguity, or inconsistent constraints fail
  closed with distinct statuses.

The degree-four bound does not cap the degree needed during closure. Admitted
square-free monomials may have higher degree. The runtime must either prove
that its admitted monomial set is closed under every quotient generator
multiplication needed by the board or abstain. A degree-truncated Macaulay span
is not the quotient algebra and may not support a certificate.

Because every variable is Boolean and `257 != 2`, the base quotient

`F_257[z] / <z_i^2-z_i>`

is a finite product of fields indexed by Boolean assignments. Every further
ideal in that product is radical. The implementation may therefore use
rootwise constancy as an equivalent assessment oracle, but the candidate
runtime must establish the same equality by a complete exact algebra
certificate. One-hot constraints are additional generators inside this
Boolean quotient; they do not relax the closure requirement.

Every board law must pass exhaustive Boolean truth-table equivalence between
its intended semantics and its `F_257` polynomial encoding. A second-prime
audit over `F_263` must preserve the Boolean zero set. A law whose coefficient
reduction changes its intended semantics, including a coefficient that
vanishes modulo the field characteristic, is rejected before any split is
created.

A fixed exact sparse Macaulay/border-basis engine computes the CPU gold ceiling
and favorable fixed-solver control:

- closure under admitted monomial multiplication;
- exact row reduction over `F_257`;
- a quotient basis;
- multiplication operators;
- normal forms for machine and query polynomials; and
- optional ideal-membership certificates.

This engine is not the primary reasoning treatment, even when implemented in
Torch/CUDA. Reimplementing a solver as tensor operations does not make its
algorithm model-owned.

The claim-bearing runtime exposes only a fixed-shape register file and these
primitive operations:

1. copy or clear a row/register;
2. add a selected row multiplied by one selected `F_257` scalar;
3. swap two selected rows;
4. request one public square-free monomial prolongation;
5. multiply one retained normal form by one selected variable;
6. compare one register to zero or another register;
7. append one bounded certificate record; and
8. `HALT_SUCCESS`, `HALT_AMBIGUOUS`, `HALT_UNSAT`,
   `HALT_INCOMPLETE`, or `HALT_OVERFLOW`.

A tied learned controller selects the opcode, row/register addresses,
coefficient, and halt decision at every cycle from only the current sealed
algebra registers, public masks, and its private recurrent state. Host code may
validate types and bounds but may not select pivots, prolongations, row
operations, basis order, or stopping time. The controller must generate the
closure/reduction/certificate schedule after the source compiler has exited.
Its maximum cycle count and every field operation are included in the resource
receipt.

The claim-bearing implementation may not import or invoke SymPy, Sage, a
Groebner/border-basis package, a SAT/SMT solver, a Python semantic search loop,
the board generator, the label oracle, the gold engine, or assessor feedback.

For every successful seal, the receipt must contain enough source-free
information for an independent exact checker to verify:

- the admitted monomial order and multiplication closure;
- every pivot and modular inverse used in row reduction;
- reduction of every source generator to zero;
- reduction of each Boolean/one-hot schema generator to zero;
- the quotient dimension and multiplication-table consistency; and
- the claimed answer or cell equality in the quotient.

Failure to emit or independently verify this certificate is abstention, not an
incorrect answer and not a successful reasoning case.

Completeness additionally requires an order ideal connected to `1`, unique
rewrites for every border monomial, pairwise commuting multiplication
matrices, `M_i^2=M_i` for every Boolean variable, all generators reducing to
zero, `1` not reducing to zero, and stable rank under one complete
prolongation beyond the retained border. Any failed condition is
`INCOMPLETE` or `UNSAT`; it may never be converted to a certified answer.

## Behavioral Uniqueness

For answer polynomial `g[q,y](z)`, answer `y` is certifiable only when:

`NF_I(g[q,y]) = 1`

and every other answer normal form is zero.

A machine cell may be sealed only when its category is constant over every root
of `I_S`. If several machines remain but every legal late query has the same
answer, that behavior may be sealed with an explicit quotient receipt. If two
roots are distinguishable by a legal late query, the compiler must abstain.

The runtime statuses are disjoint:

- `AMBIGUOUS`: complete certified algebra, multiple query-distinguishable roots;
- `UNSAT`: complete certified algebra with `1` in the ideal;
- `INCOMPLETE`: closure or certificate could not be established;
- `RESOURCE_OVERFLOW`: a registered dimension, monomial, row, or cycle bound
  was exceeded; and
- `CERTIFIED`: the claimed behavior is constant in the complete quotient.

Only the first status is epistemic abstention. `INCOMPLETE` and
`RESOURCE_OVERFLOW` are coverage failures and can never satisfy an ambiguity
gate.

This is an information boundary, not an optimization preference. No
deterministic source-deleted system can recover information absent from the
source-defined version space.

## Law-Collision Board

The decisive board must prevent finite-template and marginal shortcuts.

Every paired episode contains:

1. a partial anonymous machine whose direct records admit at least two
   completions;
2. an episode-local law that selects exactly one completion;
3. an opposite-law twin with byte-identical non-law evidence and a different
   selected completion; and
4. fresh late queries generated only after the source-side artifact is sealed.

Every primary unit is a balanced counterfactual quadruple, not only a law
pair:

1. facts `F0` with law `L0`;
2. the same facts `F0` with opposite law `L1`;
3. counterfactual facts `F1` with law `L0`; and
4. the same counterfactual facts `F1` with law `L1`.

All four target combinations must be equally represented. This prevents a law
template alone or a fact template alone from identifying the completion.
Candidate completion indices, enumeration order, and canonical labels are
never model features or training labels.

The deterministic four-state `F0/F1 x L0/L1` fixture is only the smallest CPU
mechanics falsifier. It is explicitly **ineligible for neural training,
development, confirmation, or promotion**: its four targets admit a shallow
fact-bit XOR law-bit rule, `F0` and `F1` are state-isomorphic, and its fixed
late-query address breaks that symmetry. Passing it proves only exact
enumeration, law deletion, clause deletion, and counterfactual plumbing. The
claim-bearing board must contain many independently generated, nonisomorphic
quadruples with varying queries and must separately prove that fact-only,
law-only, rank/marginal, positional, and learned direct classifiers remain at
chance.

Primary collision pairs must match:

- state/action/observer/answer cardinalities;
- all unary and row marginals;
- key frequencies and occurrence counts;
- direct visible records;
- source length and record count;
- depth-zero, depth-one, and depth-two behavior multisets;
- renderer statistics; and
- candidate-completion cardinality.

They may differ only through a determining degree-three or degree-four
relation. The law must therefore require joint composition rather than reading
one marginal or one candidate-local zero.

### Variable Geometry

The full qualification board varies:

- 5--13 states;
- 2--5 actions;
- 1--4 observers;
- 2--6 opaque answers;
- one to four hidden rows;
- transition, observer, coupled transition-observer, and independent
  trajectory-identification task families; and
- DSL, equations, controlled English, and an independently implemented
  confirmation renderer.

Padding and maximum geometry are fixed before training. Geometry values, family
IDs, renderer IDs, split IDs, source hashes, and canonical program IDs are not
model inputs.

### Split Rules

Before rendering, reject overlap in:

- canonical machine isomorphism;
- canonical Boolean zero set and law ideal under variable permutation,
  Boolean complement, and machine isomorphism;
- law circuit topology;
- generator lineage;
- task family;
- renderer implementation;
- semantic program;
- query behavior signature; and
- paired target fiber.

Training may use simple degree-two/three laws and two renderers. Development
holds out deeper compositions, one renderer, ontology sizes, and machine
families. Confirmation is generated once after source, executable, weights,
optimizer decision, and assessor are frozen; it holds out complete law
families, one task family, a separately implemented renderer, and depths
4--12.

The confirmation family definition, generator executable, count bounds, and
seed derivation must be committed before the weights are frozen. The seed
itself is derived from a later public beacon. This prevents selecting a
confirmation distribution after inspecting the trained model while preserving
post-freeze unpredictability.

## Zero-Parameter Gates

Before any neural code or board labels enter a candidate process:

1. Exhaustively enumerate the bounded version space for every CPU fixture.
2. Without the law, prove that at least two behaviorally distinguishable
   completions remain.
3. With the law, prove that exactly one completion or one query-equivalence
   class remains.
4. Law twins must share exact non-law bytes and select opposite completions.
5. Law deletion must make the gold compiler abstain.
6. Removing a determining clause must make the gold compiler abstain.
7. Adding a redundant generator already in the ideal must change nothing.
8. Opaque-key recoding must conjugate the quotient representation and preserve
   every answer.
9. A direct rank/marginal classifier must remain at chance on collision pairs.
10. The gold quotient-algebra ceiling must solve 100% or SSQAC is killed before
    a neural fit.
11. Every registered gold case must remain inside all monomial, quotient,
    workspace, and cycle bounds; overflow is a failed coverage case.
12. Intended-law truth tables and zero sets must agree over both `F_257` and
    `F_263`.

The first mandatory falsifier uses three Boolean variables:

`I_facts = <x^2-x, y^2-y, z^2-z, x-y, y-z>`.

It has roots `{000,111}`, so query `x` is `AMBIGUOUS` while `x-y` is
certifiably zero. Adding `L0=<xyz>` must leave only `000`; adding
`L1=<(1-x)(1-y)(1-z)>` must leave only `111`. Law deletion restores
ambiguity; adding generator `1` returns `UNSAT`; omitting `xyz` from the
admitted closure returns `INCOMPLETE`; and a semantic law containing a
coefficient that vanishes modulo 257 is rejected by the field-semantics
validator. Variable permutations preserve every decision and simultaneous
Boolean complement swaps `L0` with `L1`.

The reference generator and exhaustive oracle are preparation/assessment
tools. They must be absent from the candidate runtime closure.

## Neural Architecture

The provisional complete system replaces JASEC's 19,013,524-parameter
assignment-machine equilibrium rather than adding to it:

| Component | Parameters |
|---|---:|
| Protected Shohin trunk | 125,081,664 |
| Gauge-invariant source/witness encoder | 44,658,064 |
| Detached query parser | 748,033 |
| Clause-to-polynomial emitter | 3,400,000 |
| Typed monomial/gauge router | 1,600,000 |
| Tied source-deleted algebra microcontroller | 12,000,000 |
| Certificate/query/intervention heads | 2,000,000 |
| **Complete provisional system** | **189,487,761** |
| **Headroom below 200M** | **10,512,239** |

These counts are a design budget, not an admission. A live constructor must
implement masked variable geometry, opaque-answer binding, the sealed quotient
wire, and the late-query parser; recompute unique parameters; and remain
strictly below 200,000,000. The current fixed `8/3/2/4` JASEC runtime and
detached hard-machine worker do not instantiate this system. No parameter
receipt is valid until the new constructor and artifact format exist.

The complete resource vector must report more than trainable parameters:
fixed-runtime source bytes, monomial-table bytes, field-inverse bytes, peak
workspace, sealed-artifact bytes, primitive field operations, controller
cycles, sequential depth, wall time, and peak device memory. The fixed-solver
control receives the same representation bounds and its full algorithmic
resource vector is reported rather than treated as free.

The score-bearing emitter receives only raw source bytes or their ordinary
tokenization, position/mask tensors, and the frozen Shohin residuals computed
from those bytes. It must infer record boundaries, occurrence identity,
coreference, and law roles itself. Host-provided physical-key incidence,
equality partitions, record-role tensors, candidate completions, or parsed
schema fields are oracle diagnostics and are ineligible for treatment. The
emitter must not receive:

- parsed law ASTs;
- host-provided key incidence or equality matrices;
- record, clause, witness, or law-role labels at inference;
- SLRA residuals;
- target machine probabilities;
- supervisor tensors;
- target categories;
- candidate scores from an oracle;
- query answers;
- split/family/renderer IDs; or
- source/canonical hashes as features.

The algebra circuit receives emitted coefficients and fixed public Boolean /
one-hot schema generators. It may not receive target labels or reference
solutions.

### Discrete coefficient learning

Exact modular row reduction is intentionally outside the gradient path.
Training may supervise a bounded sparse sequence of monomial indices and
`F_257` coefficients derived solely from the training source law, with all
machine completions, query answers, recurrent states, and executor outcomes
withheld. At inference, logits are hardened once into exact coefficients
before the algebra circuit runs. Straight-through gradients, soft modular
arithmetic, or gradients through an oracle completion are forbidden in the
primary arm.

Each emitted term uses categorical heads for generator boundary, monomial
index, coefficient in `{0, ..., 256}`, and sequence `STOP`. Duplicate monomials
inside one generator are merged modulo 257 after hardening. Missing `STOP`,
support overflow, invalid schema type, an all-zero required generator, or a
top-two logit margin below the frozen hardening threshold produces
`INCOMPLETE`. Generator ordering is canonicalized only by source occurrence
order and public type; target/candidate order may not enter canonicalization.

Development and confirmation report three separate levels:

1. soft term/support accuracy before hardening;
2. exact hardened generator multiset accuracy; and
3. independently certified behavior after controller execution.

Only the third level is a reasoning score, and it is credited only when the
complete certificate verifies.

This makes the first SSQAC candidate a neural source-to-constraint compiler
followed by a learned source-deleted algebra controller and a primitive field
ALU. It is not an end-to-end differentiable discovery claim. A favorable
equal-parameter sequence compiler that emits the final machine directly must
receive the same source-side supervision budget.

### Late-query semantics

The first claim-bearing query language contains only public predicates over
the sealed ontology: value/category of one anonymous cell, equality of two
cells, or one public bounded composition of those predicates. The counted
late-query parser may emit only predicate kind and anonymous codebook
addresses. It may not emit answer coefficients, a target category, a normal
form, or a certificate.

After hard pointer/type selection, a source-independent public constructor
maps the typed query to Boolean answer-indicator polynomials. Those indicators
must each be Boolean in the quotient, pairwise exclusive, and sum to one. The
same constructor and costs are shared by treatment and controls. Gold query
pointers are an oracle diagnostic only.

## Source Deletion

Claim-bearing evaluation uses three processes:

1. the compiler process receives source bytes, emits the hardened exact
   constraint tensor and anonymous symbol codebook, then exits;
2. the learned algebra-controller process receives only that tensor, codebook,
   fixed controller weights, and primitive ALU runtime; it emits a complete
   quotient/certificate artifact, destroys its constraint/workspace/private
   state, and exits; and
3. only then is the late query generated. The detached query process receives
   the quotient/certificate artifact, anonymous codebook, query parser, and
   query bytes.

Each successor starts only after the predecessor's cgroup, GPU context, shared
memory, mount namespace, temporary files, and writable artifact handles are
destroyed. The claim-bearing runtime is independently packaged; it may not add
the repository to `sys.path` or import from a writable checkout. A closed
file-descriptor inventory, mount/pivot-root receipt, process-death receipt, and
network-denial receipt are mandatory.

The final detached process receives only:

- the fixed quotient/certificate artifact;
- a bounded model-emitted anonymous symbol codebook included in that artifact;
- the counted late-query parser/runtime; and
- late query bytes.

It must not receive source tokens, KV cache, residuals, record states,
constraint-emitter state, hardened source generators, algebra-controller
state, Macaulay workspace, labels, or compiler diagnostics.

Literal source keys may not be copied across the boundary as a host dictionary.
The late-query parser must match query bytes to the sealed anonymous codebook
through its counted neural path. A gold string/equality matcher is an oracle
ceiling only. Codebook deletion, permutation, collision, and donor
transplantation are required causal controls.

## Matched Controls

All neural controls must match train examples, optimizer, updates, parameters,
batching, and candidate runtime where applicable.

1. **Gold full ceiling:** exact generators plus fixed exact algebra engine.
   This is an oracle ceiling, never a reasoning result.
2. **Gold constraints plus learned controller:** isolates source-deleted
   execution planning.
3. **Emitted constraints plus fixed exact engine:** favorable semantic-compiler
   control. It may establish source-to-law compilation but not model-owned
   algebraic execution.
4. **Current JASEC:** factorized point-estimate equilibrium.
5. **Equal-parameter unstructured compiler:** predicts machine logits directly
   from the same source states.
6. **Generator derangement:** precommitted source-independent permutation of
   generator-to-record alignment.
7. **Rank-one/factorized algebra:** removes generator interactions while
   preserving coefficient marginals.
8. **Controller-program derangement:** replays a type-valid action schedule
   from another same-shape instance.
9. **Fixed pivot/closure schedule:** same primitive ALU and maximum cycle
   budget, no learned controller.
10. **Closure truncation:** stops before degree-three/four consequences.
11. **Law-masked and same-length law-shuffled sources.**
12. **Zero/random logits through any hardening/projector diagnostic.**
13. **No-projector unconstrained predictions.**
14. **Real Shohin trunk, zero trunk, random frozen trunk, and fixed
    feature-permuted trunk.**
15. **Source-retained diagnostic:** explicitly ineligible for promotion.

The treatment must beat the strongest eligible control. Beating only a broken
or information-poor control is not evidence.

## Causal Interventions

The following are required:

- opaque-key recoding;
- redundant-generator insertion;
- determining-generator deletion;
- `do(z_i=c)` ideal extension;
- law-twin swap with fixed non-law source;
- donor sealed-algebra swap between identical non-law completion fibers;
- source-record reorder and distractor insertion;
- renderer paraphrase;
- closure-depth truncation; and
- compiler-to-query process destruction.

Donor swaps are valid only when no donor completion index, target tensor,
candidate packet, source state, or query answer crosses with the sealed algebra.
The recipient's final machine and every distinguishing query must follow the
donor law.

## Promotion Gates

The CPU mechanics phase advances only if:

- gold exact machine/query result is 100%;
- every underdetermined or determining-clause-deleted case abstains;
- all recoding, redundant-generator, `do`, and source-deletion tests are exact;
- fixed-width workspace bounds never overflow; and
- no candidate runtime dependency imports an external solver.

The neural development phase advances only if all preregistered seeds achieve:

- at least 95% `CERTIFIED` coverage on information-complete cases and zero
  `RESOURCE_OVERFLOW` on the registered gold distribution;
- at least 95% exact unconstrained sealed programs in every held-out cell;
- at least 95% law/fact counterfactual consistency;
- at least 95% donor-law following;
- at least 95% detached depth-4--12 query exactness;
- at least 99% gauge/renderer invariance;
- 100% abstention on information-insufficient cases;
- the learned controller certifies at least 95% of cases certified by the
  same emitted constraints through the fixed exact engine;
- controller-program derangement and fixed-schedule controls each trail the
  learned controller by at least 20 points;
- at least 20 points over current JASEC on collision cases;
- at least 10 points over the strongest eligible matched control; and
- paired 99% confidence lower bound above five points.

Confirmation remains sealed unless every development gate passes without
threshold changes.

## Direct Behavior Inspection

Before scoring, freeze a separate nonconfirmation inspection split of at least
32 researcher-readable episodes.
For each, preserve:

- raw source;
- direct-fact completion fiber;
- emitted sparse generators;
- quotient dimension and closure status;
- algebraic certificate or abstention receipt;
- sealed machine;
- late queries and answers; and
- matched law deletion, law swap, fact mutation, paraphrase, key recoding,
  reorder, and distractor transcripts.

The model should preserve semantics under paraphrase/recoding/reorder, change
under law or determining-fact interventions, and abstain when the source no
longer determines behavior.

Sealed confirmation sources, generators, answers, and transcripts remain
unreadable until an immutable score, independent assessment, and decision
artifact exist. Post-decision transcript release cannot authorize adaptation
or rescoring on that board.

## Path To General Reasoning

Passing this board establishes only bounded unseen-law reasoning. The same
sealed algebra mechanism must then be evaluated without architecture changes
on held-out natural task families:

- finite-state procedural instructions;
- symbolic algebra and equation constraints;
- multi-step logic;
- program-state and code contracts; and
- ordinary language questions requiring source-defined rules.

Public math/code benchmarks and direct conversation remain final capability
gates. A synthetic pass with no natural-family transfer is retained as a
scientific component, not reported as genuine general reasoning.

## Immediate Work Order

1. Implement the finite-field sparse polynomial and exact row-reduction
   mechanics as a gold/control engine.
2. Implement the three-variable falsifier and smallest balanced degree-three
   law-collision quadruple; retain the latter only as a known-shortcut mechanics
   fixture.
3. Run exhaustive information, field-semantics, closure, status, and
   gold-ceiling gates.
4. Implement the primitive field/register ALU and verify that no host pivot,
   closure, basis, or halt decision is present in the treatment path.
5. Implement a scoreless learned-controller trace pilot on generated algebra
   systems before connecting raw language.
6. Implement masked variable geometry, sealed quotient wire, anonymous
   codebook, late-query parser, and the three-process deletion boundary.
7. Recompute the exact parameter and algorithmic resource vectors.
8. Hostile-audit the complete CPU/controller boundary and matched controls.
9. Freeze the neural emitter only if all prior gates pass.
10. Do not generate scored development/confirmation data before source freeze.
