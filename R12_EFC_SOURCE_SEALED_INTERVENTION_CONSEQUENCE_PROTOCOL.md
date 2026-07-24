# R12 EFC Source-Sealed Intervention Consequence Protocol

The current-board arm is named **Source-Law Residual Alignment (SLRA)**.
Source-Sealed Intervention Consequence (SSIC) remains the broader future
family name; its behavioral arm is not admissible on the current board.

## Status

SLRA's zero-parameter current-board information gate is implemented; its
9,641,096-parameter controller is not and is no longer admissible on the
current board. SSIC is the possible broader successor
to the rejected Counterfactual Machine-Repair Lattice. It has no fit, score,
reasoning claim, or pretraining authority. The protected step-300k Shohin
checkpoint remains immutable.

`train/episode_functor_source_law_residual.py` constructs every hidden-row
candidate and evaluates only the source-stated permutation/balance residual.
Seven focused tests pass. Across eight independently generated worlds the
primitive recovers every hidden transition and observer cell, is exactly
state/action/observer/answer equivariant, preserves the exact residual multiset
under explicitly conjugated hard derangement, rejects soft transport, rejects
ambiguous visibility, and has finite gradients. This proves an available
zero-parameter law signal, not that Shohin can compile or use it.

A final claim-level hostile audit closes the neural use of this signal on the
current board. Every hidden transition residual packet is a permutation of
`(0,2,2,2,2,2,2,2)` and every hidden observer packet is a permutation of
`(0,2,2,2)`. Reading the location of zero is equivalent to reading the
completion. This is lawful information, not illicit supervisor access, but a
neural model that consumes it would only learn a motor over an already solved
answer. SLRA may remain an oracle ceiling, mechanics test, or ineligible
diagnostic. It must not enter a confirmation-time forward pass.

## Why CMRL was rejected

CMRL accepted complete machine-shaped evidence and used it as a target. One
candidate feature exposed the target probability directly. Its leave-one-out
control encoded the same target invertibly, while its observational twin
removed the candidate-choice channel. Consequently a treatment advantage
could not be attributed to intervention consequences.

SSIC keeps the useful architectural conjecture, finite model-predictive
interventions, but changes the information boundary and controls.

## Current-board applicability

The current identifiable EFC board hides one cell from each action permutation
and balanced observer, then states the completion law in the source. It does
not provide an independent set of source-visible execution trajectories.
Consequently two different SSIC claims must be separated:

1. **SLRA law residual:** admissible on the current board. Candidate
   interventions may be scored against the source-visible permutation and
   balance laws without assembling a target table. This is a zero-parameter
   mechanics/oracle result only. Its residual is label-equivalent on this
   one-hole board, so no neural treatment using it is admissible.
2. **SSIC-B behavioral consequence:** **NO-GO on the current board**. Generating
   a behavioral target by first assembling the complete relation table and
   rolling it forward is forbidden; that is the rejected CMRL target leak in
   another representation.

A legitimate future SSIC-B board would have to commit, before model fitting,
both partial anonymous machine declarations and independent source-visible
behavioral observations or execution traces. At least two machine completions
must remain compatible with the direct declarations alone, while the
independent observations identify the intended completion. Split/custody
rules must prevent the board generator or supervisor from entering the
candidate process. This is a new data protocol and is not authorized by this
architecture draft.

## Source-sealed information boundary

The candidate module never accepts transition or observer target tables. Its
only evidence object is issued inside the source compiler from:

- anonymous record validity;
- anonymous record-type distributions;
- record-to-physical-key incidence;
- physical-key assignment probabilities;
- record-local answer distributions;
- the canonical anonymous source SHA-256; and
- a compiler-instance nonce.

The issuer copies those tensors into a private capability registry and binds
their shapes, dtypes, values, source hash, compiler nonce, and configuration.
The public system accepts raw source bytes, rescans them, constructs the
anonymous record object internally, and issues the capability. Callers cannot
construct or replace it. Supervisor labels, hidden machines, late queries,
expected answers, and post-seal artifacts never enter the issuance path.

The repair module receives:

`M`: current anonymous categorical machine probabilities;

`E`: the issued record-level source object; and

`S`: the fixed schema support mask.

It does not receive an aggregated relation table. Direct record compatibility
is recomputed from `E` and `M` inside every cycle. Every record whose routed
fields directly touch the row currently being intervened upon is removed from
that row's consequence term. Candidate selection must therefore use
consequences on other source records, not a duplicate or paraphrase of the
candidate row's answer.

## Candidate computation

For each supported row `i` and legal destination/answer category `c`, form

`M_ic = retract(M with row i replaced by one_hot(c))`.

For every valid anonymous source record `r`, recompute its compatibility under
`M_ic` using the same physical-key assignment that produced the current
machine. Let

`ell_r(M) = -log compatibility(E_r, M)`.

The candidate consequence packet is

`Delta_ic = SymPool_r:not-touching(i)(phi(E_r, ell_r(M_ic) - ell_r(M)))`.

`SymPool` is shared over records and cannot see record index. Additional
packets may include internally rolled behavior signatures through depth three,
but they are compared only against record-local source constraints from `E`.
Excluding only one occurrence is insufficient: all duplicate, paraphrased, or
softly routed records with nonzero incidence on row `i` are masked.

For SLRA, append only source-declared law residuals computed from `M_ic`.
For transition row `s*` and candidate destination `c`:

`T_a^(c) = T_visible + e_s* e_c^T`;

`r_c^T = [T_a^(c) 1 - 1; (T_a^(c))^T 1 - 1]`.

For observer row `s*`, candidate answer `y`, and source-declared answer-count
vector `b`:

`O_o^(y) = O_visible + e_s* e_y^T`;

`r_y^O = [O_o^(y) 1 - 1; (O_o^(y))^T 1 - b]`.

The candidate branch receives equivariant embeddings of those residuals,
their norms, and support statistics. These are public algebraic constraints,
not target-machine comparisons. For SSIC-B, independent behavioral
observations may be appended only under a separate admitted board.

Two scientific claims must remain separate:

- **Hardcoded-law SLRA:** fixed tensor algebra constructs permutation/balance
  residuals. This is a valid architectural law executor, not evidence that
  Shohin learned to interpret a law statement.
- **Compiled-law SLRA:** JASEC must compile source text into a sealed constraint
  operator and transfer across unseen law families and renderers. Only this
  stronger variant can contribute evidence about native law interpretation.
  It requires a new board where direct facts admit multiple completions and
  paired counterfactual laws select different machines. No hardcoded residual,
  law parser, or board-specific projector may enter its candidate process.
No candidate packet may contain:

- a target transition/observer probability;
- a target category one-hot;
- a supervisor loss or label;
- a late-query result;
- an externally computed candidate score; or
- a coordinate embedding.

The base candidate branch is deliberately separate:

`b_ic = Base(current_row_i[c], direct_record_support_i[c], row_context_i)`.

Every treatment and control receives exactly the same `b_ic`. Therefore any
direct parsing shortcut is matched and cannot masquerade as a counterfactual
gain.

The treatment controller is

`z_ic = E_base(b_ic) + E_delta(Delta_ic) + W_h h_i`;

`q_ic = softmax_c(Readout(F(z_ic)))`;

`M_i_next = retract((1-g_i) M_i + g_i q_i)`.

One shared controller and GRU memory are tied across rows, categories, and
four cycles. The first three cycles use learned halt hazards. The final hazard
is fixed exactly to one and is not trainable or reported as a prediction.
Mixture weights are the ordinary first-success hazard distribution and sum to
one exactly.

## Gauge-equivariant matched controls

All controls preserve the base branch, candidate-choice channel, parameter
count, tensor shapes, executed modules, and consequence information multiset.

1. **Identity treatment.** Candidate `c` receives consequence packet
   `Delta_ic`.
2. **Conjugated-transport control.** Before labels or outcomes, sample one
   uniform random derangement permutation matrix `P_i` independently for each
   source and row from a fixed control seed. Candidate `c` receives
   `(P_i Delta_i)_c`. Under a
   category recoding matrix `G`, the control tensor is transformed as
   `P_i' = G P_i G^-1`. Uniform derangements are closed under conjugacy, so the
   distribution and each explicitly conjugated execution are gauge
   equivariant. The realized tensor is committed before labels, hidden from
   the model, and transported rather than regenerated during recoding tests.
   This preserves the complete consequence multiset and candidate spread while
   breaking only candidate-to-consequence alignment. A fixed numeric
   permutation is invalid.
3. **Baseline-consequence control.** Every candidate retains its distinct
   base feature but receives the same baseline consequence packet. This tests
   whether intervention-specific consequences matter without eliminating
   categorical choice.
4. **One-step myope.** Keep record-local and depth-one consequences; zero
   depth-two and depth-three packets.
5. **Order-erased control.** Average only the ordered action-word axes while
   preserving word length and action multiplicity.
6. **Fixed-compute control.** Force every row gate to one, halt weights to
   `(0,0,0,1)`, and execute all four cycles. This removes adaptive allocation
   without silently retaining learned row gating.

The conjugated-transport control, not leave-one-out averaging, is the decisive
causal attribution arm.

A non-permutation doubly stochastic transport is not an admissible primary
control because it mixes packets and changes the residual multiset. It may be
used only to sample a realized hard permutation.

Because every arm still receives the visible table and source law, a capable
controller may recompute the residual even when the supplied residual channel
is deranged. Therefore treatment greater than derangement supports causal use
of the aligned residual channel. A tie does not disprove law reasoning, and
neither outcome proves native law compilation unless the law operator itself
is learned from source text.

## Exact provisional parameter budget

Default controller width is `D=704`, memory width is `H=384`, and consequence
width is 160.

| Component | Parameters |
|---|---:|
| Consequence stem `160 -> D` | 113,344 |
| Base/row stem `64 -> D` | 45,760 |
| Memory initializer `32 -> H` | 12,672 |
| Two `4D` residual blocks | 7,939,712 |
| Shared GRU cell | 1,255,680 |
| Memory projection | 270,336 |
| Candidate normalization/readout | 2,113 |
| Row gate | 1,089 |
| Halt head | 385 |
| Five learned global scales | 5 |
| **SSIC addition** | **9,641,096** |

With current JASEC at 189,501,285 complete parameters, the provisional SSIC
system would contain 199,142,381 parameters and leave 857,619 below the strict
200,000,000 ceiling. This arithmetic does not admit the architecture.

## Structural no-leak gates

Before any fit:

1. The source compiler must be the sole capability issuer. Reissued,
   cross-compiler, mutated, copied, forged, stale, or post-seal capabilities
   fail closed.
2. Supervisor mutation, deletion, and replacement must leave every SSIC input
   bit-identical.
3. Late-query mutation must leave attached compilation bit-identical.
4. Raw opaque-key recoding must preserve every trainable tensor and gradient;
   equality-partition mutation must change the appropriate incidence.
5. The intervention module source closure must contain no function that
   accepts or constructs a complete target machine from supervisor data.
6. Removing every record that directly touches the intervened row must be
   verified by gradient: all such direct target fields have exactly zero path
   to that row's consequence score, while downstream records retain a nonzero
   path.
7. Unsupported schema cells remain exactly zero at every cycle and have zero
   gradient.
8. Identity and conjugated-transport arms receive bit-identical base features
   and equal consequence multisets; only candidate alignment differs.
9. State/action/observer/answer recodings commute with every cycle, gradient,
   halt mixture, and hardened machine. The control permutation must conjugate
   exactly.
10. Final halt probability is exactly one and has no trainable logit.
11. The complete live parameter receipt is recomputed at authorization and
    remains strictly below 200M.

## Information and oracle gates

Before neural training, a zero-parameter exhaustive oracle receives exactly
the SSIC-issued source object and candidate machinery. It must:

- recover every current-board hidden cell from direct declarations plus the
  source-visible completion law, without reading a target table;
- retain at least two legal candidate categories on a matched ambiguous
  negative-control board where the completion law is removed;
- fail, rather than guess, when all downstream distinguishing records are
  removed; and
- change its selected repair under a source-valid downstream
  counterfactual.

If the oracle cannot repair deep faults, SSIC is information-insufficient and
is rejected before fitting. If it repairs the ambiguous negative control, the
source object leaks forbidden target information and is rejected.

On the current board, only SLRA may pass. SSIC-B must remain
information-insufficient after direct-row withholding. A behavioral pass there
would prove that forbidden complete-machine information leaked into the
candidate process.

### Implemented zero-parameter SLRA custody gate

The implemented current-board gate is narrower than the provisional neural
architecture. `SourceLawResidualIssuer` accepts raw source bytes, decodes the
partial anonymous declarations internally, and issues identity-bound
capabilities. Its private receipt binds source SHA-256 values, issuer nonce,
tensor values, shapes, and dtypes. Copied, cross-issued, mutated, or stale
capabilities fail closed. Public residual and completion functions accept only
the issuer and its exact capability; callers cannot submit relation tensors.

Visible source rows must be exact categorical one-hots and each relation must
hide exactly one row. The internal residual primitive masks hidden rows before
candidate construction, so hidden placeholders have exactly zero gradient
while visible evidence retains finite nonzero gradients. This closes the
soft-visible-table and hidden-cell side channels found by hostile review.

The same issuer derives one hard derangement independently for every
source/relation/row from the source hash and a committed control-seed SHA-256
embedded directly in issuance bytecode rather than mutable runtime state.
It rejects identity, fixed-point, and soft transport matrices, stores the
realized controls behind source-bound capabilities, and verifies their tensor
hashes on use. Recoding tests transport and conjugate those realized tensors
rather than regenerating a numeric permutation. The gate passes 7/7 focused
tests over eight generated worlds. This admits only the hardcoded-law
information mechanic; it does not admit a neural fit or native law
interpretation.

Four successive hostile reviews are closed with no remaining P0/P1. Stored
source bytes are rehashed against independent issuance hashes on every use;
capability schemas are revalidated; and public raw-key recoding derives its
bijection from occurrence-aligned original/recoded sources, verifies unchanged
non-key structure, and conjugates the already-realized control. Neither callers
nor a mutable module binding can choose a different control after evidence.

## Neural decision

A future exact source freeze may compare JASEC, SSIC identity treatment,
conjugated transport, baseline consequence, one-step, order-erased, and
equal-parameter widening. SSIC advances only if:

- hard-machine recovery materially exceeds every control on held-out worlds;
- the gain survives unseen noncommutative words through depths 4--12;
- identity treatment beats conjugated transport with the same consequence
  multiset;
- difficult cases use more expected cycles than already-consistent cases;
- source deletion and detached late-query execution remain exact; and
- an independent hostile audit finds no P0/P1 custody or attribution defect.

Passing these gates would authorize only a separate isolated neural
qualification. It would not establish native reasoning or authorize
continuation pretraining.
