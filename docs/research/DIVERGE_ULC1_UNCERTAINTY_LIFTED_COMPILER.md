# DIVERGE-ULC1 Uncertainty-Lifted Compiler

**Status:** exact CPU mechanics passed; frozen-HSC1 matched gate authorized

**Decision date:** 2026-08-05

## 1. Causal thesis

HSC1 does not mainly lack semantic knowledge. On 12,288 fresh complete options,
the frozen failed checkpoint ranks the gold semantic template first in
99.316--100% of shifted options and the gold token alignment is Viterbi in
100%. It fails because a single globally normalized parse commits every record
to one cue kind. Cue top-1 falls to 13.845% on composition shift, and one wrong
record decision invalidates the complete packet.

ULC1 changes the computational object, not the classifier:

> A language compiler must not turn local uncertainty into irreversible global
> deletion. It seals a packed structured support lattice, lifts unresolved
> source interpretations into guarded epistemic variables, executes each
> coherent interpretation through shared state groups, and commits only after
> evidence or query invariance removes the ambiguity.

The frozen rank audit shows that K=2 retains the valid interpretation in
100% of 768 shifted episodes. Across typical six-record episodes, even binary
record uncertainty implies 64 coherent worlds. Resource-corrected DIVERGE-v4
already measured a 27.365x storage advantage over duplicated particles at 64
worlds. ULC1 joins those two independently measured facts.

## 2. Architecture

### Score producer

The first matched gate freezes the exact failed HSC1 checkpoint. It may emit
record-boundary, phase, cue, and semantic-role potentials, but it may not update
weights, inspect delayed evidence, query, answer, renderer ID, ontology ID, or
gold source objects. The hard-Viterbi control and ULC1 receive identical score
tensors.

A later trainable version, only if this matched gate passes, replaces HSC1's
span-conditioned option encoder with one record-wide encoder. This is required
so all legal phase paths share one score grid and no gold/Viterbi span is needed
to score an alternative.

### Sealed parse lattice

For each exact learned record boundary, ULC1 stores:

1. a monotonic three-cut dynamic-program chart;
2. cue-position/kind alternatives with source-occurrence provenance;
3. two option finite-state charts over the fixed 128 semantic templates;
4. contiguous alias-span commitments and physical occurrence IDs;
5. exact learned log support for every retained factor; and
6. a source commitment covering the complete raw record.

No complete parse Cartesian product is materialized. Raw source bytes,
residuals, and KV state are deleted after the lattice and occurrence table are
sealed. Scores may order support but cannot delete a grammar-valid world in the
CPU reference. The matched neural diagnostic may additionally report K=2
support, but the exact reference retains the full bounded grammar.

### Uncertainty lift

Every semantic record interpretation maps to a guarded quotient:

- each record owns one categorical `BACKGROUND / ACTIVE-left / ACTIVE-right`
  variable, with the first record restricted to the two active alternatives;
- membership and active-option fields are derived from that one category rather
  than represented as independent axes;
- phase/template/alignment witnesses remain attached to that lineage;
- option programs become guarded state patches; and
- witness variants may merge only when an independent semantic certificate
  proves identical membership, occurrence identities, programs, support, and
  future state/query behavior.

Fields from incompatible parses are never averaged. A record's membership from
one hypothesis cannot be paired with the program or alias from another.

### Shared recurrent execution

Records execute in source order. The exact gate uses the existing typed
DIVERGE transactions and groups worlds only by identical complete typed state.
Every group carries a disjoint support mask and exact mass. Delayed observations
produce verifier-checked nogoods; a nogood may remove support but never create
or repair a world. Late queries return an answer only when all surviving worlds
agree, otherwise `ABSTAIN`.

The typed exact executor is an assessor/reference, not the final model-owned
reasoning claim. A neural shared reactor is authorized only after the unchanged
score/runtime gate proves that uncertainty lifting itself is useful.

## 3. First discriminating gate

Use fresh raw-source episodes from all four existing cohorts. For each record,
the independent assessor supplies the observable effect signature produced by
the gold record program on a fixed typed probe state, then removes all source
access before execution. The signature never contains alias identity, prior
class, parse index, query, or answer. Every candidate independently predicts
the same effect signature from its own complete program, and the verifier may
remove only candidates whose predicted effect conflicts with the observation.
Initial learned top-1 must be wrong in a measured subset; evidence must identify
the valid behavioral lineage through state consequences rather than labels.

Matched arms receive the same frozen score tensors and typed transition budget:

| Arm | Runtime |
|---|---|
| A | closed HSC1 single top-1 state |
| B | complete whole-state particles admitted under the exact factorized byte and transaction budgets |
| C | two independent complete trajectories |
| D | one top-1 state replayed recurrently under the factorized transaction budget |
| E | posterior-weighted soft terminal-answer aggregation |
| F | factorized version space without delayed conflict refinement |
| G | full ULC1 factorization plus verified evidence refinement |

Additional interventions are shuffled evidence guards, packet swap between
matched episodes, forced premature top-1, state reset before query, and
post-seal source poisoning. Full hypothesis fields are never averaged.
The underdetermined query is selected late from the sealed typed state as the
first register read with certified disagreement across represented worlds. If
no such read exists, the episode fails the board contract; a safe invariant
answer is never mislabeled as a false commitment.

## 4. CPU mechanics gate

Before using a neural score tensor, the independent reference must prove on at
least 1,000 calibrated episodes:

1. 100% extensional parity with complete world enumeration;
2. 100% valid-world support immediately after compilation;
3. zero verifier-accepted nogoods that remove a valid world;
4. zero false query certificates or commitments;
5. exact monotone support and mass after every refinement;
6. exact source-order noncommuting execution;
7. 100% source-poison invariance after sealing;
8. fail-closed overflow and malformed provenance;
9. canonical byte/transaction accounting; and
10. at least 2x worlds/byte over whole particles at eight or more semantic
    worlds, reported rather than assumed.

### CPU result

The frozen seed-`202608057400` reference passes all ten requirements on 1,024
episodes and 16,896 represented worlds. The calibrated support widths are 2,
6, 16, and 42 coherent worlds; a sparse cross-record guard circuit rules out a
background-to-`ACTIVE-left` transition on held-out deeper compositions.

| Measure | Result |
|---|---:|
| Initially wrong top-1 cases | 1,024 / 1,024 |
| Final sensitive exact after verified evidence | 1,024 / 1,024 |
| Extensional parity failures | 0 |
| Valid-world deletions | 0 |
| False query commitments | 0 |
| Source-poison failures | 0 |
| Packet-swap acceptances | 0 |
| Shuffled-provenance acceptances | 0 |
| Verified nogoods | 4,096 |
| Whole-particle / factorized total bytes | 64,157,184 / 11,454,285 |
| Minimum ratio at >=8 worlds | 50,696 / 12,571 = 4.0337x |
| Whole-particle / factorized transaction applications | 205,824 / 141,056 |

The factorized representation is deliberately not promoted at two or six
worlds, where its fixed packet overhead can lose to complete particles. It
earns the storage gate at 16 and 42 worlds and shares 64,768 transaction
applications. Source-order perturbation changes the answer, overflow discards
all partial support, and post-seal source poisoning does not change packet
bytes.

The immutable report is
`artifacts/reasoning/diverge_ulc1/cpu_calibration_seed202608057400.json` with
SHA-256
`4123def2e71041987a14eef28385e6491565c34427fd93cc7fc5247fe09a061b`.
Its board commitment is
`6013ea82c2949ac33b9744964f1144520e8c158ad72fa287496e553072417478`.
This result authorizes the bounded frozen-HSC1 score/runtime gate only. It is
not learned-language, neural-execution, or model-owned reasoning evidence.

## 5. Neural pass/kill rule

Freeze the score checkpoint, seeds, cohort sizes, K, transactions, evidence,
queries, and all controls before reading result accuracy.

ULC1 passes only if:

- every episode represents at least eight coherent worlds;
- gold source support is at least 95% in every cohort;
- final exactness is at least 90% in every cohort;
- wrong-top-1 exact recovery is at least 90% in every cohort;
- final sensitive-query exactness beats hard Viterbi and equal-memory whole
  particles by at least 10 points in every shifted cohort;
- final exactness beats equal-transaction single-state recurrence and soft
  aggregation by at least 10 points in every shifted cohort;
- invariant queries are 100% and underdetermined queries never falsely commit;
- provenance and state-reset interventions each lose at least 20 points;
- packet swaps are rejected and post-seal source poisoning is bit-invariant;
- verifier-derived guards never remove a represented gold world;
- conservative whole-particle component bytes are at least 2x the factorized
  packet bytes in every shifted cohort; and
- duplicated transaction applications are at least 1.25x unique shared
  applications in every shifted cohort.

If K=2 full particles equal or beat ULC1 at matched memory/FLOPs, the sharing
claim fails. If support is absent before evidence, the compiler fails. If
evidence helps only through an exact host operation unavailable to the future
neural reactor, retain the result as mechanics only and do not promote it as
model-owned reasoning.

## 6. Claim boundary

Parse forests, CRFs, packed charts, version spaces, belief propagation, particle
filters, delayed evidence, and guarded execution are established ideas. The
candidate contribution is narrower: source-sealed lifting of language-parse
uncertainty into coherent DIVERGE state variables, shared noncommuting
execution, verified contradiction refinement, and query-invariant commitment.
The rank audit and any CPU pass do not by themselves establish native or general
reasoning.
