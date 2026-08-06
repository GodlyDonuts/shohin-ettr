# DIVERGE-SC1 Raw-Source Object Compiler

**Status:** closed; read-only failure localization complete

**Decision date:** 2026-08-05

## 1. Decision

DIVERGE-v0 is closed as a broad architecture promotion because its corrected
factorized representation misses the frozen `>=2x` storage advantage at four
worlds. Its exact high-ambiguity (`>=8` worlds) delayed-recovery mechanism is
retained. The next experiment does not alter that executor. It removes the
successful component pilot's hidden source scaffolding.

The token-role/source-copy pilot received each gold physical record and each
gold option as a separate model call. It proved that frozen SmolLM2 residuals
can classify finite roles and copy complete options, but not that a model can
construct those objects from one raw source. The earlier whole-mention span
gate did read a complete source, but decoded fields independently and failed
under lexical shift through duplicate and missing roles.

DIVERGE-SC1 tests one hypothesis:

> A source-local occurrence ledger plus a globally scored complete-record
> parser can turn one unsegmented language source into the same exact sealed
> DIVERGE packet more reliably than independent token decisions, because the
> legal object is a coherent pair of complete options rather than a bag of
> locally likely fields.

This is a compiler hypothesis. It is not a DIVERGE promotion, a general
reasoning claim, or permission for continuation pretraining.

## 2. Only architectural change

The model reads the complete raw `WORLD` source once and emits:

1. source-token role scores for candidate/background cues, nominal alias
   boundaries, favored/reserve priors, and ordered action identities;
2. boundary scores for every source-token gap; and
3. no record list, option list, alias dictionary, renderer ID, ontology ID,
   answer, query, state, or execution outcome.

The hard compiler creates two separate ledgers:

- **occurrence ledger:** every selected physical alias span has a unique
  source address even when its bytes equal another mention;
- **nominal ledger:** selected occurrences with exactly equal normalized source
  bytes share a nominal signature, without fusing their physical addresses.

It globally scores complete nonoverlapping records. Every accepted record has
one source-local kind cue and exactly two nonoverlapping complete options. Each
option has exactly one alias span, one prior, and one legal ordered action
program. The decoder may select a locally second- or third-ranked field only
when the total complete parse scores higher. Invalid or overflowing parses
fail closed. Source-position canonicalization defines option IDs; generator-
private option ordering is never exposed.

The CPU calibration cap is `4,096` complete record proposals per episode. An
initial `1,024`-proposal calibration failed closed on one 452-token,
nine-record composition episode; that report is retained as a pre-freeze
capacity negative. The `4,096` cap is frozen before any neural score.

After hard compilation, only canonical occurrence addresses, nominal
commitments, priors, programs, record provenance, and accounting receipts enter
the existing DIVERGE packet. Raw source bytes, source residuals, and source KV
state are deleted before delayed evidence, execution, or query.

## 3. Grammar firewall

Allowed fixed constraints are:

- nonoverlap and source order of physical spans;
- one alias, one prior, and one legal action program per option;
- exactly two options per accepted record;
- one record-kind cue per record;
- source-local gap consistency and bounded span/candidate counts; and
- exact-byte nominal equality among already selected alias spans.

Forbidden constraints are:

- gold record/option/alias spans at inference;
- a renderer-specific parser, alias dictionary, or action phrase lookup;
- DIVERGE world state, trajectory, delayed evidence, query, answer, or target
  packet in decoding;
- retry after execution, host semantic repair, or answer-guided selection; and
- merging physical occurrences merely because their nominal bytes match.

The supervisor may use gold spans and packet fields for training and scoring.
The candidate runtime may not.

## 4. CPU nontriviality gate

Before neural training, an independent generated board must cover:

- 2--6 candidate records plus 1--3 background records;
- one- through four-token aliases and repeated nominal aliases at distinct
  physical occurrences;
- record and option order permutations;
- variable inter-record connectors with no supplied record offsets;
- local prior/action/alias decoys that create rank-two or rank-three fields;
- all four noncommuting finite programs; and
- train, lexical-shift, renderer-shift, and composition-shift cohorts.

The candidate CPU decoder receives only tokens and synthetic unary/boundary
scores. An independent exhaustive assessor receives annotations. Freeze neural
thresholds only after calibrating this board.

Required CPU outcomes:

1. `100%` exact parity between the joint decoder and exhaustive best legal
   parses on at least 1,000 episodes;
2. `100%` packet reconstruction when supplied calibrated gold-plus-decoy
   scores;
3. at least 25 points exact advantage over independent local decoding using
   the identical scores;
4. at least 95% of episodes contain a true field locally ranked second or
   third;
5. shuffling boundary scores or fusing occurrence and nominal identity reduces
   exact reconstruction by at least 20 points;
6. alpha renaming, consistent source-position reindexing, and post-seal source
   poisoning change no compiled semantics; and
7. zero accepted overlapping records/options, duplicate option fields, false
   nominal merges, or cap violations.

A CPU miss closes this decoder before H100 use.

## 5. Neural matched gate

If CPU mechanics pass, use the frozen SmolLM2-135M parent and the same full-
source residual layer, data, updates, seeds, and packet evaluator for:

| Arm | Decoder |
|---|---|
| A | existing per-record/per-option role-copy scaffold ceiling |
| B | independent local token roles from one raw-source pass |
| C | exact-surface span quotient from one raw-source pass |
| D | DIVERGE-SC1 complete joint object decode |
| E | D with occurrence and nominal ledgers fused |
| F | D with cross-option/record coherence removed |
| G | D with shuffled boundary scores |

Arm A is a ceiling, not a fair autonomous model. B--G must share the exact
encoder and unary scores where their interface permits. The treatment may not
receive additional semantic labels or source views.

The frozen neural pass gate is:

- development and shifted gold-support recall `>=99%`;
- development and shifted exact packet `>=95%`;
- shifted strict packet at least 15 points above B and C;
- every held-out cohort exact packet `>=90%`;
- five of five seeds above B, with at least four of five passing all floors;
- zero distractor records in exact packets and zero false nominal merges;
- source deletion and post-seal poison invariance exactly `100%`; and
- at least 20-point collapse for the causally relevant E/F/G controls.

The second and fourth bullets overlap. For adjudication, the stricter written
requirement controls: **every shifted exact-packet cohort must reach at least
95%**. The executable's `>=90%` `shift_packet` field is only a coarse process
exit condition for dependency dispatch and cannot override this scientific
floor. Follow-on seeds remain scheduler-held until the report is checked
against the stricter threshold.

Failure closes this raw-source compiler family. Passing authorizes one
end-to-end DIVERGE packet/recovery evaluation, not long pretraining.

## 6. Accounting

Report complete parameters, trainable parameters, charged source tokens,
candidate record/option counts, structured-decoder operations, peak activation
memory, wall time, and packet bytes. Dynamic-programming work and temporary
candidate storage count even when they add no parameters.

## 7. Novelty boundary

Occurrence ledgers, nominal equality, segmental parsing, structured prediction,
object files, and hard packets are not individually novel. The bounded
candidate is their source-sealed use as a language-to-factorized-DIVERGE
compiler that preserves physical occurrences while committing one coherent
complete packet. Any claim must remain at that conjunction and only after the
matched neural gate.

## 8. CPU result

The calibrated CPU gate passes. Six focused unit tests cover exact/reference
parity, local-versus-joint separation, noncommuting action order, repeated
nominal identities, alpha renaming, post-seal poisoning, and record/option
nonoverlap. The final seed `202608056100` board contains 1,000 episodes, 250 in
each of the four frozen cohorts.

| Measure | Final rate |
|---|---:|
| Joint exact reconstruction | 100.0% |
| Independent local exact | 0.0% |
| Pair-disabled exact | 0.0% |
| Boundary-shuffled exact | 0.0% |
| Joint/reference parity | 100.0% |
| Alpha-renamed exact | 100.0% |
| Post-seal source-poison invariance | 100.0% |
| Local rank-two/three stress present | 99.7% |
| Occurrence-fused control exact | 5.2% |
| Overflow | 0.0% |

The decoder considered a mean 59.74 complete options and 305.705 complete
records per episode. The retained pre-freeze 1,024-candidate calibration is
999/1,000 exact because one 452-token, nine-record composition episode has
1,124 valid scored record proposals and fails closed. The frozen 4,096 cap
admits it without changing any score, data, decoder objective, or gate.

Calibration/final report SHA-256 values are
`61dd90839b534f1e934d5bc3a9f2c7e88b1a99be8bc7b84fa65a85de5d48c4c2` /
`1626e3fe5fbba89203bf76c5368da8fe5a847d398cec4d2dde02bbadfaa031f0`.
This establishes exact and nontrivial structured mechanics under synthetic
scores only. It authorizes one neural raw-source seed; it is not learned
language evidence.

## 9. Reconciliation with the external architecture proposal

The external proposal's central prescription is retained: premature certainty
must be structurally avoidable, incompatible worlds must remain coherent, and
commitment must be licensed by evidence or query invariance. Its individual
ingredients are not treated as novel. Version spaces, guarded execution,
particles, conflict clauses, typed state, and source deletion all have prior
art.

The empirical record narrows the proposal in two ways:

1. Resource-corrected DIVERGE-v4 misses its frozen sharing gate at four worlds
   (`1.893x`, below `2x`). The current claim is therefore restricted to the
   observed high-ambiguity regime, where sharing first clears the threshold at
   eight worlds and grows to `27.365x` at 64 worlds.
2. The protected token-role compiler was not autonomous because it received
   gold record/option segmentation. SC1 is the first bounded test of whether a
   single raw-source pass can construct the coherent packet. A pass qualifies
   this source boundary; a failure leaves DIVERGE as exact synthetic mechanics
   without a learned raw-language front end.

This reconciliation also supersedes the review's original FCPT recommendation.
PCSD and FCPT remain closed by their measured negatives. Full particles remain
the principal matched scaffold/control, while CSDC's protected role-copy result
is not replaced or weakened.

## 10. Neural result

Newton job `742328` completed all 1,200 updates / 9,600 charged episodes, wrote
its immutable report and checkpoint, and exited nonzero because the capability
gate failed. Training itself was stable and deceptively strong: final role,
boundary, and pair accuracies were `99.999994%`, `100%`, and `98.381054%`;
loss was `0.061805`; throughput was `85.294` episodes/s; peak allocated GPU
memory was 796,946,432 bytes.

Autonomous complete-object decoding failed decisively:

| Cohort | Gold support | Exact packet | Overflow |
|---|---:|---:|---:|
| Train | 0.781% | 0% | 58.984% |
| Lexical shift | 0% | 0% | 47.266% |
| Renderer shift | 0% | 0% | 32.422% |
| Composition shift | 0% | 0% | 40.625% |

Alpha-renamed exactness, independent decode, pair-disabled decode, and
boundary-shuffled decode are all zero. Post-seal source-poison invariance is
100%, but that is the only passing executable gate. The four held follow-on
seeds `742329--742332` were canceled without allocation.

The immediate diagnosis is combinatorial error amplification, not insufficient
local fit: a small false-positive rate in a quadratic pair graph creates enough
plausible complete records to erase support or hit the hard proposal cap. One
no-gradient audit may substitute the already-accepted CPU oracle role,
boundary, and pair scores componentwise to localize the necessary failure. It
may not search thresholds, update weights, or reopen SC1.

Report/checkpoint SHA-256 values are
`1a23d1aaae3276d54ec8d27abea266b822b0c9f28a058951dd2d942108d59059` /
`7b5348cacb1772bf45e34442e94010db71a6be20bd8d689477d037ac5fee2ffd`.

## 11. Read-only failure localization

Job `742457` evaluated 128 frozen episodes (32 per cohort) without gradients,
threshold search, or parameter changes. It substituted the accepted CPU
calibrated role, boundary, and pair factors into the failed checkpoint one
component at a time.

The causal result is exact:

| Substitution | Train | Lexical | Renderer | Composition |
|---|---:|---:|---:|---:|
| Learned all | 0% | 0% | 0% | 0% |
| Oracle boundary only | 0% | 0% | 0% | 0% |
| Oracle pair only | 0% | 0% | 0% | 0% |
| Oracle role only | 6.25% | 0% | 0% | 0% |
| Oracle role + boundary | 6.25% | 0% | 0% | 0% |
| Oracle boundary + pair | 0% | 0% | 0% | 0% |
| **Oracle role + pair, learned boundary** | **100%** | **100%** | **100%** | **100%** |
| Oracle all | 100% | 100% | 100% | 100% |

The learned boundary is exact over this audit: 1,494 true positives, 34,064
true negatives, and zero false positives or misses. The pair head has 4,697
true positives, 7,942 false positives, 1,263 misses, and 445,271 true
negatives: `37.162%` precision and `78.809%` recall despite high aggregate
accuracy. Active-role detection has zero false positives and `98.873%` recall,
but record-kind cues and alias starts are confused with other active roles.

Those errors amplify combinatorially. Learned decoding creates about
111--140 option candidates per episode versus 54--63 under calibrated factors,
and 1,976--2,875 complete-record candidates versus 273--328: roughly `2x`
options and `7.2--9.4x` records. This explains both support deletion and the
hard-cap overflows.

The conclusion is architectural: independent dense edge classification plus
thresholded Cartesian proposal generation is not a viable source boundary for
support-preserving reasoning. Any future, separately frozen successor must
construct bounded whole-record assignments by design (for example, a
boundary-first typed matching object with exact-one constraints). It may not
be presented as a threshold, seed, width, duration, or loss repair of SC1.

Audit report SHA-256 is
`e6dd4874029f653f808c426d04b15a98da390c67c83b30429e9d015b00ab9799`.
