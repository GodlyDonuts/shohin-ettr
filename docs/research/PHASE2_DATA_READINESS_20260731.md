# Shohin Phase 2 Data Readiness

Status: operational snapshot, 2026-07-31 04:00 EDT.

This report distinguishes physical availability from training admission. A
token count is not an admission decision. Shohin's target is quality per
parameter, so every new general source remains at zero optimizer weight until
its exact selected payload passes provenance, semantic, contamination,
privacy, deduplication, holdout, and equal-token utility gates.

## Executive Summary

- General data currently authorized for optimizer use: **62,426,256,278
  manifest tokens** from the historical FineMath, OpenWebMath, and Python
  streams.
- Architecture-native ETTR data currently authorized for optimizer use:
  **40,000 train semantic cores / 2,560,000 rows / 1,351,680,000 charged
  positions**, plus a disjoint 5,000-core development split and sealed
  confirmation data outside the optimizer.
- Highest-value nearly ready general candidate: **4,350,429,033-token
  domain-balanced peS2o sensitive residual**. It has physical verification,
  exact retained-document review, and a zero-exclusion sensitive rescan, but
  still lacks cross-source residualization, immutable holdouts, and
  equal-token transfer/retention evidence.
- Largest broad-language candidate: **8,763,527,685-token FineWeb-Edu
  score-4+ selection**. Tokenization completed, but the original job failed
  closed on a transient Lustre identity change during final verification.
  Recovery verifier `757374` is active; exact-document review `757375` is
  dependency-held.
- No new candidate is silently mixed into training. Current ETTR experiments
  use only the frozen audited release.

## Readiness Matrix

| Corpus | Exact size | Evidence already passed | Remaining blockers | Optimizer status |
|---|---:|---|---|---|
| Historical FineMath/OpenWebMath/Python | 62,426,256,278 tokens | manifest scans, evaluation decontamination, stable token streams | legacy streams lack v3 document ledgers; FineMath nesting and cross-source uniqueness cannot be claimed without reconstruction | admitted historical stream |
| ETTR packet-v2 train | 40,000 cores; 2,560,000 rows; 1,351,680,000 charged positions | complete qualification, materialization, independent audits, main/confirmation separation, exact packet-context audit, immutable release | learning recipe must replicate causal WORLD and COMMAND effects across both seeds | admitted ETTR stream |
| peS2o balanced sensitive residual | 4,350,429,033 tokens; 718,586 documents | pinned OA licenses, domain balancing, two physical verifications, 1,000-document exact review, three credential-bearing rows removed, complete zero-exclusion rescan | exact/near cross-source residualization, train/document/domain split, full partition audits, cross-holdout and equal-token utility | quarantined candidate |
| FineWeb-Edu score 4+ | 8,763,527,685 tokens; 7,759,205 documents | pinned 140-file source, strict score/language/length/repetition/boilerplate/extraction filters, exact deduplication, domain cap, evaluation decontamination, complete v3 manifest | recovery verification and publication, exact retained review, semantic/human gate, exact/near residualization, holdouts, privacy, utility | quarantined partial; jobs `757374 -> 757375` |
| FinePDFs-Edu English parent | 6,085,701,378 tokens; 2,022,608 documents | pinned physical source, complete v3 verification, 10,000-document exact review | wholesale source rejected; use selected policy arms only | rejected wholesale |
| FinePDF core policy arm | 100,004,847 tokens; 31,702 documents | sealed v3 corpus, exact 1,000-document review, reviewer-blind matched packet | complete human labels, exact/near residualization, holdouts, privacy/license audit, identical-start utility | quarantined challenger |
| FinePDF residual policy arm | 100,000,971 tokens; 29,641 documents | sealed v3 corpus, exact 1,000-document review, reviewer-blind matched packet | same as core arm; core currently appears denser but has not cleared the human gate | quarantined control |
| Essential-Web reasoning core | 308,147,626 tokens | pinned hard reasoning/correctness/extraction policy, final v3 corpus, exact 1,000-document review across 886 domains | human semantic gate, exact/near residualization, holdouts, privacy, equal-token utility | quarantined challenger |
| Nemotron Formal Logic | 97,941,363 tokens | pinned source, final v3 verification, exact retained review, zero sampled exact contamination | narrow synthetic multiple-choice style, cross-source residualization, independent transfer holdouts, equal-token utility; maximum proposed weight 1% | zero-weight challenger |
| DCLM Baseline | 25,000,001,792 tokens | structural scan and encoding health | mixed forums, dated news, and SEO-like content; requires document-level quality stratification and all downstream gates | held wholesale |
| OpenMathInstruct-2 PT | 5,000,000,144 tokens | structural scan and encoding health | unreliable generated derivations observed; requires answer and trace verification plus all downstream gates | held wholesale |

## Quality Interpretation

The historical stream is large enough to train, but it is not the desired
final Phase 2 mixture. It is math/code heavy, and its legacy packing predates
the document-level provenance needed for exact cross-source residualization.
It is usable as an incumbent baseline and continuity stream, not proof of a
pristine broad corpus.

The new v3 candidates are materially better controlled. They retain exact
document provenance after packing, bind every source and tokenizer input, can
remove a specific document reproducibly, and can support domain-level
holdouts. Their remaining utility gates are intentional: upstream quality
labels have already admitted content farms, institutional noise, generated
errors, and narrow templates in manual samples.

The best immediate expansion path is:

1. recover and review the completed FineWeb-Edu score-4+ payload;
2. exact- and near-residualize FineWeb, peS2o, Essential-Web, and bounded
   FinePDF/Nemotron challengers in declared retention-priority order;
3. freeze train, representative-document, and whole-domain holdouts;
4. run privacy, license, contamination, and structural audits on every
   physical partition;
5. run same-initialization, equal-token utility arms against the historical
   incumbent and score every arm across the complete holdout matrix;
6. admit only sources whose paired evidence improves aggregate utility
   without material general-language or ETTR causal regression.

## Current Training Interpretation

Nine physically verified H100 lanes are running matched 5,000-update ETTR
experiments. The observed seed-1 causal signal is real on both 32- and
128-batch source-deleted boards, but the opposite seed has not yet shown both
WORLD and COMMAND causal margins. No large run is authorized from a one-seed
effect. The first recipe that passes every strict gate in both preregistered
seeds will be promoted across the largest healthy worker set.

This is not compute hesitation. It is avoiding an expensive scale-up of a
non-replicating mechanism while all available verified H100s run identifiable
experiments and the CPU data lane advances independently.

