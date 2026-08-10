# LTR1: Ledger Transaction Revision

Status: CPU admission failed; exact lane closed before GPU use

Date: 2026-08-10

## Hypothesis

CTE1 produces a causal executable ledger on `598/666` development rows but
solves only `134`. Its failure is semantic program selection, not arithmetic
execution. Raw-natural-text pointer editing is already closed because wrong
and correct trajectories share too little contiguous text. LTR1 asks whether
canonical transaction records create useful edit locality by construction.

The proposed later model-owned system is:

```text
source -> CTE1 proposal ledger
source + proposal ledger -> record edit policy
generic record editor -> committed ledger -> frozen LAM1 -> answer
```

Revision operates on complete addressed records, never arbitrary prose
characters or averaged states. Allowed later actions are `KEEP`, `DELETE`,
`REPLACE`, `INSERT`, and `COMMIT`. Training-only gold ledgers determine edit
scripts. At inference there is no gold ledger, verifier, answer label, solver,
or semantic host repair.

## Frozen CPU Admission

Before any training-data generation or GPU fit, compare the exact immutable
CTE1 aligned development proposals with their canonical gold ledgers. Extract
only complete `<<...>>` records and compute an exact record-sequence LCS and
minimum insertion/deletion/replacement distance. Source identity, ordering,
and all invalid/exhausted proposals remain visible in the accounting.

Advance only if every condition holds:

- all 666 source identities join exactly once;
- at least 500 wrong proposal ledgers contain one or more complete records;
- mean gold-record copy fraction on wrong proposals is at least 35%;
- median gold-record copy fraction on wrong proposals is at least 25%;
- at least half of wrong proposals are repairable with at most two record
  edits; and
- no public-test or holdout data are read.

These thresholds test whether record editing materially shrinks the semantic
generation problem. A miss closes exact LTR1 before GPU use. It may not be
rescued by character/token matching, fuzzy arithmetic equivalence, filtering,
or threshold changes.

## Conditional Neural Gate

Only a CPU pass permits one separately frozen paired record-edit canary. It
must use source-disjoint train/development identities, exact CTE1 proposals,
record-level gold scripts, deterministic execution, and matched hidden-ledger,
record-shuffled, label-permuted, and full-regeneration controls. Its numerical
gate must be frozen after corpus custody and before model output. No such fit
is authorized by this document alone.

## Claim Boundary

LTR1 would test structured model-owned draft revision. It is not a retry of
raw PSET1 byte replacement, a claim that CTE1 is already capable, or permission
to tune on public GSM8K.

## Frozen Result

Immutable CPU job `750093` completed in two seconds from runtime
`225205c` (manifest `78ab0ac9...ebbcb`). All 666 identities join once.
Among the 532 wrong CTE1 proposals:

- 515 contain at least one complete transaction record;
- mean gold-record LCS copy fraction is `9.9787%`;
- median gold-record copy fraction is `0%`;
- 90th-percentile copy fraction is `33.33%`;
- median record edit distance is three; and
- only `143/532 = 26.8797%` require at most two record edits.

The minimum copy-fraction and two-edit gates fail. A ledger editor would still
need to generate nearly the complete semantic program on most wrong rows, so
the proposed structured revision does not create the required advantage.
Exact LTR1 closes before training-data generation or GPU use. Report SHA-256
is `e03509c6...e0a`; public test and holdout remain sealed.
