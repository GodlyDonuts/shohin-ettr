# DIVERGE-SOT1 Result: Stage-Local QUERY Ownership Fails

Status: closed negative after the one sealed evaluation and one read-only
attribution diagnostic. No SOT1 retry family is authorized.

## Sealed result

Newton job `744146` completed normally on `evc32`. The immutable evaluation is
`artifacts/reasoning/diverge_sot1/result_baa50a1_r1/evaluation.json`, SHA-256
`a38f95a691b7177e7e8fe6db447950033ae039c542410354d629bb675cdd6e7c`.

The qualified owners retained their contracts:

- WORLD programs: `256/256`;
- natural EVIDENCE transactions: `3,072/3,072`;
- fully sealed episodes: `256/256`;
- protected TOL3/NVE1: `1,024/1,024` and `256/256`;
- WORLD and EVIDENCE owner hashes: bit-identical before and after QUERY fit.

The fresh isolated QUERY owner failed:

- query transactions: `485/768`;
- sensitive / invariant / underdetermined: `0/256`, `243/256`, `242/256`;
- sensitive terminal answers: `6/256`;
- forced QUERY role swap answers: `256/256`.

The promotion conjunction is false. The role-swap result localizes a complete
semantic inversion, not a near-threshold reasoning miss.

## Single read-only attribution

`train/diagnose_diverge_sot1_query_ownership.py` replays unchanged weights over
the sealed board and a complete mode-by-renderer counterfactual matrix. Its
immutable result is
`artifacts/reasoning/diverge_sot1/diagnostic_baa50a1_r1/query_ownership.json`,
SHA-256
`c186884ccb4698d1d1faf3912cc48eb2635e073b8f6152da46d11e30b9c0010a`.

The sealed board confounds query mode and renderer. Removing that confound
shows the isolated QUERY owner is:

| Renderer | Exact | Complete role swap | Total |
|---|---:|---:|---:|
| 0 | 0 | 768 | 768 |
| 1 | 712 | 56 | 768 |
| 2 | 724 | 44 | 768 |

The qualified EVIDENCE symbol head is not a drop-in query solution: it reaches
only `1,132/2,304 = 49.13%` across the same counterfactual matrix. All owner
hashes are identical before and after the diagnostic.

## Decision

Close lifecycle-stage-local query classification. Do not run SOT1 width,
duration, seed, renderer, optimizer, parser, or loss variants. The pre-staged
NPW1 narrative-WORLD route required SOT1 PASS and is therefore not launchable.

The one allowed structural successor changes the ownership axis: one
exchange-equivariant semantic REFERENT primitive must serve both EVIDENCE and
QUERY, while qualified WORLD and numeric-evidence owners remain immutable.

