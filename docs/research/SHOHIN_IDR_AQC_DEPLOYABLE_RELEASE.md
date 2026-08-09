# Shohin IDR+AQC Deployable Release

Status: qualified architecture packaged and exercised end to end on 2026-08-09.
This is a deployment result, not a new benchmark or causal-mechanism result.

## System

The strongest measured Shohin system is now represented by one immutable
delta release over pinned `Qwen/Qwen3.5-9B` revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`:

1. `draft_adapter.pt` produces one complete model-owned internal draft;
2. `revision_adapter.pt` reads the source and draft and emits a trained
   revision;
3. `draft_adapter.pt` produces the matched unchanged continuation from the
   same source and draft; and
4. `commit.pt` reads both complete candidates and selects one coherent
   trajectory.

No external proposal model, verifier, correctness signal, benchmark label,
task router, or tool is used at inference. The complete path is implemented by
`train/hf_idr_aqc_interact.py`; immutable packaging is implemented by
`pipeline/package_idr_aqc_release.py`.

## Qualified evidence

The release binds the existing protected product evidence:

| System | Solved | Five-domain macro |
|---|---:|---:|
| unchanged continuation | `316/538` | `67.263%` |
| trained revision | `374/538` | `75.005%` |
| learned whole-trajectory commit | `383/538` | `75.815%` |
| coherent oracle ceiling | `399/538` | `78.619%` |

The learned commit is useful, but the antisymmetric treatment beat its matched
independent-score control by only one answer. The release therefore claims
coherent learned commitment, not a uniquely causal antisymmetric primitive.

## Immutable lineage

- private source commit: `8f0bd8d`;
- complete runtime: `idr_aqc_8f0bd8d_r2`;
- runtime `SHA256SUMS` SHA-256:
  `2d4e8fd845bdd023fca5a01c767ed218909ab810fa4e06b25d2888f90f3388c5`;
- release: `idr_aqc_release_8f0bd8d_r1`;
- release manifest SHA-256:
  `554e841f71edd3a19063411348340e337532db2db05dd5e1e2adc25a3d347e7b`;
- release `SHA256SUMS` SHA-256:
  `0dad031312dec0859e35bb7e9daea8aef688ef350b9053f587fba5acdc9c58c5`;
- interaction prompt SHA-256:
  `980f025b3da68fc6335ceca283afd5737c3c0577b276102533609fe7c49cf1f3`.

The packager verifies the base config, both adapter checkpoints and reports,
the trained revision's draft warm start, the qualified commit checkpoint and
report, and the protected product report before writing. The runtime verifies
every packaged file and the external base config before model load. It refuses
unqualified commits, hash drift, incomplete file coverage, output overwrite,
candidate truncation, and order-inconsistent commitment.

## End-to-end smoke

Job `747423` completed on one H100 in `3m26s`. The atomic interaction report is
`smoke_report.json`, SHA-256
`47e1bd3ab2bd81132de5f40e038f553eebcd210c436fb01c12acfeb9b2f7171b`.
The five prompts are original non-benchmark interactions covering arithmetic,
modular composition, counterfactual ordering, thermodynamics, and Python.

| Stage | Wall time | Peak CUDA | Generated tokens |
|---|---:|---:|---:|
| internal draft | `83.912s` | `18.946 GB` | `426, 768, 661, 198, 30` |
| trained revision | `17.427s` | `19.135 GB` | `8, 8, 12, 213, 100` |
| unchanged continuation | `70.862s` | `19.135 GB` | `532, 69, 755, 217, 210` |
| whole-trajectory commit | `3.934s` | `19.606 GB` | no generation |

Commit order consistency is exact and maximum swap error is `0.0`. No commit
candidate was truncated. One internal draft exhausted its 768-token budget,
but it had already derived and verified the correct answer.

Manual transcript inspection finds correct final substance on all four
math/logic/science prompts. The code attempt identifies the central
predicate-twice defect and proposes a one-pass stable algorithm, but the
selected response includes prose and omits the requested standalone function
definition, so it fails that interaction contract. The commit chooses the
unchanged continuation on all five prompts. This smoke therefore establishes
that the packaged architecture executes faithfully; it does not add evidence
that commitment improves arbitrary out-of-distribution prompts.

Two preserved pre-result failures preceded the valid run. Job `747421` failed
before model load because the first runtime archive was incomplete. Job
`747422` executed every model stage but could not write into the intentionally
read-only release directory. Both failures changed only packaging/output
location; job `747423` used the same model artifacts, prompts, generation
budget, seed, and selection policy. Total charged wall time across all three
jobs is approximately `0.126` H100-hours.

## Boundary and next use

This release makes the strongest existing Shohin reasoning architecture
reproducible and directly inspectable. It remains a multi-pass 9B system, not
the 125M scratch checkpoint and not a newly proven MoE mechanism. Future work
should use this package as the positive dense control. It should not tune the
commit policy on these five qualitative prompts or reinterpret this smoke as a
benchmark.
