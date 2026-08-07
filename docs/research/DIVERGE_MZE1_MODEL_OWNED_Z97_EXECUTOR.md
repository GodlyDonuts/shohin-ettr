# DIVERGE-MZE1 Model-Owned Z/97 Executor

Status: confirmed across development and five fixed confirmation seeds.

## Capability hypothesis

The confirmed CWC1 -> EWC1 -> NPL2 path should not need a hard-coded mapping
from operation identity to state transition. A tied, outcome-supervised
finite-field law owner can infer each opaque operation's complete 2x2 linear
action and recurrently execute it without importing the exact PL1 operation
function.

MZE1 changes only the executor. Confirmed CWC1 selection, EWC1 structural
transcription, NVE1 evidence, NPL2 branch-local plasticity, EIC1 late queries,
data, seeds, prompts, verifier, and all thresholds remain fixed.

## Architecture

For each of eight opaque operation identities and two output registers, MZE1
learns a categorical distribution over coefficient rows `(a,b)` with
`a,b in {-2,-1,0,1,2}`. Training exposes only random input states and observed
successor states. It marginalizes every row compatible with each observed
outcome:

```text
L = -log sum_{(a,b): ax+by = target mod 97} p(a,b | operation, output)
```

Autonomous execution commits the highest-weight row once and applies the same
learned law at every recurrent step. The candidate runtime contains no import
of `diverge_pl1_data` or the exact `apply_operation` function. Modular
arithmetic and the bounded coefficient catalog are explicit architectural
priors; the operation semantics are learned.

## Frozen component gate

One treatment and one equal-parameter/equal-update shuffled-outcome control
start from identical state and receive 256 updates of 4,096 transitions. The
component passes only if:

- treatment is exact on all `8 * 97 * 97 = 75,272` one-step transitions;
- treatment is exact on 2,000 unseen programs at each depth 4, 8, 16, and 32;
- the shuffled-outcome control is at most 5% exact against true operations;
- the runtime source imports no exact operation implementation; and
- parameter and update budgets match.

A miss closes MZE1 without coefficient-range, seed, update, learning-rate, or
catalog variants.

## Frozen composition gate

Only a component pass opens one development run. The learned transition owner
replaces the exact transition by hash-bound injection before any NPL2 arm is
evaluated. The exact verifier remains unchanged and is explicitly disclosed.
The development report must pass every existing CWC1/EWC1/NPL2 condition and
must additionally preserve the learned executor checkpoint/state receipt.

Only a conjunctive development pass opens the five already-fixed confirmation
seeds. A confirmation pass qualifies removal of the hard-coded operation
semantics from this controlled reasoning path. It does not remove engineered
candidate generation, the exact verifier, the synthetic mini-language, or the
finite-field architectural prior, and it is not a public reasoning claim.

## Frozen result

The 400-parameter treatment and equal-size shifted-outcome control each train
for 256 updates of 4,096 outcome-only transitions. Treatment is exact on all
`75,272/75,272` one-step transitions and all 2,000 held programs at each depth
4, 8, 16, and 32. The shifted control scores `0.2657%` against the true
operations. The candidate runtime imports neither the PL1 data module nor the
exact operation function. Checkpoint and component-report SHA-256 values are
`0526e0e4...a9124c` and `5846528b...349bf`.

Development job `744698` passes the complete composition at `84.8145%`,
exactly equal to oracle, but takes 16m45s because it re-materializes the
committed coefficient rows on every microstep. A semantics-preserving cache
commits those same rows once. Replay `744700` completes in 4m34s and reproduces
all score, control, gate, state, and custody fields other than elapsed time and
packaging paths. Its result SHA-256 is `5fc8d0d3...34cec4`.

Five fixed confirmation jobs `744701`--`744705` run concurrently, followed by
aggregate `744706`. Aggregate late-query exactness is
`35,066/40,960 = 85.6104%`, exactly equal to oracle on every seed. The strongest
non-oracle remains `3.9185%`; all original WORLD, EVIDENCE, QUERY, plasticity,
reset, shuffled-credit, wrong-branch, transplant, eligibility, rollback,
source-deletion, and owner-custody gates pass. Aggregate SHA-256 is
`14f05b8a8a7f30f01418c0d00034a91e528da18395652304dd20d92e0325dbdd`.

MZE1 therefore removes the hard-coded operation semantics from the strongest
controlled Shohin path. It does not remove exact verification, explicit
complete-candidate construction, the narrow mini-language, or the bounded
linear-law catalog. Those are the next capability boundaries.
