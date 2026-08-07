# DIVERGE-MZE1 Model-Owned Z/97 Executor

Status: frozen before end-to-end scoring.

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
