# Deferred Whole-Presentation Closure

Status: strong completion gain but causal gate failed; closed before
confirmation.

## Hypothesis

The learned PSPA pilot exposed an optimization conflict. A row-local compiler
learns every observed generator row, while end-to-end Sinkhorn projection
prevents the same encoder from learning even those source facts. Global
consistency is useful at execution but harmful during early evidence fitting.

Deferred Whole-Presentation Closure (DWPC) separates those phases:

1. learn source evidence with independently normalized generator rows;
2. preserve the resulting local uncertainty through compilation;
3. at the source/query boundary, project every complete generator to one
   indivisible permutation; and
4. execute the late query only through that committed presentation.

No weights, data, source loss, or query supervision change. The only changed
factor is when algebraic closure is imposed. The forward commit selects whole
generator actions; it never averages fields from incompatible presentations.

This resembles deferred projection, proximal commitment, and
neural-symbolic execution. Novelty is not claimed. The potentially useful
architectural principle is **plastic local evidence first, discrete global
closure once**, instead of differentiating through the global closure during
every update.

## Evidence and controls

The frozen source is the failed seed-47 learned-PSPA checkpoint. Its matched
row-soft compiler reaches 100% accuracy on observed rows but violates global
permutation closure. The jointly Sinkhorn-trained compiler remains near the
uniform floor and is the negative control.

The exploratory 128-row-per-cohort read-only diagnostic gives DWPC answer
exactness of `52.3/49.2%` cyclic, `75.0/75.0%` dihedral, and `54.7/43.0%`
random permutation at word lengths 8/12. This is roughly 58.2% macro, versus
25.8% for unclosed row-soft execution and 9.15% for joint Sinkhorn training.

The complete development evaluation uses all 1,024 rows in each frozen
cohort and must report:

- DWPC, row-soft, and jointly projected answers;
- source-challenge exactness;
- complete-table exactness;
- shuffled-challenge outcomes; and
- whole-presentation lineage swaps.

The development signal authorizes one unchanged-weight confirmation only if
DWPC beats row-soft by at least 10 macro points, gains on every family, and
both interventions cost at least five points. Confirmation uses new episode
and renderer seeds and requires at least 50% OOD macro, positive gain on every
family, and at least ten-point losses under both interventions. Failure closes
DWPC without retraining this checkpoint.

## Resource envelope

Development and confirmation are evaluation-only. Each is one single-H100
job with a 20-minute ceiling and expected use below 0.1 H100-hour. No training
or long pretraining follows unless unchanged weights pass confirmation.

## Full development result

Evaluation job `739360` uses the unchanged failed learned-PSPA checkpoint and
all 1,024 examples in each of the six frozen OOD cohorts.

| Arm or diagnostic | Six-cohort OOD macro exact |
|---|---:|
| DWPC | **58.643%** |
| Unclosed ROW_SOFT | 25.798% |
| Jointly projected PRESENTED | 9.147% |
| DWPC with shuffled challenge outcomes | 58.561% |
| DWPC with whole-presentation lineage swap | 11.865% |
| DWPC source-challenge exact | 71.407% |
| DWPC complete-table exact | 40.218% |

DWPC gains 32.845 points over its own unclosed checkpoint and improves every
family. Cyclic length-8/12 is `55.273% / 49.414%`, dihedral is
`75.977% / 76.562%`, and random permutation is `52.637% / 41.992%`.
Whole-lineage swapping removes 46.777 points, proving that coherent committed
tables cause the answers.

The challenge intervention is decisive in the opposite direction: exchanging
all source-challenge outcomes changes macro by only 0.081 points. DWPC is not
using counterexamples. It parses observed rows, closes each permutation, and
guesses the unresolved completion. The approximately 50% cyclic table score
and 13--14% random three-generator table score match that interpretation.

This misses the fixed causal rule, so confirmation remains unopened and DWPC
closes without retraining. The reusable result is the phase separation:
evidence-first local learning followed by one whole-object commit is far more
trainable than differentiating through closure. The successor must add
explicit counterexample-conditioned whole-presentation selection at that
single commit, not another continuous projection.

Report SHA-256 is
`17e4e0da326bf659d89a80804caf3793ac285c064acc7bf671070f3954fe31d2`.
