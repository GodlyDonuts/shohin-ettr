# Counterexample-Selected Deferred Closure

Status: development and unchanged-weight confirmation pass; typed-source
reasoning baseline established.

## Hypothesis

DWPC proves that local source facts can be learned and that a single
whole-presentation commit makes long composition work. It fails because the
commit greedily chooses one completion without using source counterexamples.

Counterexample-Selected Deferred Closure (CSDC) preserves the successful phase
separation and changes only the commit:

1. train row-local source evidence without global projection;
2. identify each generator's two least-certain rows;
3. construct the two whole permutation completions for each generator;
4. form at most eight complete multi-generator presentations;
5. execute every source-owned challenge against every presentation;
6. select the one complete presentation with the fewest contradictions; and
7. execute the late query through that selected lineage only.

No query, answer, hidden target, external model, or tool participates in
candidate construction or selection. The frozen seed-47 ROW_SOFT compiler is
unchanged. This gate isolates whether explicit source-counterexample
falsification closes the final ambiguity that DWPC guesses.

The first gate consumes typed challenge fields. It is not yet a free-form
language parser claim. The rendered-language compiler has already shown 100%
observed-row parsing; a later bridge must recover typed challenge fields from
ordinary text. This test focuses on the reasoning operator.

## Fixed controls and gate

- `CSDC`: confidence-derived whole candidates plus challenge selection;
- `DWPC`: one greedy whole closure with no challenge selection;
- `ROW_SOFT`: no closure;
- `SHUFFLED_CHALLENGE`: CSDC with outcomes exchanged across episodes; and
- `LINEAGE_SWAP`: CSDC with selected complete presentations exchanged before
  the late query.

Development uses all six existing 1,024-row OOD cohorts. CSDC advances only if
it reaches at least 90% macro exact, improves every family by at least 20
points over DWPC, recovers at least 95% of complete presentations and source
challenges, and loses at least 20 points under each intervention. A passing
development result authorizes one unchanged-weight confirmation on new
episode and renderer seeds, requiring at least 85% macro and the same causal
directions. Failure closes this candidate constructor without K, confidence,
scoring, or duration tuning.

## Resource envelope

This is evaluation-only. Development and any confirmation each use one
single-H100 job with a 20-minute ceiling and expected use below 0.1 H100-hour.
No training or Shohin integration follows from an incomplete causal pass.

## Development result

Job `739370` evaluates all six frozen 1,024-row OOD cohorts.

| Arm or diagnostic | Six-cohort OOD macro exact |
|---|---:|
| CSDC | **99.577%** |
| DWPC | 58.643% |
| ROW_SOFT | 25.798% |
| CSDC with shuffled challenge outcomes | 53.418% |
| CSDC with whole-lineage swap | 13.216% |
| Source-challenge exact | **99.908%** |
| Complete selected-table exact | **99.072%** |
| Candidate set contains true table | **99.707%** |

CSDC is 100% exact on both cyclic cohorts, `99.609% / 99.023%` on
dihedral length 8/12, and `99.316% / 99.512%` on random permutation length
8/12. It clears every fixed threshold. Report SHA-256 is
`45fd76fcc7702e19a79799c939e318db44a5bb4df84e28861f562ceed17101f4`.

## Unchanged-weight confirmation

Job `739373` uses new episode seeds (`82000+`) and renderer seeds (`83000+`)
without changing the checkpoint, candidate constructor, thresholds, or
selection rule.

| Arm or diagnostic | Confirmation macro exact |
|---|---:|
| CSDC | **99.723%** |
| DWPC | 58.577% |
| ROW_SOFT | 26.383% |
| CSDC with shuffled challenge outcomes | 52.376% |
| CSDC with whole-lineage swap | 13.916% |
| Source-challenge exact | **99.943%** |
| Complete selected-table exact | **99.284%** |
| Candidate set contains true table | **99.805%** |

Every confirmation family and depth is at least 99.316% exact. Shuffled
challenges remove 47.347 points and lineage swaps remove 85.807 points. The
effect therefore replicates and depends on both counterexample outcomes and
one coherent selected lineage. Report SHA-256 is
`c0c565b3f35d7fde306aad955d15891a0e324ada55926cb39db7cb122f798d3a`.

## Claim boundary and next gate

CSDC is a genuine synthetic causal reasoning result: a learned local compiler
constructs whole executable hypotheses, source counterexamples select among
them, and unseen long queries execute through the selected lineage. It is not
yet general natural-language reasoning. The challenge start, generator word,
and outcome enter the selector as typed source fields; only the observation
table compiler is learned from rendered records.

The next gate must learn the rendered-language-to-typed-challenge bridge while
freezing this selector and executor as the protected oracle ceiling. It must
stay within five points of CSDC, preserve both causal intervention effects,
and transfer to unseen renderer templates. Failure of the bridge does not
invalidate the reasoning operator; it localizes the next problem to semantic
compilation.
