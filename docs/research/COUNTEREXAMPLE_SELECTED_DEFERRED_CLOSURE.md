# Counterexample-Selected Deferred Closure

Status: checkpoint-only gate frozen after DWPC's causal failure.

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

