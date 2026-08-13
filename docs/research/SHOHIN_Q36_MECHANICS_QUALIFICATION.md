# Q36 mechanics-only qualification

Status: **PASS** on 2026-08-13; no scientific graph was launched.

The sole authorized H100 allocation, Slurm job `754602` on eligible node
`evc35`, completed in 497 seconds with exit `0:0` and zero restarts. The exact
Qwen3.6-35B-A3B mechanics passed: all 693 weights loaded; the aligned causal
draft intervention changed response state and native expert routing in all 40
layers; the draft-hidden control was invariant in state and routing; one
finite source-only update completed; every protected router/expert byte was
unchanged; checkpoint restore was exact; and sealed holdout, product, and
public access stayed zero. The immutable result is
`Q36_MTR_MECHANICS_QUALIFICATION_RESULT_20260813.json`.

Q36 reached two distinct no-score mechanics infrastructure failures before any
draft, capability evaluation, or assessor access. A future full 61-request
graph is therefore ineligible until the repaired mechanics path passes once on
the exact pinned Qwen3.6-35B-A3B host.

The qualification is intentionally separate from the scientific phase:

- one `normal`-partition H100 and the unchanged node-exclusion set;
- `--no-requeue`, one immutable output root, and no retry;
- the same exact 24 B1 source-only rows, NF4 host, seeds, update, restore,
  causal draft intervention, native-router receipt, and protected-parameter
  checks as the frozen Q36 mechanics gate;
- both terminal-null receipts must be byte-exact and must attest zero scoring,
  zero assessor reads, and no scientific gate entry;
- no assessor, source-disjoint development row, holdout, product board, or
  public board is an input;
- no dispatcher or submission call exists in the wrapper; and
- PASS and failure both stop for evidence preservation. Neither authorizes the
  scientific graph or a successor.

`pipeline/authorize_q36_mtr_mechanics_qualification.py` builds a write-once
authorization and independently verifies it inside the allocation before
model staging. `train/jobs/q36_mtr_mechanics_qualification.sbatch` is the sole
qualification wrapper. Its authorization explicitly records
`scientific_graph_authorized=false`, `capability_scoring_authorized=false`,
`assessor_access_authorized=false`, and `submission_capability=false`.

The authorization is published mode `0444`. Allocation-time verification
rehashes the exact runtime manifest, model manifest, environment receipt, B1
source, and both terminal-null receipts, and requires their resolved paths to
match the authorization. Any missing, writable, substituted, or changed
authorization—and any missing, substituted, or changed custody input—fails
before model staging.

This boundary proved the repaired causal-generation mechanics on the real host
without consuming or weakening the publication experiment. It changed no arm,
prompt, threshold, seed, source identity, training budget, or terminal rule.
