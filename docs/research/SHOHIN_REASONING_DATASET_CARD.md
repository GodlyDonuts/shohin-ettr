---
pretty_name: Shohin ETTR Reasoning Data
---

# Shohin ETTR Reasoning Data

Private, hash-bound product-reasoning training artifacts for Shohin research.
Every data file is paired with a machine-readable report. A file's presence in
this repository records storage and provenance; it does not by itself promote
the file into a final training recipe.

## Current artifacts

| File | Rows | SHA-256 | Status |
|---|---:|---|---|
| `data/v8_balanced_35m20p20c25t_unique_r1.jsonl` | 36,250 | `aebf832278b8b0792cdde423b87f187808b918f5b6dc84631fde81e63a0b7fee` | Controlled balanced-data diagnostic |
| `data/openscience2_expected_verified_10k_r1.jsonl` | 10,000 | `eaca4020fc5dceab1cff41d5bae94e5308949773ee262a9153ee767deec89173` | Verified-science pilot |

The balanced V8 diagnostic contains 12,688 math, 7,250 code, 7,250
procedural, and 9,062 teacher rows. It has zero duplicate questions and zero
replay against the local evaluation inventory. It intentionally raises code
exposure from the original V8 pilot's roughly 1% to 20%, but its teacher rows
are mostly synthetic HY3 traces and its code mix is not yet a final broad
execution-verified recipe.

The OpenScience pilot is a deterministic no-replay subset of a 500,000-row
answer-matched build. The full build has SHA-256
`e11e1923d237e1986725a7148503219e8871523649072cb38c835176854a5caa`.
Selection required the generated final answer to match the published expected
answer, unique normalized prompts, general quality checks, and no exact or
13-gram overlap with the local benchmark inventory. Expected-answer agreement
is a strong filter, but it is not a proof that every intermediate rationale is
scientifically flawless.

## Provenance and license

The science rows derive from
`nvidia/OpenScienceReasoning-2@174b02c9cdf231f220765b2a1d5ece4550921894`,
licensed CC-BY-4.0. See the upstream
[dataset card](https://huggingface.co/datasets/nvidia/OpenScienceReasoning-2/blob/main/README.md).
The exact attribution record is stored beside the selection report.

The balanced V8 artifact derives from Shohin's internal frozen reasoning-v8
candidate. Its report records the exact source hash, weights, group counts,
seed, duplicate count, and replay count.

## Admission policy

Before a stored artifact enters a promoted training mix, the campaign must
record its exact hash, source revision and license, domain counts, duplicate
and benchmark-overlap audit, verification method, and known limitations.
Executable code requires test execution. Math and science require answer or
solver verification. Live teacher-writer files are never direct optimizer
inputs.
