# DIVERGE-VCR1: Verifier-Supervised Counterexample Revision

Status: completed once on 2026-08-08. The source-disjoint revision gate passed
decisively; the conditional product gate failed one conjunct because generated
code was `29/40` against a fixed `30/40` floor. Exact VCR1 is closed without a
rescue variant.

## Hypothesis

PCJ1's 9B joint judge recovered 24 answers over QPT1 but missed its fixed
threshold by two answers and remained order-inconsistent. VCR1 does not add
another selector. It trains one model-owned revision policy to read the
problem plus both complete B1/QPT1 attempts and generate a new corrected
solution. This allows recovery when both source attempts are wrong, which the
whole-lineage oracle and every selector cannot do.

VCR1 is a practical multi-stage reasoning system, not yet a standalone Shohin
architecture claim: its two proposal lineages are exact 4B B1/QPT1 and its
revision host is exact pinned Qwen3.5-9B B1.

## Runtime Boundary

The model receives one string containing:

1. the complete original task, including code tests where applicable;
2. complete Candidate A and Candidate B attempts;
3. an instruction to solve and repair rather than classify;
4. the original task repeated at the end so bounded left truncation preserves
   the problem.

It receives no task/benchmark identifier, gold answer, correctness bit,
execution result, outcome class, or assessor field. Evaluation code passes
only the `question` runtime field to the model.

## Data

The immutable source remains the CVG1 corpus with SHA-256
`45f1d66ce5e87dc2a1f4c3594bdde2bae26e9417e879d16eb4eddb228b696afe`.
VCR1 uses the corrected NUL-delimited PCJ1 split and binds every identity to
the independently verified source bank.

Training targets are:

- the verified source candidate when exactly one lineage is correct;
- the shortest verified candidate when both are correct;
- the source-verified answer or execution-verified reference program when both
  are wrong.

Disagreement rows receive four deterministic A/B presentations; other rows
receive one. This produces 9,655 train presentations from 5,824 source pairs:
5,108 verified-candidate, 1,253 shortest-verified-candidate, and 3,294
source-repair targets. Development and holdout remain 1,289 and 1,279 unique
identities.

| Artifact | SHA-256 |
|---|---|
| Train | `55635fbeed1f342dca961827c184e08a4959390dc9889ac7b7ceb72433773d9d` |
| Development | `f0f982446ddb856346193dec7594043a48d459b921b04e89bddfdb87fbbb328e` |
| Holdout | `f135fcff4b4e0a7028448b1b386722950069cfc9e489526e2bd3ec297d0fd148` |
| Data report | `b6e4e94d7e8d2f0e8c98dd6cb6c687a1ffa5f0ae3d451b4b51447d58545953b3` |

## Frozen Fit And Evaluation

VCR1 warm-starts the exact Qwen3.5-9B B1 checkpoint. It trains the existing
four-layer rank-8 LoRA owner for exactly 256 AdamW updates, batch 1,
accumulation 8, context 4,096, LR `2e-5`, cosine decay, train seed
`2026080815`, and data seed `2026080814`. No backbone, head, or source
checkpoint is overwritten.

Development and holdout run as independent dependency-released one-H100 jobs
with greedy decoding, batch 2, maximum 768 new tokens, and seed `2026080816`.
Both use the same exact scorer as the source banks. Holdout passes only if all
conditions hold:

- generated accuracy beats QPT1 by at least two points;
- math and science/logic each regress by at most two points versus QPT1;
- code solves at least as many examples as QPT1;
- at least one both-wrong source pair is repaired.

On the 1,279 holdout rows, QPT1 solves 471. VCR1 therefore needs at least 497
correct. A holdout failure closes exact VCR1 without prompt, LR, duration,
seed, order, context, or decoding variants. Only concordant development and
holdout success may open one reconstruction and score of the preserved
568-example product board.

## Budget

Projected charge before launch is `3--6` H100-hours including training, both
source-disjoint evaluations, and a conditional product evaluation. Current
monthly balance before launch is approximately 1,591 H100-hours. No larger
campaign is authorized by this gate.

## Result

Training job `745618` completed all 256 updates in `00:21:34`. The final
checkpoint SHA-256 is
`28c23825dacec511c8e5a147446a3aad60bf3c2740421db3a7086a17c3bc2f7f`;
the training report SHA-256 is
`c5b79653dd4b9a6784c1c9455316f1d9cb0270592f3befb0689e7c0f05b0b9cf`.

The independent development and holdout jobs both pass every frozen gate:

| Split | B1 | QPT1 | Source oracle | VCR1 | Both-wrong repairs |
|---|---:|---:|---:|---:|---:|
| Development | 340/1,289 | 453/1,289 | 522/1,289 | **575/1,289** | 113 |
| Holdout | 344/1,279 | 471/1,279 | 552/1,279 | **643/1,279** | 153 |

Development and holdout result SHA-256 values are respectively
`4d04fbc64ad416627a629be6db2bdfa3347df38374c6f84baf47ac2bcf7ae6ad`
and
`942579931cc11b813fa7aab26ec7c61bed5521d51f8e5b9b5551a22d5957dce7`.
The holdout result is not selection: VCR1 exceeds the two-candidate oracle by
91 answers because it generates corrected solutions for 153 cases where both
inputs were wrong.

Both passing splits release one unchanged reconstruction of the preserved
568-example product board. Product-data SHA-256 is
`c1ebc24bab12e79753797cd14a790d1c5c4f750348412749c9329d2b968f5c4f`;
its report SHA-256 is
`f655c9a20493d9f42e19a67880389b6edbbde31e5a2b7b63c48e2ab62eed7170`.
Job `745624` produces:

| Domain | B1 | QPT1 | VCR1 | Source oracle |
|---|---:|---:|---:|---:|
| GSM8K | 85/100 | 93/100 | **92/100** | 96/100 |
| MATH-500 | 50/100 | 54/100 | **68/100** | 64/100 |
| Executable code | 30/40 | 26/40 | **29/40** | 34/40 |
| GPQA | 30/198 | 87/198 | **101/198** | 97/198 |
| BBH logic | 53/100 | 57/100 | **78/100** | 75/100 |
| Five-domain macro | 55.630% | 62.588% | **72.302%** | 73.798% |
| Solved | 248/538 | 317/538 | **368/538** | 366/538 |
| AIME-2024 | 4/30 | 1/30 | **6/30** | 5/30 |

VCR1 improves the strongest single lineage by `+9.714` macro points and 51
main-board answers, improves four of five domains, stays within one point on
GSM8K, and generates 38 correct answers where both source attempts were
wrong. It also loses 35 answers that at least one source candidate solved.
Every product condition passes except `code_at_least_30_of_40`; therefore the
conjunctive product decision is **FAIL**, despite the large aggregate gain.
Product report SHA-256 is
`f7a6b8606505dfeb5e0821f04f56436c77ef9ce910847df472975d0f562fafbc`;
candidate SHA-256 is
`013375c8711df7e9e691830418e5239da06d56b82ee6c5a8a4fd3edb48e9587b`.

The four scientific GPU jobs consumed `1.040` H100-hours in total, below the
projected `3--6` hours. Exact VCR1 receives no prompt, seed, duration, context,
decoding, threshold, or code-routing rescue. Retain it as the strongest
measured practical reasoning system in this campaign, while keeping the claim
bounded: the runtime uses two 4B proposal lineages and a 9B reviser and is not
a standalone Shohin architecture result.
