# DIVERGE-VCR1: Verifier-Supervised Counterexample Revision

Status: frozen before any VCR1 model score on 2026-08-08.

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
