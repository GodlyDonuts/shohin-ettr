# DIVERGE-PCJ1: Pairwise Counterfactual Judge

Status: frozen before any PCJ1 model score on 2026-08-08.

## Question

CVG1 judged each complete solution independently and selected only 35.211% of
its holdout, below the stronger QPT1 lineage at 38.850%. Its coherent oracle
was 43.545%, so complementary answers existed but the independent scores did
not compare the competing explanations reliably.

PCJ1 asks the narrower, structurally different question: can a model that sees
the problem and both complete solutions jointly identify enough B1-only wins
without sacrificing QPT1 wins?

## Frozen Mechanism

The judge input contains only:

1. problem text;
2. complete Candidate A solution;
3. complete Candidate B solution.

It receives no task, benchmark, gold answer, correctness, execution result, or
lineage identity. A three-class head predicts `A better`, `tie`, or `B better`
from the final hidden state. Training presents deterministic identity-derived
A/B orders. Evaluation scores every example in both orders.

Verdicts are mapped back to semantic lineages. PCJ1 commits to B1 only when
both orders consistently select B1. Ties, inconsistent orders, and consistent
QPT1 verdicts all default to QPT1. There is no tunable threshold.

## Immutable Corpus And Split

The source is the completed CVG1 corpus:

- rows: `8,392`;
- pair SHA-256: `45f1d66ce5e87dc2a1f4c3594bdde2bae26e9417e879d16eb4eddb228b696afe`;
- source report SHA-256: `d38e67908ca3f19c81d08b1772c4fd2d0af6d1d113a3a82058f5bc9c5128e4fd`;
- content: `4,096` MATH-500-style, `4,096` BBH/science-logic, and `200`
  independently execution-verified MBPP-style examples.

PCJ1 uses a newly frozen identity partition with seed `2026080811`:

| Split | Rows | Identity-list SHA-256 |
|---|---:|---|
| Train | 5,869 | `22ba82442717368f8bc7c2c60d2651f24f6ed5e5b1ce4f94dae47e6a4056f6ef` |
| Development | 1,265 | `184659902f1dbd106dd80f9b3ea1aeccf2ed91d0c5aefc3d3d1a787dec7d87fb` |
| Holdout | 1,258 | `d722aed3c24531109d290e1b9d7e82bd2946a93aa2a63ef3bcc62d441a59d52b` |

Rule: `sha256(seed + NUL + identity)[:8] mod 10000`; buckets
`[0,7000)`, `[7000,8500)`, and `[8500,10000)` are train, development,
and holdout. All splits contain every task and outcome class.

The holdout has `86` B1-only, `219` QPT1-only, `287` both-correct, and
`666` both-wrong rows. Thus B1/QPT1/oracle solve `373/506/592` of `1,258`.

## Predeclared Arms

Two independent one-H100 arms use identical data, updates, labels, head, and
gate:

1. exact Qwen3.5-4B B1 host, the scale-matched model-owned control;
2. exact Qwen3.5-9B B1 host, a stronger contemporary judge and explicit
   system-scale control.

The candidate completions remain the same frozen 4B B1/QPT1 lineages in both
arms. A 9B pass is therefore a stronger-host system result, not a claim that
the 4B architecture alone acquired the capability.

Frozen optimization: `256` AdamW updates, `8` presentations/update, balanced
task/outcome strata, head width `512`, context `3,072`, LoRA LR `2e-6`, head
LR `2e-4`, seed `2026080812`, cosine decay, and no score-dependent retry.

## Holdout Gate

An arm passes only if all conditions hold on the new holdout:

- A/B semantic order consistency `>= 90%`;
- B1 commit rate in `[2%, 50%]`;
- disagreement selection accuracy `>= 80%`;
- selected exact accuracy at least `2` percentage points above QPT1.

The last condition requires at least `532/1,258` correct, compared with QPT1
at `506/1,258`. Failure closes that host arm without threshold, seed, width,
duration, renderer, or loss variants.

Only a passing holdout arm may score the already-preserved 568-example
development board. Its unchanged product gate requires code `>=30/40`, at
least `+3` five-domain macro points and `+15` solved answers over the strongest
single lineage, at least three improved domains, and no domain regression over
two points.

## Cost And Successor

Expected charge before launch is `2--4` H100-hours total for both fits and
their conditional applications. If both comparators fail, PCJ1 closes and the
next structural mechanism is verifier-grounded counterexample revision of one
whole lineage, not another classifier or threshold sweep.
