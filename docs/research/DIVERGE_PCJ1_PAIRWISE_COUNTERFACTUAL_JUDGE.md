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
| Train | 5,824 | `c1aaa7f6372ad34cd41e36984dd715d899b877645422d639ce2a3b4118c46495` |
| Development | 1,289 | `dc634881ce38a9c3f37b37794e07dbfe3710ed19ea0ded07dd014c232334ce18` |
| Holdout | 1,279 | `51534f07ba70d8ffc0f42b6fab0d83770c0f7490cb3e535ebdc966cade921d35` |

Rule: `sha256(seed + NUL + identity)[:8] mod 10000`; buckets
`[0,7000)`, `[7000,8500)`, and `[8500,10000)` are train, development,
and holdout. All splits contain every task and outcome class.

The holdout has `81` B1-only, `208` QPT1-only, `263` both-correct, and
`727` both-wrong rows. Thus B1/QPT1/oracle solve `344/471/552` of `1,279`.

Receipt correction: the prelaunch audit command that printed the first table
escaped the delimiter as two literal characters (`\\0`) while the frozen rule,
implementation, and jobs used one NUL byte (`\0`). The wrong printed counts
were discovered only after immutable reports closed. The corrected table above
is reproduced independently by both reports. This is a custody-report error,
not a data, seed, model, threshold, or score change.

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

The last condition requires at least `497/1,279` correct, compared with QPT1
at `471/1,279`. Failure closes that host arm without threshold, seed, width,
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

## Result

Both arms complete all `256` updates with zero train/development/holdout
truncations and unchanged protected hosts. The conditional applications fail
closed before scoring the 568-row product board.

| Holdout metric | 4B judge | 9B judge |
|---|---:|---:|
| QPT1 | 471/1,279 (36.826%) | 471/1,279 (36.826%) |
| PCJ1 selected | 489/1,279 (38.233%) | 495/1,279 (38.702%) |
| Net answers | +18 | +24 |
| Disagreement selection | 78.201% | 80.277% |
| A/B consistency | 60.751% | 59.030% |
| B1 commit rate | 4.848% | 5.473% |
| Gate | FAIL | FAIL |

The 9B arm is a meaningful near-miss but remains below the fixed `497`
correct threshold by two answers and far below the 90% order-consistency
requirement. Development agrees: 4B/9B select `465/472` versus QPT1 `453`.
No public score is assigned.

Exact 4B/9B report SHA-256 values are
`5a02855ff52c70246e08d7cd5a100b510c1b34b2f11fd60e041058f5a13018eb`
and `6773fb0387c113e7e6a410d93381f1565ff3e1e612163b981c969096abea960e`.
Judge SHA-256 values are
`2c55960b769ab92f6013cfb33ad4b3d55bc1245b97a22e2846c17d64eaf6ad55`
and `28fa4728b4a141d3af28fb8648d9c84163e27c1337791a19a85a872fc102651c`.
Total charged wall time is `0.8717` H100-hours.
