# DIVERGE-AQC1: Antisymmetric Quotient Commit

Status: complete on 2026-08-08. Both learned commit policies pass the frozen
capability floors. The antisymmetric treatment wins the practical comparison
by one answer but fails the required five-answer causal margin over its
matched independent-score control.

## Capability hypothesis

IDR1 and its untrained same-family continuation solve complementary cases on
source-disjoint holdout. IDR1 solves `625/1,279`; the continuation solves
`495`; their coherent whole-answer union is `671`. A fixed classifier trained
only on development metadata reaches `645`, proving at least twenty answers of
deployable selection headroom without holdout fitting.

Prior CVG1 scored candidates independently and lost to its strongest lineage.
Prior PCJ1 compared a joint ordered sequence and was only about 60% invariant
to swapping A/B. AQC1 tests a different interface: shared candidate encodings
feed an antisymmetric relational margin. For candidate representations `a,b`,

`margin(a,b) = phi(a,b) - phi(b,a)`.

Swapping candidates therefore negates the margin exactly. The selected object
is always one complete trajectory; fields are never averaged.

## Frozen data

- Candidate 0: exact trained IDR1 source-disjoint output.
- Candidate 1: exact original-9B-B1 `source + internal draft` control output.
- Source pool: the 1,289-row IDR1 development split and untouched 1,279-row
  IDR1 holdout split.
- Seed `2026080820` partitions development identities 80/20 into AQC1 train
  and development. Original holdout remains entirely holdout.
- Both candidate orders are presented during training. Candidate provenance,
  correctness, answer labels, and task labels are supervisor-only.
- Runtime input is only the problem and two complete candidate trajectories.

## Frozen arms

1. **AQC1 treatment:** shared 9B B1 plus trainable LoRA, shared candidate
   encoder, and antisymmetric relational head.
2. **Independent-score control:** same host, candidate encoder, data, update
   count, optimizer, seed family, and selection rule, but each candidate gets
   an independent scalar correctness score and the larger score wins.
3. **Always-IDR control:** `625/1,279` holdout.
4. **Development-trained metadata control:** `645/1,279` holdout. This is a
   diagnostic control, not the model-owned architectural result.
5. **Whole-answer oracle ceiling:** `671/1,279`; not deployable.

Both learned arms use 128 updates, gradient accumulation 8, maximum sequence
length 3,072, head width 512, backbone LR `2e-6`, head LR `2e-4`, AdamW,
and fixed seeds `2026080821` treatment / `2026080822` control. Training is
balanced across task and candidate-outcome class. Decisive pairs receive
pairwise logistic loss; ties receive a zero-margin calibration loss.

## Gates

The source-disjoint holdout gate is conjunctive:

- selected overall at least `646/1,279`;
- MATH at least `255/621`;
- logic at least `349/625`;
- code at least `24/33`;
- exact 100% A/B order consistency;
- control-only cases selected often enough to produce a net positive delta;
- treatment exceeds the matched independent-score control by at least five
  answers for the AQC mechanism claim.

If the independent scorer reaches the capability floors but AQC1 does not
beat it by five, preserve the scorer as the practical commit policy and reject
the relational novelty. If neither learned arm reaches 646, close this
same-family branch-arbitration direction without width, duration, seed, loss,
threshold, or renderer variants. Product evaluation remains sealed unless an
arm passes every capability floor. Prelaunch total estimate: `2--4`
H100-hours for both arms including evaluation.

## Claim boundary

A pass would establish useful model-owned coherent trajectory commitment on a
Qwen3.5-9B host. It would not establish native reasoning in the 125M Shohin
backbone, broad frontier competitiveness, or novelty of ensembling by itself.

## Result

Two-update integration jobs `745675/745676` exposed one order-dependent
selection bug: an exact zero margin selected the first candidate in either
ordering even though both heads satisfied `margin(a,b) = -margin(b,a)` with
zero numerical error. Commit `38bc1f3` replaces only that tie behavior with a
lexical, candidate-owned canonical tie rule. Five focused tests pass. The
scientific settings, data, model states, losses, seeds, thresholds, and
evaluator did not change.

Full jobs `745677/745678` then completed all 128 updates and 1,024 pair
presentations in `19m59s/19m58s`. Both protected B1 adapter hashes remain
bit-identical. Training and development have zero truncated prompts; holdout
contains one pre-existing truncated candidate prompt in each arm. Maximum
swap error is exactly zero and semantic order consistency is 100%.

| Holdout | Antisymmetric | Independent | IDR1 | Oracle |
|---|---:|---:|---:|---:|
| Overall | **652/1,279** | 651/1,279 | 625 | 671 |
| MATH | **272/621** | 271/621 | 248 | 280 |
| Logic | 354/625 | 354/625 | 351 | 365 |
| Code | 26/33 | 26/33 | 26 | 26 |

Both arms clear every capability floor. AQC1 treatment improves over IDR1 by
27 answers and the frozen metadata selector by seven. However, treatment
exceeds its matched control by only one answer, not the required five. The
correct conclusion is therefore:

`promote_the_652_answer_whole_trajectory_commit_as_the_practical_same_family_architecture;_reject_a_distinct_antisymmetric_relational_mechanism_claim;_do_not_run_nearby_aqc1_variants`.

Treatment/control/aggregate report SHA-256 values are
`9f72644c...5563`, `fdf9ead0...26b`, and `c56b0401...e74`. The two full
jobs consume `0.666` H100-hours; the two integration canaries consume another
`0.368` H100-hours.
