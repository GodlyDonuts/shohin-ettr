# DIVERGE-AQC1: Antisymmetric Quotient Commit

Status: frozen before data materialization or CUDA results on 2026-08-08.

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
