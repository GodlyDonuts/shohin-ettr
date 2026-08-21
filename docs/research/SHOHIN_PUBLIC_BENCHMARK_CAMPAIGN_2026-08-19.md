# Shohin public benchmark campaign — 2026-08-19

Status: **running; early official scores exist, but no aggregate or website
placement is authorized yet.**

## Purpose

Measure the Qwen3.5-9B Shohin revision architecture on every benchmark currently
listed on Project Shohin. The campaign separates three reportable arms:

- `direct_base`: one ordinary greedy model pass;
- `unchanged_continuation`: an equal-compute second pass using the unchanged
  draft adapter;
- `trained_revision`: the trained Shohin revision pass.

This distinguishes learned-revision gains from gains caused only by a second
generation pass or a larger visible token envelope. The trained draft is also
generated for custody but is not a primary graph point.

## Compute custody

- Newton allocation: Slurm job `766196`
- resource boundary: exactly one H100, requested once
- wall-time ceiling: seven days
- campaign artifact root:
  `/lustre/fs1/home/sa305415/shohin/artifacts/public_bench_qwen9_766196_r1`
- generation is serial at the benchmark level; no second GPU campaign is
  authorized.

## Frozen board

The combined manifest contains 20,265 prompts across ten benchmark views:

| Benchmark | Rows | Scope |
|---|---:|---|
| HumanEval+ | 164 | complete official board |
| MBPP+ | 378 | complete official board |
| IFEval | 541 | complete official board |
| MuSR | 756 | complete official board |
| CorrectBench | 739 | complete official board |
| LiveBench | 1,000 | objective 2024-11-25 release |
| LiveCodeBench | 1,055 | release-v6 cumulative board |
| RULER | 2,600 | 13 tasks × 50 rows × 4k/8k/16k/32k |
| LongBench Pro | 1,000 | official non-thinking 8k–64k subset |
| MMLU-Pro | 12,032 | complete official test board |

Combined manifest SHA-256:
`1deb2dc8a7a26e613832ada2bb267dda67588eaea7ecaa113c8e82ce03314b47`.

RULER is an official-generator screen rather than the benchmark's entire
possible context-length sweep. LongBench Pro excludes 128k and 256k because the
current repeated-source revision envelope does not admit those lengths. Both
limits must remain visible on the website.

## Pinned official evaluators

- EvalPlus: `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`
- LiveCodeBench: `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`
- LiveBench: `50a270cfd77b966753f57a91507e766c5a012fa4`
- NVIDIA RULER: `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`
- LongBench Pro repository: `a1d3fd6eab275981a75f03ebda2169b72b7b876e`
- LongBench Pro embedding model: Qwen3-Embedding-8B revision
  `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`

Generated-code benchmarks run in a network-isolated Bubblewrap environment.
The official score ledgers are identity-bound and feed one final unweighted
macro-average of the ten completed primary percentages.

## Immutable progress snapshot — 2026-08-20 18:42 EDT

The live allocation is still healthy on `evc32`: job `766196` is `RUNNING`,
has zero restarts, and has retained its single H100.  The runtime checkout is
clean at `021645159dab708ca77105374f027f1bd2a5774f`; the generation source is
SHA-256 `bbb551089cb21457e24655efad201da9bf58e64f080800526db48f9b0a179841`.

Five complete four-arm generation reports now have exact row coverage:

| Benchmark | Rows per arm | Generation-report SHA-256 |
|---|---:|---|
| CorrectBench | 739 | `a0975c28de2a820395a3d490d76b29ab19a9b4ff5a4f93a482a6b9b9b6ad81a9` |
| HumanEval+ | 164 | `84eb3a1148da128c8164165ffa35fde7dbf128d0e61272fe42b0fe8997259710` |
| IFEval | 541 | `65cef28476bc57f4cd59c73d933005fa52cbbb4302411a539be91ca046e9164f` |
| MBPP+ | 378 | `c1639761d7abc55fbf590f1827a8f36c715d151c28bc5f43313a35d001ca1975` |
| MuSR | 756 | `afba4a5e68da08b55ec3abaab5dfb6c6f506c4c52491f5d73b2f80550b97aabd` |

Together these reports contain 10,312 completed arm rows: 2,578 benchmark
identities under each of the four matched generation arms.

Every report binds Qwen revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, model-tree SHA-256
`37048cc496c8992ea778fc1395f10b3c1d2dcb434f5de066f9f5c4bbf832903a`,
draft checkpoint SHA-256
`854a7cc44fbc2b54418f4e5bd09b7efeed0da44fc9ce217b0bb6b1997b722971`,
and revision checkpoint SHA-256
`df3c264d426941fef8ba9c10a90fe9fab304ec2864738209a4d79f9f81e0c473`.

LiveBench has completed all 1,000 `direct_base` rows and is generating the
matched draft arm; 699 rows were durable at this snapshot.  This is progress
evidence, not a score.  The isolated official-scoring queue is already waiting
for the terminal MMLU-Pro generation report and therefore cannot race or score
partial ledgers.  Its source SHA-256 is
`e441242d7248aab29251b66a820c8c0a361d5afcebd650c5809bc70ee80c54e9`.
No official score or aggregate result exists yet.

## Early official score receipts — 2026-08-20 19:02 EDT

Independent CPU scoring was released only for generation reports that already
had complete four-arm row coverage.  It did not regenerate model output or use
another GPU.  Three ordinary exact-match scorers and both official EvalPlus
scorers have now completed:

| Benchmark | Rows | Direct base | Unchanged | Trained revision | Revision − unchanged | Revision − direct | Wins / losses | Baseline-correct retention | Report SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| IFEval | 541 | 81.8854 | 72.2736 | 75.0462 | **+2.7726** | -6.8392 | 27 / 12 | 96.9309% | `ebfa8fb9053dc8bc20f9abfae65685b5b2f759e8942a3196f652388e39619e99` |
| HumanEval+ | 164 | 68.9024 | 70.7317 | 73.1707 | **+2.4390** | **+4.2683** | 5 / 1 | 99.1379% | `b3ec70ace2b4441efa31c3a1c6a2e0ab97f84d5f2bee07d0703598e359258139` |
| MBPP+ | 378 | 5.5556 | 5.8201 | 6.0847 | **+0.2646** | **+0.5291** | 1 / 0 | 100.0000% | `3e37ef792dbbada48dad34b7d5540277031837f38b09c6e0d960e0fc6ecd3d4e` |
| MuSR | 756 | 78.0423 | 68.3862 | 44.8413 | -23.5450 | -33.2011 | 38 / 216 | 58.2205% | `50ca5f1bcb79fd7d2144f17345e1727959ee37fdd9be8a5e8768c2fb40a73e5d` |
| CorrectBench | 739 | 35.7240 | 32.3410 | 14.8850 | -17.4560 | -20.8390 | 0 / 129 | 46.0251% | `bc01e307e202494539470aafd2897ac87213debd1b283dc06eab125fb7dd5247` |

Every score report has exact outcome coverage and unique identities equal to
the stated row count.  IFEval's matched 27-versus-12 improvement has a
two-sided sign-test p-value of `0.0237027`; HumanEval+'s 5-versus-1 direction
has p-value `0.21875`, and MBPP+'s single discordant win has p-value `1.0`.
The three positive results therefore do not erase the large MuSR and
CorrectBench regressions.  They establish measured capability-specific gains,
not a broad benchmark win.

The scoring jobs were `767645` (IFEval), `767650` (HumanEval+), `767655`
(MBPP+), `767639` (MuSR), and `767640` (CorrectBench), all zero-restart CPU
allocations.  IFEval required one isolated dependency repair before replay: sealed
`absl-py==2.5.0`, wheel SHA-256
`0f17b89f2a4eaaedc4f28c622998aa690564b3012a396a4ffad0821007fe03ba`,
55-entry manifest SHA-256
`bffd39bba10e64bfeeee610c1f90d5bc66ed8b1271b6e8817ff52718299bb67e`,
and receipt SHA-256
`ae3dd6171a37a736c7c078d62befcd9cdc5a646a32dde44d72e5c8be18acd066`.

Two additional pre-score defects were found without producing scientific
reports: the Bubblewrap virtual-environment projection omitted the absolute
base interpreter, and MBPP assessor IDs lacked EvalPlus' canonical `Mbpp/`
namespace.  The fixes are pushed at commits `266546e5` and `1b5e86d3`; the
combined public-benchmark regression suite has 81 tests passing.  MBPP+ job
`767655` replayed from a fresh work root and completed in 232 seconds.  Its
three official score ledgers contain exactly 378 unique identities each; no
artifact from either pre-score failure was reused.

All five reports and their 15 identity ledgers are owner-nonwritable and bound
by Newton evidence receipt
`/lustre/fs1/home/sa305415/shohin/artifacts/public_bench_qwen9_766196_r1/evidence/early_official_scores_20260820_r1/receipt.json`,
SHA-256 `a9b64c5cbb51533fea3fe10f505fa9ba9bb1ec73bd295a686f88fabf786a2bf3`.
It covers 3,564,211 bytes and binds the zero-restart Slurm accounting snapshot
at SHA-256 `7405c1369cd84990bcae5bdc1afc74e37ed6cb25aed3c7378a5c096aca214287`.

### Measured revision-horizon boundary

The five complete ledgers isolate a sharper architecture diagnosis.  Every
unchanged/revision prompt hash matches, yet the revision median is only 26
characters on MuSR and 12 on CorrectBench, compared with 1,885.5 and 863 for
unchanged.  The trained arm changed 95.24% and 95.13% of rows on those two
negative boards.  Conversely, the three positive boards retained a median
revision-to-unchanged character ratio of `1.0` and changed only 48.24%, 11.59%,
and 14.29% of rows.  Revision also exhausted the token ceiling less often than
unchanged, so the short outputs are learned early stops rather than truncation.

The exact public-development diagnosis is recorded in
`SHOHIN_QWEN9_REVISION_HORIZON_BOUNDARY_20260820.json`.  Its prospective
hypothesis is a task-agnostic draft-relative reasoning-horizon identity prior:
preserve the existing learned residual, but prevent early-EOS collapse when a
model-owned draft contains a long reasoning trajectory.  This is preparation,
not a new result or authorized launch; it requires a new source-disjoint broad
development screen and untouched confirmation before any claim.

Generic analyzer job `767657` independently replayed all 2,578 matched rows
from immutable generation and official-score ledgers.  Its owner-nonwritable
report has SHA-256
`338ca55c0bd068a8ee3b57e28437d0d3b806f5f0001a87a2d976d0be98a82a93`;
the execution receipt has SHA-256
`576a2453c816a531b1b17f295a5db75f24d58d118f140229749f457eabbbde42`.
The analyzer is pushed at `89af615b` with 87 public-campaign tests passing and
is reusable without benchmark labels in model-visible data on every subsequent
dense or MoE host.

### Training-target provenance of the horizon collapse

The matched public outputs can now be connected to the exact immutable IDR1
curriculum rather than inferred from checkpoint behavior alone.  A standalone
auditor replayed all `9,655` training presentations from input SHA-256
`6df3204573ce807db1b5057bce709189366b6674e38e5224ee3d17a3e6f0ac6c`.
The resulting owner-nonwritable Newton report has SHA-256
`866e0904199b28e6f121d3d711f1fb93e3872904896a0a8ad56ddfff2e2d37e7`;
its exact repository mirror is
`SHOHIN_QWEN9_IDR1_TRAINING_TARGET_HORIZON_20260820.json`.

The curriculum contains `3,294` `source_verified_repair` presentations, all
from the `both_wrong` outcome class.  Their response median is only `11`
characters, `2,969` are under 20 characters, `3,214` are under 80, and
`3,245` are exactly a boxed-answer response.  None contains a `<think>` block.
Those targets follow a median 1,776-character model-owned draft, producing a
median response-to-draft character ratio of only `0.006103`.  By contrast, the
`5,108` `verified_candidate` presentations have a 706-character response
median, a 1,897-character draft median, and a `0.402486` median ratio; only 16
responses are under 20 characters.  The short policy covers 3,294 unique
sources once each, while 1,277 verified-candidate sources are each presented
four times.  This is a direct target-policy asymmetry, not a post-hoc label
inferred from public benchmark scores.

The geometry closely predicts the public failure rather than merely sharing a
qualitative label: MuSR's matched unchanged/revision medians are 1,885.5 and 26
characters (ratio `0.012293`), while the both-wrong training medians are 1,776
and 11 (ratio `0.006103`).  CorrectBench shows the same direction at 863 versus
12 characters.  This alignment remains curriculum-plus-behavior evidence, not
an intervention result, but it materially narrows the mechanism to learned
draft-relative early termination.

The auditor also replays the exact executed optimizer population, rather than
summarizing rows the run never reached.  The frozen revision report has
batch size one, eight-microstep accumulation, 256 updates, data seed
`2026080814`, and exactly `365,028` charged response tokens.  Replaying its
2,048 selected microsteps with tokenizer SHA-256
`5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`
reproduces that charged-token total exactly.  Of those microsteps, `702`
(`34.2773%`) are answer-only source repairs, despite contributing only `7,008`
tokens (`1.91985%`).  Because standard batch-one cross entropy averages within
each response before equal gradient-accumulation scaling, terminal EOS occupies
`13.0906%` of an average source-repair row loss versus `0.7662%` for a
verified-candidate row: a `17.086×` denser stop target.

This identifies a minimal prospective intervention.  Preserve every existing
answer token and the matched source-plus-draft prompt, but mask terminal EOS
loss only on the answer-only both-wrong repair presentations.  The control must
retain the original EOS labels under identical initialization, selected row
order, optimizer, updates, charged non-EOS tokens, and source-disjoint boards.
This is an EOS-debiased correction objective, not another teacher, synthetic
fault source, natural-draft retry, or selector.  It remains code/data
preparation until an independently frozen matched development gate authorizes
execution.

The evidence supports a narrower successor than a generic “reason longer” or
another selector.  It must alter the answer-only both-wrong target/EOS geometry
while keeping the correction residual and using only the model-visible draft
trajectory at inference.  It is distinct from all closed lanes: VFR1's teacher
could not generate reliable fault traces; CFR1's synthetic faults caused
overlong draft trust; NDR1's natural aligned drafts lost to shuffled drafts;
and TCS1 showed post-hoc selection was not the bottleneck.  No new fit is
claimed or launched by this diagnosis.  A prospective intervention still
requires a matched source-disjoint development screen before untouched
confirmation.

## Cross-family expansion staged — 2026-08-20 17:34 EDT

The second exact dense host is SmolLM3-3B at upstream revision
`a07cc9a04f16550a088caea529712d1d335b0ac1`.  Its surviving immutable
checkpoint pair was reverified before staging:

- normalized draft SHA-256
  `b260d1acb20931e53f9f380f67a9d6b3feab89ae26f79dabb874f991f9c10edb`;
- trained revision SHA-256
  `e2b7a1798aa9430e139118222d3e469de42dc8cfd9affc954819ab5b0db37691`.

To restore that independent model family without reducing the 128-GiB
durable reserve, the completed and exactly re-downloadable GPT-OSS-120B base
was reclaimed.  Its 63,747,048-KiB tree passed a fresh full manifest replay,
had zero active Slurm/process references, and was the only deleted target.
The GPT score, accounting, mechanics report/checkpoint, trained checkpoint,
overlay, and publication evidence remain present.  The preserved base
manifest/config/revision receipt is
`/lustre/fs1/home/sa305415/shohin/artifacts/q36_mtr_evidence/gpt-oss-120b-b5c939de-r1-base-reclamation/reclamation.json`,
SHA-256 `2de0df3ea5e2836f8c705be2e2ab19d66725ad0e855e99e626f641102c1b3623`.
Settled quota usage fell from 978,864,880 to 915,117,868 KiB.

CPU restore job `767443` completed in 77 seconds on `evc1`, exit `0:0`, with
zero restarts.  The sealed 12-file model root contains 6,167,865,576 bytes and
has tree SHA-256
`6badcd593aee3052e3d66afb315b979e2cc62c4a61f9cef31c07203912478a0f`.
Its manifest and restoration-receipt SHA-256 values are respectively
`e689bcce197b02c4d2e8b600696ec3137b1e1724104954cc1735d5d8848e6945`
and `4672fc549809d89f0489a5e82045d54d3b5580718dcf40631a31807fd7415c85`.
One-GPU allocation `767126` is already queued independently for this host;
at this snapshot Slurm estimates admission at 2026-08-21 03:46 EDT, subject
to the account CPU-minute limit and normal nonbinding scheduling changes.
The host-generic controller
claims its artifact root atomically, runs one resumable screen, then the same
ten frozen benchmark manifests with identical greedy arm contracts.  It does
not alter or duplicate the active Qwen campaign.

## Publication gate

The website graph may be populated only after:

1. all direct, unchanged-control, and trained-revision ledgers have exact row
   coverage;
2. every official scorer completes without unresolved scorer errors;
3. the aggregate report records per-benchmark direct/control/revision scores,
   paired deltas, wins, losses, retention, strata, and hashes;
4. graph labels preserve the RULER and LongBench Pro scope limitations.

Until then, the graph remains an explicitly pending measurement surface.
