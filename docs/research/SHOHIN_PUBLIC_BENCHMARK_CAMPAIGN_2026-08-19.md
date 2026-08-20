# Shohin public benchmark campaign — 2026-08-19

Status: **running; no result or website placement is authorized yet.**

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

## Immutable progress snapshot — 2026-08-20 17:23 EDT

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

Every report binds Qwen revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, model-tree SHA-256
`37048cc496c8992ea778fc1395f10b3c1d2dcb434f5de066f9f5c4bbf832903a`,
draft checkpoint SHA-256
`854a7cc44fbc2b54418f4e5bd09b7efeed0da44fc9ce217b0bb6b1997b722971`,
and revision checkpoint SHA-256
`df3c264d426941fef8ba9c10a90fe9fab304ec2864738209a4d79f9f81e0c473`.

LiveBench has completed all 1,000 `direct_base` rows and is generating the
matched draft arm; 121 rows were durable at this snapshot.  This is progress
evidence, not a score.  The isolated official-scoring queue is already waiting
for the terminal MMLU-Pro generation report and therefore cannot race or score
partial ledgers.  Its source SHA-256 is
`e441242d7248aab29251b66a820c8c0a361d5afcebd650c5809bc70ee80c54e9`.
No official score or aggregate result exists yet.

## Publication gate

The website graph may be populated only after:

1. all direct, unchanged-control, and trained-revision ledgers have exact row
   coverage;
2. every official scorer completes without unresolved scorer errors;
3. the aggregate report records per-benchmark direct/control/revision scores,
   paired deltas, wins, losses, retention, strata, and hashes;
4. graph labels preserve the RULER and LongBench Pro scope limitations.

Until then, the graph remains an explicitly pending measurement surface.
