# Shohin Dense Public Benchmark Campaign

Status: prospective execution contract frozen on 2026-08-15. This campaign is
parallel to the active Q36/Super/Mixtral work and must not cancel, duplicate,
alter, or delay those jobs.

## Exact executable inventory

Five dense hosts were measured historically. Only three still have every exact
treatment byte required for inference. A missing checkpoint closes the current
execution path; it does not authorize retraining or substitution.

| Host | Exact base revision | Surviving architecture bytes | Current status |
|---|---|---|---|
| Qwen3.5-0.8B | historical pinned host | reviser `540771a3...b357`; required draft warm start absent | historical only, not executable |
| Qwen3.5-4B | historical pinned host | required adapter `f7354e6a...1feb` absent | historical only, not executable |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a` | draft `854a7cc4...2971`; revision `df3c264d...473`; commit `434d1ec0...aabc`; qualified release manifest `554e841f...e7b` | executable after exact base restoration |
| SmolLM3-3B | `HuggingFaceTB/SmolLM3-3B@a07cc9a04f16550a088caea529712d1d335b0ac1` | normalized draft `b260d1ac...0edb`; revision `e2b7a179...7691` | executable after exact base restoration |
| OLMo2-7B | `allenai/OLMo-2-1124-7B-Instruct@470b1fba1ae01581f270116362ee4aa1b97f4c84` | raw-base draft; always-revise checkpoint `24105cd0...a6a1` | executable negative control after exact base restoration |

Qwen9, SmolLM3, then OLMo2 execute one host at a time. Each base is restored
from its exact upstream revision, checked against its exact config hash and a
complete file manifest, and reclaimed only after every admitted shard, score,
and host analysis closes. At least 128 GiB and 150,000 Lustre inodes remain
reserved after projected restoration for active Super/Mixtral work and the
verified Ultra capsule.

## Matched architecture comparison

The primary comparison is trained temporal revision versus unchanged
continuation. Both arms receive the same model-owned internal draft, exact same
revision prompt bytes, exact same native chat envelope, greedy decoding, seed,
stop tokens, and second-pass token ceiling. The only changed factor is the
second-pass adapter. Qwen9 additionally reports its learned whole-trajectory
commit as a diagnostic; it is not called compute-matched because selection adds
inference work.

OLMo2 uses the raw base for the draft and unchanged continuation, but the model
visible envelope remains byte-identical to the revision arm. This preserves the
historical non-Qwen negative instead of silently omitting it.

## Prospective broad screen and gate

Each host receives three independently frozen 256-row screens, stratified by
official benchmark strata with seed `2026081517`:

- MMLU-Pro: official test revision
  `b189ec765aa7ed75c8acfea42df31fdae71f97be`, official five-shot CoT,
  deterministic decoding, 2,048 output tokens;
- IFEval: Google Research revision
  `589e977488f21a336a3d3da9b96da91ddbcf935e`, official strict and loose
  prompt/instruction metrics, 2,048 output tokens;
- MuSR: revision `b1f4d4168a9cfc6760e8b74d728e4516023dfaa5`, all three
  domains represented, official cot+ zero-shot prompt and terminal parser,
  2,400 output tokens.

Exact normalized model-visible prompts are checked against the frozen Shohin
MATH, logic/science, and MBPP training banks before publication. Model workers
receive question-only files; assessors stay CPU-only.

The predeclared 768-row promotion gate is conjunctive:

1. trained revision gains at least 16 answers overall (`>=2.0` points);
2. it retains at least 95% of unchanged-correct answers;
3. no benchmark loses more than two answers; and
4. at least two of three benchmarks have nonnegative paired deltas.

A miss stops promotion for that host but remains a measured result. A pass
authorizes full official confirmation; the screen is never described as a full
benchmark.

## Official benchmark inventory

| Benchmark | Frozen official source | Full contract | Current admission |
|---|---|---|---|
| MMLU-Pro | repo `f418b116...872`, data `b189ec76...f97be` | 12,032 test rows, official five-shot CoT and answer extraction | 256-row screen admitted; full only after gate |
| IFEval | `589e9774...935e` | 541 rows; strict/loose prompt and instruction metrics | 256-row screen admitted; full only after gate |
| MuSR | `b1f4d416...aa5` | 756 flattened questions, official three-domain cot+ scoring | 256-row screen admitted; full only after gate |
| LiveBench | `50a270cf...fa4` | fully public 2024-11-25 release, objective scorers, no LLM judge | inventoried; not in first screen |
| LiveCodeBench | `28fef95e...a24` | full `release_v6`, 1,055 problems, pass@1; lite must be labeled | inventoried; full cost deferred to promotion |
| HumanEval+ / MBPP+ | EvalPlus v0.3.1 `e5d0ed0...cb2`; pinned dataset revisions | official EvalPlus execution and pass@1 | inventoried; sandbox/package qualification required before execution |
| RULER | `c3f5e3b4...bf3a` | 13 tasks with per-length reporting | deferred; no collapsed score authorized |
| LongBench Pro | data `4996884d...fa15` | 1,500 bilingual, 8k-256k, three stochastic runs, 1k non-thinking output, middle truncation | OLMo incompatible; Smol only <=64k; Qwen mechanics unqualified, so unavailable now |
| CorrectBench | `HCR050806/CorrectBench@a5fc5aca...41cc` | self-correction benchmark | repository lacks a complete frozen public dataset/scorer path; unavailable now |

No unavailable benchmark may be replaced by a similarly named task. No lite or
screen run may be labeled full.

## Execution graph

Data are prepared once. For each executable host the graph is: one CPU exact
base restore, one independent single-H100 nonbenchmark mechanics job, 24
independent single-H100 benchmark shards (eight per benchmark), three CPU
official scorers, one CPU host analyzer, and one manifest-verifying base
reclaim. Host graphs are serialized only at the reclaim/next-restore boundary;
within a host all identity-disjoint shards can backfill concurrently.

The complete graph is 94 jobs: 75 exact single-H100 requests and 19 CPU jobs.
Every request has requeue disabled, `nice=10000`, and excludes
`evc26,evc29,evc31,evc32,evc38,evc46`. It requests no idle GPU without a bound
mechanics or benchmark shard. One report is permitted per identity and output
paths are write-once. Temporary staged model copies are node-local and removed
on exit. Restored base trees are the only durable redownloadable bytes reclaimed;
adapters, manifests, source exports, generation reports, scores, analyses, and
dispatch/accounting evidence are preserved.
