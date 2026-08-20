# Shohin One-H100 Public Benchmark Campaign

Status: executing on Newton as allocation `766196`, beginning 2026-08-19. No
benchmark result exists until the corresponding generation ledger has complete
identity coverage and its official deterministic scorer closes.

## Question being measured

The campaign separates three comparisons that must not be conflated:

1. **Original model:** one-pass Qwen3.5-9B with no Shohin adapter.
2. **Equal-compute control:** the trained draft owner receives its own draft and
   performs an unchanged second pass.
3. **Shohin treatment:** the same draft, revision prompt, decoding, and token
   ceiling are passed to the trained revision owner.

The original arm supplies the hollow website point. The equal-compute control
isolates whether a gain comes from learned revision rather than merely spending
another generation pass. The Shohin treatment supplies the solid point. All
three use the same model-visible benchmark envelope; control and treatment use
byte-identical second-pass prompts.

## Frozen model custody

- Base: `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Base config SHA-256:
  `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05`
- Draft checkpoint SHA-256:
  `854a7cc44fbc2b54418f4e5bd09b7efeed0da44fc9ce217b0bb6b1997b722971`
- Revision checkpoint SHA-256:
  `df3c264d426941fef8ba9c10a90fe9fab304ec2864738209a4d79f9f81e0c473`
- Screen manifest SHA-256:
  `8499147647bd8fb128fb5700805ee4376f7ac3c9d1d58d1e33e4c2e7a7f9c2f5`

The base restoration has a complete immutable file manifest. Generation is
greedy. Every completion is appended and fsynced as one write-once JSONL row;
restart is allowed only from an exact identity-prefix ledger.

## Website benchmark coverage

| Website tab | Frozen public source | Primary publication metric | Admission note |
|---|---|---|---|
| MMLU-Pro | official 12,032-row test revision `b189ec76...f97be` | accuracy | 256-row source-disjoint qualification first |
| IFEval | Google Research `589e9774...935e` | strict prompt accuracy; instruction and loose metrics retained | 256-row qualification first |
| MuSR | official `b1f4d416...aa5` | accuracy, with all three domains retained | 256-row qualification first |
| LiveBench | official 2024-11-25 release, scorer source `50a270cf...2fa4` | official category-macro score | 1,000 rows |
| LiveCodeBench | release_v6 dataset `0fe84c39...d505`, scorer source `28fef95e...fa24` | full pass@1 | 1,055 rows; no lite substitution |
| HumanEval+ | dataset `d32357cf...321b`, EvalPlus v0.3.1 | pass@1 | 164 rows; sandbox qualification required |
| MBPP+ | dataset `b2d74c91...fa077`, EvalPlus v0.3.1 | pass@1 | current official 378-row release |
| RULER | NVIDIA source `c3f5e3b4...bf3a` | each context length and 13-task average | never collapsed across unrun lengths |
| LongBench Pro | dataset `4996884d...fa15` | official task-aware metric | 8k-64k subset can run; full 128k/256k is incompatible with the current repeated-source revision prompt and stays `N/A` |
| CorrectBench | dataset `4927112c...636a` | exact answer accuracy and correction lift | 739 rows |

LongBench Pro `N/A` at 128k/256k is an architecture/context result, not a zero.
Likewise, any RULER context length not fully executed remains absent rather than
being imputed from shorter contexts.

## Standardized overall score

Raw scores with different units are never pooled by item count. Once complete,
the website's default value is the unweighted macro-average of each admitted
benchmark's official primary percentage. The graph also preserves:

- original one-pass score;
- equal-compute unchanged score;
- trained-revision score;
- Shohin minus original, in percentage points;
- Shohin minus equal-compute control, in percentage points;
- baseline-correct retention;
- paired wins, losses, and two-sided sign-test value.

Screens, subsets, and context-specific runs are labeled as such and are excluded
from a "full benchmark" macro-average until their complete official contracts
finish.
