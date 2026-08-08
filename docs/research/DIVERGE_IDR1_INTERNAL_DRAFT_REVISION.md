# DIVERGE-IDR1: Internal Draft Revision

Status: exact training, source-disjoint evaluations, and the matched
no-revision attribution control are complete. Trained revision is causally
positive, the conjunctive promotion gate failed narrowly, and product remains
sealed.

## Hypothesis

VCR1 proves that candidate-conditioned revision can exceed its two source
attempts. SDR1 proves that the same verified targets without candidate
trajectories lose 153 holdout answers. IDR1 preserves temporal computation
while removing external proposal models: one pinned Qwen3.5-9B B1 owner first
generates a complete draft from the source, and a later 9B revision owner sees
`source + internal draft` and emits the final solution.

The model family owns both passes. No 4B B1, 4B QPT1, task router, gold answer,
correctness bit, evaluator feedback, or external tool is available at
inference. The proposal and revision owners share the exact pinned 9B
backbone, but use distinct small LoRA states. This is a deployable two-pass
reasoning architecture rather than source-only SFT.

## Frozen Draft Collection

The source bank is the exact 8,392-row CVG1 bank already partitioned by the
NUL-delimited source identity split. Draft generation uses:

- pinned `Qwen/Qwen3.5-9B@c202236...337b9a`;
- exact 9B B1 checkpoint SHA-256
  `854a7cc44fbc2b54418f4e5bd09b7efeed0da44fc9ce217b0bb6b1997b722971`;
- one greedy sample, no thinking mode, maximum 768 new tokens;
- reasoning prompt style, prompt batch 4, seed `2026080818`;
- eight 512-row math shards, eight 512-row science/logic shards, and one
  200-row execution-verified code shard;
- 17 independent one-H100 jobs with isolated atomic outputs.

Every shard must bind the immutable runtime, model, checkpoint, and source
bank. Merge must prove 8,392 unique identities, exact bank coverage, one draft
per identity, and unchanged source scoring. No draft score is a gate by itself;
drafts become model-visible training inputs only after the complete receipt
passes.

The prelaunch draft-generation estimate was `3--6` H100-hours. At 05:19 EDT,
early measured progress from 16 concurrent shards corrected that estimate to
`16--21` H100-hours. The jobs remain scientifically valid and already hold the
allocations, so they continue unchanged. The later revision fit and
source-disjoint evaluation are separately expected to consume `2--4`
H100-hours. Both the original estimate and correction are preserved.

## Draft And Data Result

All 17 jobs `745628--745644` completed for an exact charge of `17.837`
H100-hours. The complete source-owned first pass scored:

| Domain | Correct | Exhausted at 768 tokens |
|---|---:|---:|
| MATH | `728/4,096 = 17.77%` | `2,992/4,096` |
| Science/logic | `919/4,096 = 22.44%` | `2,949/4,096` |
| Execution-verified code | `137/200 = 68.50%` | `4/200` |
| Total | `1,784/8,392 = 21.26%` | `5,945/8,392` |

Merged drafts SHA-256 is
`509b114e42773dbe8d14536ff5ce7e5a2f92b45b3acd57a37a035fce43505d73`;
the complete receipt is
`2154a49811dece70f25ccb309c8ef1fdc08d4513880477b500a4cbfba156a852`.
The frozen revision data contain `9,655/1,289/1,279` train/development/holdout
rows. Their SHA-256 values are `6df32045...ac6c`, `0c52dd35...4224`, and
`df347d59...36f2`; report SHA-256 is `8e4b2817...c80e`.

CPU build `745645` failed before output on a CLI receipt keyword mismatch.
Replay `745648` completed and preserved the valid merge, then failed before
revision data on an omitted transitive runtime import. Data-only recovery
`745650` reused the hash-bound merge and completed. These were packaging
failures only: no scientific input or setting changed.

## Revision Training Result

Job `745652` completed all 256 frozen updates in 19m50s. It charged 365,028
target tokens at 320.55 target tokens/s. Checkpoint SHA-256 is
`df3c264d426941fef8ba9c10a90fe9fab304ec2864738209a4d79f9f81e0c473`.

## Source-Disjoint Gate Result

| Split | Overall | MATH | Logic | Code | Both-wrong repairs | Gate |
|---|---:|---:|---:|---:|---:|---|
| Development | `589/1,289` | `223/623` | `349/637` | `17/29` | `178/767` | FAIL |
| Holdout | `625/1,279` | `248/621` | `351/625` | `26/33` | `183/727` | FAIL |

Development missed the frozen MATH and code floors by one and two answers.
Holdout passed overall, logic, code, and both-wrong floors but missed the MATH
floor by seven. The exact gate is therefore closed and the product board
remains sealed. Development and holdout report SHA-256 values are
`0880e59c03460f3ab0f5c8da87136fd9f60961f3fa16a762225188ac998b7a40`
and `74834cad3ee4c32e1e263d968bbb2f5b1f4dfeb6eca91b124e1a4f5a03148b53`.

The result is nevertheless a large, independently reproduced capability gain.
On holdout, revision improves the internal first pass from `265` to `625`
correct, exceeds one-pass source-only SDR1 (`490`) and the 9B QPT1 expert
(`471`), and finishes 18 answers below externally proposed VCR1 (`643`). Of
363 cases repaired after an incorrect draft, 356 drafts had exhausted the
768-token budget and 327 contained no explicit candidate answer; only three
initially correct drafts were lost. This identifies a major learned
continuation/finalization effect, especially for long truncated trajectories.

## Matched Attribution Control Result

The matched control ran the same frozen internal drafts, source-plus-draft
prompt, evaluator, generation budget, and split using the original exact 9B B1
adapter rather than the trained IDR1 revision state. Slow long-form generation
made monolithic jobs `745657/745658` unable to fit their scheduler windows.
They were replaced by four contiguous batch-aligned shards per split,
`745659--745667`, with no model, prompt, ordering, batching, decoding, seed, or
scoring change. One failed CUDA-invisible allocation is replaced exactly.
Complete identity coverage was proven before merge and scoring. The result is:

| Split | Original 9B B1 second pass | Trained IDR1 | Delta |
|---|---:|---:|---:|
| Development | `464/1,289` | `589/1,289` | `+125` |
| Holdout | `495/1,279` | `625/1,279` | `+130` |

Holdout gains are MATH `+83` (`165->248`), logic `+46` (`305->351`), and
code `+1` (`25->26`). Development gains are `+73/+50/+2`. Trained IDR1 fixes
176 control errors while losing 46 control-correct answers on holdout. On the
914 token-exhausted drafts, control/trained score `286/402`; on the 365
non-exhausted drafts they score `209/223`. The result therefore proves a
learned revision and termination policy, not just the generic benefit of
providing a draft to an additional inference pass.

Development and holdout control reports hash to
`08f5e12f006426df2d9a9a71dc3a69a9a0825a508121309e1fa386054be60d1a`
and `37bb0c5300d5b8dec45a6e8b3a07b20b708a17cdff2ce5b22481c002702ccb9a`.
Valid sharded evaluation cost `10.220` H100-hours. Superseded monolithic and
failed/canceled `evc33` allocations add `1.104`, making the exact attribution
control cost `11.324` H100-hours. Including draft collection, training, and
the primary source gates, complete IDR1 cost is `30.124` H100-hours.

The attribution result cannot reopen the exact IDR1 gate or authorize the
sealed product board. It does establish IDR1 as the strongest measured
model-owned same-family architecture in this campaign.

## Frozen Revision Gate

After complete draft custody, IDR1 will use the exact SDR1/VCR1 verified
targets and split. The only new input is the frozen model-owned 9B draft.
Training geometry remains the 256-update pinned 9B B1 LoRA schedule. Holdout
must retain VCR1 within two percentage points (`>=618/1,279`), meet the same
fixed MATH/logic/code and both-wrong floors, and reproduce on development
before the product board opens.

Failure closes exact IDR1 without draft sampling, prompt, seed, LR, duration,
rank, layer, context, decoding, or threshold variants. The measured failure is
now final. The matched no-revision control is attribution, not a rescue.
