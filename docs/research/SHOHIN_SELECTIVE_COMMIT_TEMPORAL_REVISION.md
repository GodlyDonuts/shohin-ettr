# Selective Commit Temporal Revision (SCTR1)

Status: **closed negative**, 2026-08-08. Exact TTR1 is closed and is not
reopened by this mechanism. Holdout was not opened.

## Result

The full OLMo2-7B development gate completed over all 1,289 identities:

| Arm | Correct | Accuracy |
|---|---:|---:|
| Selective commit | 229 | 17.7657% |
| Unchanged selective second pass | 231 | 17.9209% |
| Standard always revise | 259 | 20.0931% |
| Long single generation | 190 | 14.7401% |
| Draft-masked independent selective | 215 | 16.6796% |
| Shuffled-command selective | 221 | 17.1451% |

Selective commit is `-0.1552` points versus unchanged, `-2.3274` points
versus always revise, and only `+0.6206` points versus shuffled supervision.
It emits eight malformed commands. Domain deltas versus unchanged are `+3`
MATH, `-9` logic/science, and `+4` MBPP, so all-domain nonregression also
fails. Comparison SHA-256 is
`ee9e2c551921d71971fbcb3d9d4660f11656710d3a13593aac1fdf556ba765cf`.

A read-only candidate contingency closes the obvious binary-selector followup.
Always-revise preserves 221 of 222 correct internal drafts and repairs 38
incorrect drafts; the exact union of original draft and always-revise output
contains 260 correct answers, only one above always-revise's 259. A perfect
selector can therefore add at most `1/1,289 = 0.0776` absolute points on this
board. The bottleneck is stronger revision/execution, not deciding whether to
keep the draft. Exact SCTR1 receives no command-token, classifier-head, seed,
rank, duration, or prompt rescue.

## Capability Hypothesis

TTR1 learned a strong trajectory-conditioned repair policy on SmolLM3-3B,
improving development accuracy from `27.7735%` to `36.3848%`, but it damaged
executable code (`9 -> 4`). Read-only attribution shows many failures contain
useful or correct Python wrapped in prose, Markdown, boxes, duplicated code,
or truncated imports. The bottleneck is therefore not only reasoning; forcing
every example through a newly serialized answer destroys capabilities already
present in a correct first trajectory.

SCTR1 changes the commitment mechanism, not the benchmark route. Given source
`x` and model-owned draft `d`, the same-family controller emits exactly one of:

```text
<KEEP>
<REVISE>
complete replacement trajectory
```

`<KEEP>` commits `d` byte-for-byte. `<REVISE>` commits one complete new
trajectory. Any other output fails closed. Incompatible fields or partial
answers are never averaged. At inference there is no verifier, task router,
answer label, teacher, or external proposal model.

## Training

The training supervisor evaluates only the model-owned draft:

- verified draft: target `<KEEP>`;
- incorrect draft: target `<REVISE>` plus the same verified complete target
  used by TTR1.

Source identities, train/development/holdout partition, draft generation,
revision prompt, tokenizer, decoding, and evaluator remain matched. The
commit decision and replacement trajectory are learned jointly by one
same-family role state. Existing TTR1 weights may be used only as an explicitly
reported warm start; a standard always-revise arm receives equal updates.

## First Discriminating Gate

SmolLM3 data may be used only for parser/data/mechanics verification because
its exact TTR1 development result already selected the hypothesis. The first
capability gate is on pinned `allenai/OLMo-2-1124-7B-Instruct` revision
`470b1fba1ae01581f270116362ee4aa1b97f4c84`, a larger non-Qwen family.
Gemma 3 12B was considered first, but an actual file request returned the
provider's gated-repository denial despite metadata visibility. This is an
access boundary, not an experimental result, and it did not influence model
scores.

Matched development arms are:

1. SCTR1 selective commitment;
2. standard always-revise TTR1;
3. unchanged second pass;
4. long single generation;
5. equal-update independent commitment;
6. shuffled KEEP/REVISE supervision as a causal control.

Every arm uses the same source identities, model, internal drafts, final
generation budget, evaluator, and reported token/FLOP accounting where the
mechanism permits. Development passes only if SCTR1:

- exceeds unchanged second pass by at least five absolute points overall;
- is at least as accurate as standard always-revise TTR1 overall;
- exceeds the strongest fully matched nontrained control by at least three
  absolute points overall;
- exceeds the deterministic shuffled-command control by at least three
  absolute points overall;
- has nonnegative MATH, logic/science, and executable-code correct-count
  deltas versus unchanged second pass;
- has complete coverage and no malformed commitment.

The source-disjoint holdout remains sealed until every development condition
passes. A failure closes exact SCTR1 without seed, rank, layer, duration,
prompt, parser, or threshold rescue.

### Frozen OLMo execution graph

Source-only drafting uses 17 independent single-H100 shards over all 8,392
training-bank identities. The resulting draft corpus is merged once, then the
same identities and partitions produce selective and always-revise datasets.
Four 256-update LoRA fits run at identical rank, layer count, learning rate,
batching, and data budget:

- selective commitment on the real command labels;
- selective commitment on a deterministic within-task/count-stratum command
  permutation;
- independent selective commitment on the real labels while attention to the
  complete internal-draft token span is zeroed without changing input IDs or
  positions;
- standard always-revise temporal training.

The independent arm intentionally retains the exact selective prompt and
`<KEEP>/<REVISE>` target space. Training it on a different always-revise
prompt would confound draft access with task format. Unchanged-second-pass and
long-single-generation controls use the unmodified OLMo backbone. Each of the
six arms is evaluated over eight batch-aligned shards and merged only after
exact identity/hash checks. Runtime `sctr1_1260cce_r6` and dispatcher `746411`
encode this graph; holdout is not part of the dispatcher.

## Evidence Boundary

The negative result shows that explicit generated KEEP/REVISE commitment is
not a reliable transferable improvement on this host. It does not negate the
standard trained-revision effect: always-revise still exceeds the unchanged
second pass by 28 answers. That effect is too small to promote this OLMo route,
and the SCTR1 holdout remains sealed.
