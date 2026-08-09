# Selective Commit Temporal Revision (SCTR1)

Status: frozen successor contract, 2026-08-08. Exact TTR1 is closed and is
not reopened by this mechanism.

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

A pass would show that model-owned temporal revision can preserve a complete
correct trajectory while selectively replacing an incorrect one across a
new model family. It would not prove general reasoning, optimality, or
architecture transfer to MoE systems. Only after dense-family development and
holdout pass may the same contract advance to one small/medium open MoE.
