# VTE1: Verified Transaction Equivalence Learning

Status: **closed negative** after the one frozen source-disjoint evaluation.
KCR1, NDR1, and every earlier edit lane remain closed. Broad development,
the draft-hidden control, and holdout were not opened.

Execution receipt: CPU job `750779` admitted
11,218 sources, 33,654 presentations, and 65,206 verified candidate
transactions with zero truncation (maximum 3,107/4,096 tokens). One-update
mechanics replay `750781` passed with finite loss/gradient, exact candidate
grouping, 2,704,896 trainables, and 35.74 GB peak memory. Fit `750791`
completed all 256 updates; dispatcher `750814`, shards `750841--750844`, and
merge `750845` completed the one immutable source-disjoint canary.

## Final result

Aligned VTE1 scores `1285/1566 = 82.0562%` semantically, below the immutable
KCR1 parent at `1294/1566 = 82.6309%`. Presentation scores are natural-owner
`379/522`, verified-CONTINUE `414/522`, and verified-KEEP `492/522`; only
`324/522` sources are correct in all three states. All 1,566 outputs are valid
transactions, but the model emits RESTART on every row. Exact execution falls
to `69/1566`, KEEP byte preservation is `0/692`, counterfactual action
consistency is zero, and 197 generations exhaust the 768-token budget.

The set-valued objective therefore removed canonical-label punishment but
collapsed to universal regeneration. It misses the semantic, per-state,
parent-margin, all-three, KEEP-preservation, and exhaustion gates. Exact VTE1
closes without any temperature, candidate-family, delimiter, rank, layer,
duration, seed, decoding, parser, or threshold retry. Exact hashes and gate
receipts are preserved in `SHOHIN_VTE1_RESULT.json`.

## Capability hypothesis

KCR1 imposed a unique latent program label even when several programs produce
the same correct final semantics. Its source-disjoint canary contains 244
semantically correct outputs with a noncanonical action; 210 occur on expected
RESTART rows. VTE1 tests one structurally different learning mechanism:
maximize probability mass over the complete equivalence class of coherent,
independently verified transactions, rather than cross-entropy to one
arbitrary KEEP/CONTINUE/RESTART serialization.

This is not a retrospective KCR1 pass. KCR1 remains failed. VTE1 changes the
training object from a labeled branch to a set-valued executable program and
gets one bounded pass.

## Immutable substrate

- Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`;
- KCR1 update-512 parent SHA-256
  `07e08abe2480782afc77e35031d23bea71a737d019f307066af2bde786dd2ebd`;
- final-four rank-8 LoRA, exactly 2,704,896 trainables;
- the same 11,218 admitted training sources and 522 source-disjoint canary
  sources;
- 4,096-token complete-sequence custody, no truncation;
- frozen deterministic transaction parser/executor and semantic assessors;
- no verifier, solver, gold label, hidden host repair, or task route at
  inference.

## Verified equivalence sets

For each source-local presentation, the builder reconstructs the visible draft
and independently verified final trajectory from the immutable KCR1 source
artifacts. It enumerates only transactions that the frozen executor and
assessor certify:

1. KEEP iff the visible draft is already the verified trajectory;
2. CONTINUE with the exact missing suffix iff the verified trajectory has the
   draft as a byte prefix;
3. RESTART with the complete verified trajectory for every presentation; and
4. for non-code wrong natural drafts only, CONTINUE with a fixed correction
   delimiter plus the complete verified trajectory, admitted only when the
   independent assessor confirms the executed result.

Duplicate byte-identical transactions collapse. Every retained source must
keep all three presentations, every equivalence set must be nonempty, every
candidate must parse and execute, and every candidate must independently score
correct. The runtime row exposes only the source/draft prompt plus its verified
transaction set during training. Assessor fields are never model inputs.

## Set-valued objective

For transaction `t`, `ell(t)` is KCR1's normalized loss: action-marker and
payload/EOS spans each receive one half of presentation mass when a payload
exists; KEEP averages marker and EOS. With frozen temperature `tau = 0.1`,

```text
L_eq(E) = -tau * log((1 / |E|) * sum_{t in E} exp(-ell(t) / tau)).
```

The model therefore needs to assign high probability to at least one complete
verified transaction, without being penalized for selecting a different
semantically equivalent lineage. There is no auxiliary action classifier.
The generated transaction is parsed and executed exactly once at inference.

VTE1 resets optimizer state from the immutable KCR1 parent and runs exactly
256 updates, batch 1, accumulation 8, LR `2e-5`, seed `2026081021`, and data
seed `2026081020`. One update must first prove finite gradients, candidate
grouping, charged-token accounting, and zero protected-weight mutation.

## Matched arms

1. aligned verified equivalence sets;
2. identical fit with the entire draft span causally hidden at every layer;
3. immutable KCR1 update-512 parent with no VTE training.

The hidden arm uses identical prompts, candidate sets, parameters, updates,
optimizer, and token/FLOP schedule. No action-label permutation is meaningful
once labels are quotiented by verified execution.

## Frozen source-disjoint gate

On the exact 1,566-row KCR1 canary, aligned must satisfy all conditions:

- at least `1,410/1,566 = 90.04%` executed semantic correctness;
- at least 85% semantic correctness on each of verified KEEP, verified
  CONTINUE, and natural-owner presentations;
- at least 99% valid transactions and 99% KEEP byte preservation;
- at least `418/522 = 80.08%` sources semantically correct in all three states;
- at least 78 answers over the immutable KCR1 parent and at least 13 over the
  matched draft-hidden fit;
- at most ten decode-limit exhaustions; and
- complete set-cardinality, parameter, token, FLOP, memory, latency, and hash
  receipts.

Only this conjunctive pass opens broad development. Broad development retains
KCR1's `603/1,289` total and `223/349/17` domain floors, requires at least 13
answers over the hidden arm and immutable KCR parent, at least 13 net repairs
on wrong exhausted drafts, and no more than 400 payload exhaustions. Only a
conjunctive broad pass opens one sealed holdout.

## Stop rule and claim boundary

Any CPU admission, mechanics, source-disjoint, or broad gate miss closes exact
VTE1 without temperature, candidate-family, delimiter, rank, layer, duration,
seed, decoding, parser, or threshold variants. A pass would establish that
set-valued executable supervision resolves KCR1's branch-label
nonidentifiability and improves model-owned temporal revision. It would not
prove optimal planning, unrestricted reasoning, or a native 125M mechanism.
