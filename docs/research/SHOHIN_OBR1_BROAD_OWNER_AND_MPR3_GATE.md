# OBR1 Broad Owner and Conditional MPR3 Temporal Revision

Status: frozen prospective development contract, 2026-08-09. No holdout or
larger-MoE result is authorized by this document.

## Decision boundary

MPR1 and MPR2 established a useful but insufficient boundary on pinned
OLMoE. A simple all-layer shared post-MoE residual improves source-only
performance from 191 to 247, but aligned model-owned drafts did not beat
shuffled or source-only controls. DPR1 then showed that eight stochastic
trajectories contain a large whole-panel oracle on math and logic/science but
miss the code floor. The next experiment must improve the first-pass owner's
actual competence before asking a second pass to exploit its draft.

This is not another residual-width, seed, or short-duration retry. OBR1 changes
the training-data scale and quality while preserving the strongest simple
OLMoE intervention. MPR3 is conditional on OBR1 qualification and returns to
the exact aligned-versus-shuffled/source-only causal question.

## Pinned host and owner geometry

- host: `allenai/OLMoE-1B-7B-0125-Instruct` at
  `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`;
- frozen embeddings, attention, routers, experts, and language head;
- one rank-18 shared residual after each of all 16 sparse blocks;
- exactly 1,179,648 trainable parameters and 73,728 adapter MAC/token/layer;
- no external model, verifier, router, or tool at inference;
- context 4,096, complete-source/target admission only;
- 2,048 AdamW updates, LR `2e-5`, batch one, accumulation eight, seed
  `2026080911`, data seed `2026080912`.

## OBR1 data

The source is the existing verified 16M-target-token broad reasoning mix:
approximately 42% math, 16% code, 27% science, 13% procedural, and 2% teacher
by charged target tokens. It is rebound through the pinned OLMoE tokenizer.
Every row is wrapped in the established revision envelope with the draft span
causally unavailable to the full model, making OBR1 a direct source-only
owner rather than a temporal-revision treatment.

Before training, the builder must:

1. remove exact normalized collisions with all 1,289 development sources;
2. remove any shared development-unique normalized word 13-gram with a
   development source; n-grams repeated across development sources are
   benchmark-format boilerplate and are counted but excluded from the
   content-collision set;
3. deduplicate source questions;
4. reject any row whose complete OLMoE-tokenized source, envelope, and target
   exceed 4,096 tokens;
5. bind the input reports, tokenizer root, development split, outputs, and all
   counts/hashes; and
6. never read or use holdout.

After decontamination, at least 40,000 rows and 12M charged target tokens must
remain. This floor prevents a quality filter from being weakened merely to
retain a nominal fraction of the original corpus.

## OBR1 qualification gate

On the unchanged 1,289-row development board, OBR1 qualifies only if every
condition passes:

1. overall at least `300/1,289`;
2. MATH at least `75`;
3. logic/science at least `215`;
4. executable code at least `10/29`;
5. zero development overlap, zero protected trainables, zero sequence
   overflow, and complete data/model/runtime/update/token/FLOP receipts.

One update must pass mechanics first. A qualification miss closes exact OBR1
without width, seed, duration, LR, prompt, or threshold rescue and blocks
MPR3. A pass freezes its exact update-2,048 checkpoint as the first-pass draft
owner and authorizes MPR3 development only.

## Conditional MPR3 causal gate

MPR3 uses the qualified OBR1 owner to generate exactly one greedy model-owned
draft for every unique training and development source. A fresh residual with
the identical all-16-layer rank-18 geometry trains for 256 updates in three
matched arms:

1. **Aligned:** source plus its exact OBR1-owned draft.
2. **Shuffled:** source plus a different same-task, nearest-token-length
   OBR1-owned draft.
3. **Hidden/source-only:** exact aligned IDs and positions, but every draft key
   is causally unavailable throughout the model.

The fourth comparator is the direct OBR1 owner. Source, target, updates,
parameters, prompts, decoding, evaluator, and charged target tokens are exact
matches across trained arms. MPR3 passes development only if:

1. aligned is at least 39 answers (+3.0 points) above direct OBR1;
2. aligned is at least 13 answers (+1.0 point) above shuffled;
3. aligned is at least 13 answers above hidden/source-only;
4. aligned does not regress in MATH, logic/science, or executable code versus
   direct OBR1;
5. conservative possible-semantic repairs minus strict breaks versus direct
   OBR1 are at least 13; and
6. all ownership, source/draft/target retention, protected-hash, parameter,
   update, token, FLOP, memory, latency, and generated-token receipts are
   complete.

Only a conjunctive development pass opens one sealed holdout with the same
margins and nonnegative domain deltas. A development-plus-holdout pass makes
Shohin temporal revision operational on small OLMoE and authorizes staged
transfer to `Qwen3.6-35B-A3B`, followed by one different-family MoE. A miss
closes this practical small-OLMoE temporal route; no nearby MPR4 is allowed.

## Interpretation

- OBR1 fails: the small host/owner remains the capability bottleneck.
- OBR1 passes but MPR3 fails: broad post-MoE adaptation is useful, but the
  model-owned draft has not earned a causal temporal claim.
- MPR3 passes both development and holdout: temporal revision is causally
  useful on a MoE; scale by MoE size while retaining the aligned/shuffled
  control and full compute accounting.
