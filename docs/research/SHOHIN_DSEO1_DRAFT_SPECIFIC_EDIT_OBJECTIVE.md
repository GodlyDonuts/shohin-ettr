# DSEO1: Draft-Specific Edit Objective

Status: frozen prospective successor, 2026-08-09. OBR1 and its conditional
MPR3 campaign remain unchanged. No DSEO1 capability output has been produced.

## Identifiability defect

Prior temporal-revision corpora train a final trajectory `y` from source `x`
and draft `d` with ordinary autoregressive cross entropy:

\[
  \mathcal L_{final} = -\log p_\theta(y\mid x,d).
\]

When the verified target is the same regardless of draft quality, the model
can achieve the source-only optimum

\[
  p_\theta(y\mid x,d)=p_\theta(y\mid x),
\]

and ignore `d`. MPR1 and MPR2 empirically exposed this non-identifiability:
draft-hidden/source-only fitting matched or beat aligned fitting.

DSEO1 changes the supervised action for the same source and same final
trajectory. Each source has paired presentations:

\[
  (x,d_{clean})\to (a_{keep},y),\qquad
  (x,d_{fault})\to (a_{fault},y).
\]

Because `x` and `y` are held fixed while `a` changes, a source-only model
cannot solve the action task. The action is emitted autoregressively as the
first response span, and every final-response token attends causally to it.
It is not an auxiliary classifier.

## Frozen host and owner selection

- pinned host: `allenai/OLMoE-1B-7B-0125-Instruct` at
  `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`;
- frozen base attention, routers, experts, embeddings, and language head;
- all-16-layer rank-18 shared post-MoE residual, exactly 1,179,648
  trainables;
- no extra prepass, controller, router training, expert adapter, recurrence,
  or segment embedding in DSEO1-v0;
- context 4,096 with zero retained source/draft/target truncation.

The first-pass owner is selected by a rule frozen before OBR1 output: use the
exact OBR1 update-2,048 checkpoint only if its registered qualification gate
passes; otherwise use the immutable MPR1 hidden/source-only update-256 owner.
This rule changes draft quality, not thresholds or labels.

## Action vocabulary and targets

The fixed action strings are:

- `<KEEP>` for a verified clean draft;
- `<FIX_FINAL>` for a format-preserving final-answer corruption;
- `<FIX_STEP>` for an independently executable arithmetic/operator mutation;
- `<FIX_CODE>` for a code mutation whose original passes and mutant fails an
  independent test suite; and
- `<REWRITE>` for a natural owner draft that is independently verified wrong
  but has no deterministic local corruption class.

Natural drafts are optional in Stage 1. They cannot delay the paired
identifiability gate. Training targets are the action string, one newline, and
the complete final trajectory. Clean targets copy the exact verified clean
draft after `<KEEP>`; fault targets emit the untouched verified solution after
the corresponding fault action. At inference the model emits both action and
trajectory. A deterministic parser may remove only the initial registered
action string before ordinary answer scoring.

## Loss

Let `A_i` be the action-token positions and `Y_i` the final-trajectory token
positions for presentation `i`. DSEO1-v0 uses per-presentation normalized
losses:

\[
  L_A(i)=|A_i|^{-1}\sum_{t\in A_i} CE_t,\qquad
  L_Y(i)=|Y_i|^{-1}\sum_{t\in Y_i} CE_t,
\]

\[
  L_{DSEO1}=\tfrac12\,\mathbb E_i[L_A(i)]
             +\tfrac12\,\mathbb E_i[L_Y(i)].
\]

Thus the short action span receives half the aggregate objective and cannot
be swamped by a long solution. The implementation must report unweighted
action CE, final CE, weighted components, exact action/final token counts,
and effective per-token weights. DSEO1-v0 has no contrastive term. If the
paired action canary fails, no full fit or margin-loss rescue is authorized.

## Source and corruption custody

The source is a broad, verified, OLMoE-tokenized corpus already hard-filtered
against the 1,289-row development split by normalized exact identity and
development-unique word 13-grams. DSEO1 adds a new source-disjoint paired
canary split before fitting and never reads holdout.

Every corruption record binds:

1. source identity and clean response SHA-256;
2. corruption family and deterministic seed/operator;
3. exact changed character and token spans;
4. pre/post draft SHA-256;
5. original verifier result and mutated verifier result; and
6. complete OLMoE source/draft/action/target token counts.

Registered corruptions are:

- final-answer substitution that preserves the original answer surface form
  while deterministically choosing a different numeric or choice value;
- logic/science choice mutation to the next valid non-gold option;
- arithmetic/operator mutation only where an independent executor can prove
  the clean result and reject the mutant; and
- code mutation only after rerunning the source test suite and proving the
  original passes while the mutant fails at least one test.

Rows without a verifier-backed mutation are excluded rather than weakly
labeled. Pair order is deterministic, clean/fault counts are exact, and every
training batch contains equal clean and fault presentations.

## Matched arms

All arms use identical host, initialization, residual geometry, paired source
multiset, update count, optimizer, context, target trajectory, weighted-loss
scale, and charged final tokens.

1. **Aligned action:** correct action labels and visible aligned drafts.
2. **Within-source swapped action:** clean and fault action labels exchanged;
   final trajectory remains identical.
3. **Draft hidden:** same IDs/positions/action/final geometry, but all draft
   keys are causally hidden throughout the model.
4. **Final-only:** a constant registered prefix occupies the same positions;
   its loss is masked, and the final trajectory receives the same aggregate
   weighted loss as the other arms.

Fixed references are unchanged OLMoE and the selected source-only owner.

## Stage 0 paired canary

Before any broad capability evaluation, use 8,192 source identities for
training and a separately frozen 1,024-source paired diagnostic, stratified by
available corruption family. Train exactly 256 updates at LR `2e-5`, batch
one pair, accumulation eight pairs, and seed `2026080915`. Every condition is
required:

1. aligned action accuracy `>=95%` overall and in every represented
   corruption family;
2. aligned counterfactual consistency `>=90%`: changing only clean versus
   fault draft changes the exact action correctly;
3. swapped and hidden action accuracy each `<=60%` against correct labels;
4. zero regression in verifier correctness on clean pairs;
5. verifier-correct repaired final trajectory on `>=90%` of fault pairs;
6. no source-only action cue detected by a source-only control; and
7. complete hashes, pair balance, loss components, updates, FLOPs, peak
   memory, latency, and generated-token receipts.

A miss closes exact DSEO1-v0 before benchmark capability spend. It does not
authorize action vocabulary, loss-weight, seed, adapter-width, duration, or
threshold variants.

## Action-causality intervention

On a fixed subset of the paired diagnostic, generation is rerun after forcing
only the registered first action span through decoder prefill/logit masking:
force `<KEEP>` on faults and force the correct fault action on clean drafts.
The downstream trajectory must change in the predicted direction: forced
KEEP must reduce fault repair, and forced fault actions must reduce exact clean
copying. If final trajectories are invariant, the action is decorative and
DSEO1 fails regardless of action classification accuracy.

## Conditional capability gate

Only a Stage 0 pass opens one full development fit. Before any DSEO1
capability output, promotion is frozen as:

1. aligned treatment at least 13 answers above the direct selected owner;
2. aligned at least 13 answers above swapped, hidden, and final-only controls;
3. nonnegative MATH, logic/science, and executable-code deltas versus the
   direct owner;
4. positive conservative possible-semantic repairs minus strict breaks;
5. development-compatible action accuracy `>=95%` and counterfactual
   consistency `>=90%`; and
6. the registered action intervention remains causal.

One conjunctive development pass opens exactly one sealed holdout with the
same margins and nonnegative domains. Only development plus holdout pass
authorizes larger-MoE scaling. If Stage 0 actions pass but final accuracy is
control-equivalent, close the interface rather than increasing parameters.

## Claim boundary

A pass supports the narrow claim that draft-specific autoregressive edit
actions make model-owned temporal revision causally identifiable and useful
on a MoE. It does not by itself establish novel routing, expert specialization,
general self-correction, or scaling to large MoEs.
