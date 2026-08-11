# KCR1: Keep/Continue/Restart Causal Revision

Status: prospective successor frozen after exact NDR1 closed and before KCR1
model output. CPU data admission passed on 2026-08-10. NDR1 remains closed.
Holdout remains sealed.

## CPU admission result

CPU job `750637` admitted `11,218/11,220 = 99.982%` source identities and
rejected only two sources lacking a safe semantic continuation boundary. It
created `33,654` rows with all three presentations retained per source and
`33,654/33,654` exact transaction round-trips. The action distribution is
16,396 KEEP, 11,218 CONTINUE, and 6,040 RESTART. The actual pinned Qwen
tokenizer reports maximum 2,104 prompt tokens and 1,490 target tokens, zero
4,096-token truncation, and 3,416,424 charged target tokens. Holdout was not
used and the only runtime-visible field is `question`.

- train SHA-256:
  `3e63c4e248a32c9f4008a1c245ea6e18bcb2f8497058b6c0c237ca5a719863b9`
- report SHA-256:
  `f6236490f2e435c841267c849460183db6fbf90fa30f6d1a61b3bd4a64b79b8e`
- local result: `SHOHIN_KCR1_DATA_RESULT.json`

The implemented training path preserves the action field through reservoir
selection, proves that each tokenized response begins with its declared action,
and applies the frozen loss exactly: KEEP averages action plus terminal EOS;
payload transactions assign one half of presentation loss to action-marker
tokens and one half to newline, payload, and terminal EOS. This path is opt-in;
ordinary historical language CE is unchanged. One hash-bound update must pass
before any 512-update fit.

## Evidence that selects the changed mechanism

NDR1 trained ordinary full-trajectory CE on aligned natural drafts and a
same-domain nearest-length shuffled control. Aligned scored `306/1,289`,
shuffled `343`, and unchanged `340`; aligned also exhausted 879 generations
versus shuffled 768. Read-only identity-level attribution binds all 1,289
immutable rows and finds:

- only `1,405/11,220 = 12.52%` of training drafts were token-exhausted;
- `910/1,289 = 70.60%` of development drafts were token-exhausted;
- aligned loses 37 pairwise answers overall;
- 33 of those 37 answers are lost on wrong, token-exhausted drafts; and
- aligned emits 75,804 more tokens than shuffled.

This does not reopen NDR1 or justify an exhaustion-reweighted retry. It selects
a different output mechanism: draft termination state must choose a distinct,
executed computation branch rather than remain optional prompt text for a
full-answer generator.

## Architecture

The first owner emits a complete draft `d` and one deterministic controller
bit `z` recording whether generation stopped normally or reached its token
ceiling. The later owner receives source `x`, exact draft `d`, and `z`, then
emits one transaction:

```text
<KEEP>
```

```text
<CONTINUE>
payload
```

```text
<RESTART>
payload
```

A generic deterministic transducer executes exactly one branch:

\[
T(d,a,p)=
\begin{cases}
d & a=\mathrm{KEEP}\\
d\Vert p & a=\mathrm{CONTINUE}\\
p & a=\mathrm{RESTART}.
\end{cases}
\]

Malformed actions, a payload after `KEEP`, or an empty payload after
`CONTINUE/RESTART` fail closed. No verifier, answer label, solver, task router,
or host repair exists at inference. The cutoff bit is generation-controller
state owned by the first pass, not semantic supervision.

This differs materially from NDR1: the action changes output semantics and
correct drafts can be preserved byte-for-byte. It also differs from DSET1's
last-span synthetic editor: KCR1 handles nonlocal natural failures by choosing
between exact preservation, append-only continuation, and complete restart.

## Prospective data contract

Use the already-admitted, evaluation-disjoint 11,220-source NDR1 bank and its
immutable natural B1 drafts. Each source contributes a balanced three-state
episode around one verified response. A conservatively verified natural KEEP
may preserve a different but independently correct model-owned trajectory:

1. **KEEP:** the complete verified response is the visible draft; transaction
   target is exactly `<KEEP>`.
2. **CONTINUE:** a deterministic semantic-boundary prefix of the verified
   response is the visible cutoff draft; transaction target is `<CONTINUE>`
   plus the exact remaining suffix.
3. **NATURAL:** the exact immutable B1 draft is visible. A conservative
   deterministic assessor may label a complete verified non-code draft KEEP;
   all unverified, code, or exhausted natural drafts receive RESTART plus the
   full verified response. No natural draft may be relabeled CONTINUE without
   an independently verified prefix certificate.

The builder must preserve all natural drafts, report every action by domain and
draft termination state, and prove exact transaction execution. The complete
prompt plus transaction must fit 4,096 Qwen tokens without truncation. A source
is admitted only if all three presentations fit; no presentation-level
filtering is allowed. Development and holdout sources are forbidden by the
existing exact and protected word-13-gram boundary.

## Model and optimization

- pinned Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`;
- immutable B1 update-256 checkpoint SHA-256
  `854a7cc44fbc2b54418f4e5bd09b7efeed0da44fc9ce217b0bb6b1997b722971`;
- final-four rank-8 revision LoRA, exactly 2,704,896 trainables;
- pair-balanced source episodes, 512 updates, batch 1, accumulation 8,
  learning rate `2e-5`, and the NDR1 seeds;
- action-token CE and payload-token CE each contribute one half of the
  per-presentation objective when a payload exists; KEEP uses action CE only;
- greedy transaction generation with a separately frozen bounded payload
  budget.

No GPU fit opens until CPU data/token/execution admission and a one-update
gradient mechanics check pass.

## Matched arms

1. **KCR aligned:** correct source-local draft state and transaction.
2. **Within-source action permutation diagnostic:** the three visible drafts
   remain source-local, but KEEP/CONTINUE/RESTART transactions are cyclically
   permuted. This is a causal canary, not a capability baseline; its executed
   targets intentionally differ.
3. **Constant RESTART:** identical model, source/drafts, parameters, and
   target responses, but every presentation executes a complete restart.
   Updates are adjusted prospectively to match charged target tokens and
   training FLOPs rather than presentation count.
4. **Draft-hidden:** exact prompt/token geometry with draft content causally
   hidden through every model layer; the cutoff bit remains visible and is
   reported separately.

The already-closed NDR1 aligned/shuffled arms and the qualified IDR1 reviser
remain immutable references, not tunable controls.

## Gates

### CPU admission

- 100% exact execution round-trip for all generated transactions;
- zero prompt/target truncation at 4,096;
- all three presentations retained per admitted source;
- exact source and natural-draft coverage receipts;
- zero evaluation overlap under the existing exact and protected-13-gram
  assessor; and
- no verifier or correctness field in runtime-visible examples.

### Source-disjoint transaction canary

Before broad capability scoring, aligned must achieve at least 95% action
accuracy overall and per branch, at least 95% executed-trajectory correctness,
at least 99% KEEP byte preservation, and at least 90% counterfactual
consistency across each source's three states. Action-permuted and draft-hidden
controls must each remain at or below 60% action accuracy. Forced-action
intervention must change the executed trajectory exactly as specified.

### Capability development

On the frozen 1,289-row IDR1 development board, all conditions are conjunctive:

- at least `603/1,289` answers;
- at least 13 answers over constant RESTART and draft-hidden controls;
- math at least 223, logic/science at least 349, and code at least 17;
- at least 13 net repairs on wrong exhausted drafts;
- no more than 400 payload exhaustions; and
- complete parameter, token, FLOP, memory, latency, action, branch, and hash
  receipts.

Only a conjunctive development pass opens one sealed holdout. Any miss closes
exact KCR1 without action syntax, split rule, loss weight, rank, layer, update,
seed, decoding, parser, or threshold variants.

## Claim boundary

A pass would establish that explicit model-owned termination state and a
causally executed branch transaction improve temporal revision. It would not
prove unrestricted reasoning, optimal branch semantics, or a native 125M
Shohin mechanism. A failure closes this exact three-branch interface.
