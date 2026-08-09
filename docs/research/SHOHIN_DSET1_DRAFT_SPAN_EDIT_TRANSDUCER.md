# DSET1: Draft-Span Edit Transducer

Status: frozen prospective mechanism gate, 2026-08-09. DSEO1-v0 and the
read-only DSEC0 binary-gating ceiling are closed. No DSET1 model output exists
at freeze time. Development and holdout benchmarks remain unopened.

## Why DSEO1 is insufficient

DSEO1 makes the edit action identifiable, but its full-response target remains
the same for clean and faulted drafts. The final generator can therefore learn
the action and still ignore it. Measured evidence is aligned action `94.73%`
versus swapped `5.91%`, while answer accuracy is statistically and practically
control-equivalent (`89.21%` aligned, `89.01%` swapped, `90.82%` final-only).

DSEC0 then grants the binary action a deterministic execution role: `KEEP`
copies the draft and `FIX` uses DSEO1's generated rewrite. It reaches only
`1,870/2,048`, ten answers above final-only, with `83.40%` fault repair. Even
oracle actions reach only `84.57%` fault repair. Binary selection is not the
remaining bottleneck; full-response rewrite is.

## Changed mechanism

DSET1 replaces full-response regeneration with a model-owned edit script and a
deterministic copy/edit transducer. The only legal scripts are:

```text
<KEEP>
```

and

```text
<REPLACE_LAST>
old_surface
new_surface
```

`KEEP` emits the exact visible draft. `REPLACE_LAST` finds the last exact
occurrence of `old_surface` in the draft, replaces it once with `new_surface`,
and emits the resulting complete trajectory. Empty or multiline surfaces,
missing old surfaces, malformed scripts, and unknown actions fail closed. No
verifier, solver, answer label, host repair, or hidden source access exists at
inference. The transducer is part of the decoder contract, analogous to a copy
head: the model owns every discrete edit field, while deterministic execution
only materializes those fields.

For a fixed source `x`, clean and fault drafts require different script tokens
and produce different executed trajectories. The target is therefore not
solvable by the DSEO1 shortcut:

\[
  (x,d_c)\to KEEP\to d_c,
  \qquad
  (x,d_f)\to REPLACE\_LAST(o,n)\to d_f[o\mapsto n].
\]

The old surface `o` is draft-specific. A source-only model may infer `n`, but
cannot emit the exact script without observing the faulted draft.

## Data repair and custody

DSET1 derives from the immutable DSEO1 train/diagnostic identities but does
not trust DSEO1's `clean_verifier_passed` flag. A new CPU builder independently
requires:

1. clean and fault trajectories differ only at the registered span;
2. the span is the final occurrence of the clean/fault surface;
3. the registered span is the structurally final explicit-answer or boxed
   answer surface, with token-boundary checks that exclude LaTeX command
   letters;
4. exact execution restores the complete clean trajectory, so scoring does
   not depend on an ambiguous broad answer extractor;
5. both script presentations fit the pinned OLMoE 4,096-token context without
   truncation; and
6. train and diagnostic source identities remain disjoint.

This explicitly removes the discovered DSEO1 corruption bug where an
incorrect one-letter metadata label could mutate the `e` in `\\text{F}` while
leaving answer `F` unchanged. Every dropped pair and reason is reported.

## Host, initialization, and optimization

- pinned `allenai/OLMoE-1B-7B-0125-Instruct` revision
  `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`;
- frozen attention, routers, experts, embeddings, and LM head;
- unchanged all-16-layer rank-18 shared post-MoE residual, exactly 1,179,648
  trainables;
- immutable MPR1-hidden update-256 owner initialization;
- three arms trained independently for 512 updates, LR `2e-5`, seed
  `2026080916`, one pair/microstep and eight pairs/update;
- script-token CE normalized per presentation, then averaged across the pair;
- greedy script decoding with at most 32 new tokens.

The richer script receives 512 updates prospectively, not as a rescue chosen
from output. All arms use identical pair order, source multiset, parameter
count, optimizer, context, and update count.

## Matched arms

1. **Aligned:** correct script with the aligned visible draft.
2. **Within-source swapped:** clean receives its paired fault script and fault
   receives `KEEP`; source, pair, token budget, and initialization are fixed.
3. **Draft hidden:** correct script target with the complete draft span causally
   hidden through every model layer.

The immutable DSEO1 final-only score (`1,860/2,048`) is the practical answer
reference, not a trained DSET1 arm.

## Frozen gate

All conditions are conjunctive on the source-disjoint paired diagnostic:

1. aligned exact script accuracy `>=90%` overall and per corruption family;
2. aligned counterfactual pair consistency `>=90%`;
3. aligned executed answer accuracy `>=95%` overall;
4. aligned clean-copy accuracy `>=99%` and fault-repair accuracy `>=90%`;
5. aligned exceeds DSEO1 final-only by at least 13 answers;
6. aligned exceeds swapped and hidden execution by at least 13 answers each;
7. swapped and hidden exact script accuracy are each `<=60%`;
8. zero accepted malformed edits, missing old surfaces, or non-final
   replacements; and
9. complete data, checkpoint, parameter, token, memory, latency, and hash
   receipts.

A one-update mechanics pass is required before the three fits. Any gate miss
closes exact DSET1-v0 without script syntax, seed, width, duration, layer, or
threshold variants. No capability benchmark or holdout opens on a miss.

## Claim boundary

A pass would establish that a draft-conditioned model can own a compact edit
program whose deterministic execution materially improves complete answers on
a MoE. It would not establish general planning, arbitrary-span editing,
natural-draft fault localization, or large-MoE transfer. Those require later,
separately frozen gates.
