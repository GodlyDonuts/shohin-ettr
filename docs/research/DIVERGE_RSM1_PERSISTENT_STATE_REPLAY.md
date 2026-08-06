# DIVERGE-RSM1: Persistent Discrete State Replay

Status: architecture and component gate frozen before neural execution.

## Why this is not a CRP1 repair

CRP1 established that guarded first-error packets can cause complete
arithmetic and register corrections. It failed because the selected packet was
only converted into a short prefix for a frozen autoregressive generator. The
generator could independently solve some rows, ignore packet identity on
others, and never transferred to held symbolic tapes. RSM1 does not change the
CRP1 packet, its training, its checkpoint, its selector, or its thresholds.

RSM1 replaces the entire post-selection substrate. A frozen selected packet
must initialize one bounded discrete state. A tied transition core then
consumes the remaining step surfaces one at a time, emits a hard state after
every transition, and receives only that emitted state at the next transition.
The terminal answer is the terminal state itself. There is no language-model
generation, answer prompt, hidden host execution, beam, retry, or fieldwise
averaging.

## Capability hypothesis

The large CRP1 localization delta (`313/480` guarded versus `64/480`
unguarded) should become a material answer delta only when the selected fault
line owns downstream execution. A hard state bottleneck should also reveal
whether symbolic failure came from free-form language replay or from the
packet/source representation itself.

## Runtime

```text
complete problem + complete draft
  -> frozen SmolLM3 source features
  -> frozen CRP1 candidate packet and hard selection
  -> initial state-byte distribution from selected whole packet
  -> hard straight-through byte state
  -> tied recurrent transition over each remaining surface step
  -> hard byte state after every step
  -> exact terminal byte decoding
```

State vocabulary is deliberately small and domain-neutral: digits, lowercase
letters, minus, comma, `EOS`, and `PAD`, in 24 fixed positions. Scalar values,
ordered register pairs, and symbol tapes use the same state machine. Surface
step spans come from the rendered draft. Exact program objects and state
trajectories exist only in the supervisor and assessor; they are never passed
to the candidate runtime.

The selected candidate controls both the initial packet and the replay start.
`NO_ERROR` performs no transition. Candidate `e > 0` starts immediately before
step `e` and executes steps `e..N`. Selecting an earlier valid rollback point
is allowed to recover; selecting after the true first error begins from a
corrupted lineage and should fail.

## Ordered experiment

1. Reconstruct and hash exact state trajectories for the existing CRP1 board.
   Validate final-state parity, state-code round trips, and surface-only
   operation extraction on every row.
2. Load the immutable guarded CRP1 checkpoint and freeze the full source plus
   packet. Train only RSM1 with gold packet selection. This is the oracle-
   selection component gate.
3. Evaluate the same checkpoint twice: forced gold packet and autonomous
   packet. Do not train a matched control unless the forced component gate
   passes.
4. On component pass, train the identical RSM1 from the same initialization on
   the immutable unguarded packet and run autonomous guarded/unguarded, reset,
   shift, and packet-swap evaluations.

Training uses three declared losses with equal weight: selected-boundary state
decoding, full hard free-running replay, and one-step transition supervision
from gold predecessor states. The one-step term teaches the tied transition
algebra without changing autonomous inference; the free-running term is always
active and receives only model-emitted hard states. No scheduled-sampling ratio
or teacher-forcing curriculum is tunable after the gate begins.

The one frozen component run uses seed `2026080605`, 1,600 optimizer updates,
four board identities per microbatch, accumulation two, and therefore eight
wrong/correct identity pairs (16 rendered traces) per update. This is 12,800
identity exposures, about 2.67 passes over the 4,800-row board. AdamW uses
`lr=3e-4`, betas `(0.9, 0.95)`, weight decay `0.01`, unit gradient clipping,
and one cosine decay to zero. State width is 256 with 24 hard byte slots,
eight attention heads, and a four-times feed-forward expansion. A two-update
smoke may test mechanics and memory only; it cannot alter this scientific
budget.

## Frozen component gate

All conditions are conjunctive:

- exact terminal state at least 432/480 under forced gold packet selection;
- at least 136/160 forced exact states in every family;
- exact free-running intermediate state trajectories at least 80% per family;
- zero invalid byte strings, overflow, host semantic calls, or changed frozen
  source/packet tensors; and
- finite loss, gradients, and hard-state feedback throughout.

Failure closes RSM1 before matched autonomous composition. No seed, width,
state vocabulary, duration, teacher-forcing, loss, or family-specific repair is
allowed.

## Frozen autonomous promotion gate

If and only if the component gate passes:

- guarded exact terminal states at least 288/480 and 80/160 per family;
- guarded beats matched unguarded by at least 48 answers and by at least ten
  answers in every family;
- guarded preserves at least 432/480 correct no-op twins;
- reset, shifted selection, and packet swap each cost at least 48 answers;
- at least 80% of guarded autonomous answers have an exactly correct complete
  hard-state replay; and
- parameter, update, source-feature, packet, and transition-FLOP receipts match.

A pass establishes only a bounded model-owned packet-to-state replay
mechanism. Natural verified traces remain a separate transfer gate.
