# DIVERGE-HSC1 Support-Rank Diagnostic

**Status:** frozen read-only diagnostic; no HSC1 training or promotion standing

**Decision date:** 2026-08-05

## Question

HSC1 is closed: its single Viterbi parse reaches 96.094% exact packets on the
train distribution but only 8.594% under lexical shift and zero under renderer
and composition shifts. This audit does not repair or rescore HSC1. It asks one
question needed to choose a different successor architecture:

> Does the frozen HSC1 checkpoint still assign enough structured posterior mass
> to the valid shifted interpretation that a compact packed support lattice
> could preserve it, or is the valid interpretation absent even from a broad
> bounded envelope?

## Frozen assessor

The assessor loads the exact failed checkpoint with no gradients and regenerates
the same four 256-episode cohorts at fresh seeds. Record boundaries remain the
frozen learned boundaries. For each gold record it measures:

1. exact rank of the gold monotonic phase triple among legal cut triples;
2. exact rank of the gold cue-position/kind assignment;
3. exact rank of each gold semantic template among all 128 templates, scored by
   the exact log-partition over every legal token alignment; and
4. whether the gold token alignment is Viterbi within its gold template.

The candidate packed envelope is deliberately not a top-K list of complete
parses. It retains the complete cut and cue lattices, K semantic templates per
option, and every finite-state alignment inside each retained template. The
audit reports packed dynamic-program cells separately from the corresponding
unmaterialized Cartesian interpretation count.

This use of gold spans and labels is assessor-only. It measures whether the
frozen scores contain a recoverable interpretation; it is not an executable
compiler and has no capability standing.

## Frozen decision

Evaluate `K = 1,2,4,8,16,32,64,128`.

- If some `K <= 64` retains every gold fault-line interpretation in at least
  95% of episodes in **each** lexical, renderer, and composition shift, a new
  support-lattice compiler is justified. That successor must encode each record
  once, retain uncertain record membership/cuts/templates as guarded variables,
  prove exact extensional parity on CPU, and compare packed execution against
  full-particle and single-Viterbi controls.
- If no `K <= 64` clears that floor, do not build the lattice successor from
  HSC1 scores. Replace the language interface/backbone or training task instead.
- `K=128` is reported only as an exhaustive fixed-grammar diagnostic. It cannot
  satisfy the compact-envelope gate.

No seed, width, duration, loss, cue, template, threshold, source-layer, or
optimizer change is permitted. A positive rank audit would qualify a distinct
architecture experiment, not reopen HSC1 or the failed broad DIVERGE claim.
