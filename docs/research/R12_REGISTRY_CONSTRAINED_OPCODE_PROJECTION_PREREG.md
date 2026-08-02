# R12 Registry-Constrained Opcode Projection Preregistration

**Status:** implementation complete; fixed 5,000-update diagnostic pending.

**Date:** 2026-08-02 EDT

**Claim boundary:** parameter-free inference diagnostic. This experiment can
localize a whole-program decoding defect, but it cannot establish native
reasoning unless both unchanged source-deleted causal axes improve and the
result later survives trained replication.

## Causal hypothesis

Contract-v8's 5,000-update syntax-graph compiler improves every local schedule
field while exact terminal packets regress to zero. The flat sticky selector
then collapses to the dominant opcode skeleton. These results leave a narrower
hypothesis: the per-step compiler contains useful conditional opcode evidence,
but independent argmax decisions form invalid or incoherent complete programs.

The treatment projects those existing per-step probabilities onto one complete
train-registry opcode path. It scores each valid path by mean per-step log
probability and makes one global hard selection. Registry order, empirical
class counts, targets, and candidate answer quality do not enter the score.
This removes both stepwise hybridization and the learned majority-class prior.

## Frozen inputs

- Base schedule: completed contract-v8 5,000-update V100 arm `725573`.
- Base report SHA-256:
  `7922f9b26a41be8125983149d3d896ba812deb4a65765d5b7a02f4bfad8c8623`.
- Train-only registry SHA-256:
  `03fc92829bc4a1c9f9e8381953ac506e04afeef746871a60ebfca1e482cbafcc`.
- Registry payload SHA-256:
  `d58185b4a5c7b28e54cd9497215dd8d5f0e52f7339a968f10facbc6669497b4b`.
- Architecture/data seed: `31/11`.
- Development batches: 32, exactly 512 rows.
- Learned parameter delta: zero.

The base compiler checkpoint and registry remain hash-bound and read-only. The
projection may consume only the base compiler's opcode probabilities and the
frozen train-only opcode sequences. It may not consume QUERY, answers, terminal
targets, oracle programs, development-derived templates, or candidate scores.

## Fixed projection

For a base opcode distribution `p[t, opcode]` and candidate program `z`:

```text
score(z) = mean_t log p[t, z[t]]
z*       = argmax_z score(z)
```

Programs longer than the requested rollout are ineligible. Each candidate's
terminal opcode is replayed after its end, which is state-neutral because the
exact algebra has already committed, halted, or rejected. Source, target,
relation, type, and value predictions are unchanged from the sealed base
compiler. The selected opcode skeleton is replayed only by the existing exact
transaction algebra.

## Required report

- projected opcode and joint schedule accuracy;
- oracle-initial and autonomous-initial exact terminal packets;
- fully autonomous factual top-1;
- strict, margin-1, and intervention-DID WORLD and COMMAND gates;
- registry, schedule-run, source, and output hashes; and
- an explicit comparison with the unprojected 5,000-update endpoint.

## Decision rule

The projection justifies a trained structured successor only if it restores a
nonzero exact terminal packet rate without factual regression, or crosses at
least one fully autonomous strict causal axis. It is a native-reasoning
candidate only if both strict axes improve, then replicate across seeds and
held-out population orderings. A gain visible only with oracle initial state
remains an interface diagnostic.

If projection remains zero on exact terminal state and both causal axes, the
valid-path decoding hypothesis is closed for the current opcode evidence. The
next architecture must change how program evidence is represented, not merely
how independent decisions are decoded.
