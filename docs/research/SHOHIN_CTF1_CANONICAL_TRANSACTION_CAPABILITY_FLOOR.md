# CTF1: Canonical Transaction Capability Floor

Status: closed negative on development; public test and holdout sealed

Date: 2026-08-10

## Question

CTE1 taught pinned Qwen3.5-0.8B to emit executable canonical traces, but
accuracy fell to `134/666` versus its direct owner at `267/666`. DTC1, DTMC1,
CTE1, and LTR1 jointly localize failure to semantic planning before learned
execution. CTF1 changes only owner capacity: can an untouched, stronger
same-family model already use the exact canonical transaction interface?

## Frozen Intervention

- exact pinned `Qwen/Qwen3.5-4B` revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`;
- no adapter, fine-tuning, warm start, selector, verifier, or host repair;
- exact immutable 666 CTE1 development identities and prompt;
- greedy no-thinking generation, maximum 512 new tokens, seed `2026081053`;
- unchanged DTC parser, typed graph, and learned LAM1 executor;
- normal and same-register-depth source-shuffled views;
- aligned state-reset and opcode-permutation interventions; and
- no holdout or public GSM8K access.

CTF1 is an interface/capability-floor test, not a parameter-matched comparison
or a claim that scale is the Shohin mechanism.

## Prospective Gate

Every condition is conjunctive:

- at least 600 traces compile and execute normally;
- aligned reaches at least `300/666` exact answers;
- aligned exceeds trained 0.8B CTE1 by at least 100 answers;
- source shuffle is at most `67/666`;
- at least 300 aligned rows contain a causal state read;
- state reset loses at least 20 points on linked-correct rows;
- opcode permutation loses at least 30 points from aligned;
- zero normal execution invalidity; and
- public test remains closed.

A pass establishes that the source-to-canonical-ledger-to-learned-execution
interface is viable above the observed 0.8B capacity floor and permits one
separately frozen 4B post-training experiment. A miss closes this interface
across both tested scales without prompt, parser, decoding, threshold, or
nearby model variants.

## Claim Boundary

CTF1 cannot establish that LAM improves an already-capable 4B model, because
there is no matched direct-answer comparison in this read-only ceiling. It
tests whether a capable model can own the semantic program consumed by the
existing learned executor under causal controls.

## Frozen Result

The first jobs `750098/750099` failed before generation because the new raw
owner wrapper selected the adapter embedding path. Commit `80c11a0` changed
only that dispatch boolean and added a guard. Immutable replays `750111` and
`750112` then completed in 917/867 seconds from runtime manifest
`36ec677f...fb97`.

- aligned learned-execution answers: `419/666 = 62.9129%`;
- source-shuffled answers: `7/666 = 1.0511%`;
- compiled/executable rows: `562/562`;
- linked rows: 544;
- state-reset linked-correct: `0/419`;
- opcode-permuted correct: `3/666`;
- normal execution invalid: zero; and
- exhausted generations: seven.

The score clears the capability, scale-margin, source-causality, state, opcode,
and validity conditions, but misses the frozen `>=600` coverage condition.
CTF1 is therefore conjunctively FAIL; the public test remains sealed.

Read-only attribution clarifies the boundary. The untouched 4B owner's own
claimed final answer is correct on `487/666`. Path overlap is: 408 both direct
and ledger correct, 79 direct-only, 11 ledger-only, and 168 neither, for an
oracle union of 498. Of the 104 compile-invalid rows, 75 have a correct claimed
final answer. Stronger scale restores semantic planning, but forcing every
answer through this exact canonical ledger discards more correct model-owned
answers than it repairs.

Exact CTF1 closes without prompt, parser, decoding, threshold, adjacent scale,
or post-training rescue. Normal/source-shuffled/aggregate report SHA-256
values are `dc4e939b...05f0`, `9ce8dae6...2c63`, and
`8bccbe9a...5b77`.
