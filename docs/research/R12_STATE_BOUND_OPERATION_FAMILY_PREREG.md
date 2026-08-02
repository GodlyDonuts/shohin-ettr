# R12 State-Bound Operation-Family Arbiter Preregistration

**Status:** completed and rejected at the frozen gate; local family line closed
**Contract:** `shohin-ettr-parallel-terminal-state-contract-v20`  
**Promotion gate:** at least 90% independent held-out operation-family exactness  
**Reasoning gate:** unchanged source-deleted WORLD and COMMAND strict pairs

## Measured reason for the mechanism

The corrected full-corpus audit covers 494,480 training and 60,512
development operations. Its development conditional-oracle family accuracies
are:

| Interface | Oracle accuracy |
|---|---:|
| Public operation syntax | 80.1692% |
| Syntax plus preceding-state topology | 80.4435% |
| Syntax plus exact preceding-state values | 94.0756% |

Exact state values therefore contain enough information to cross the frozen
90% family gate, but a smoothed additive factor model remains at 73.7110% in
all three modes. LINK is perfectly separated; the residual ambiguity is NONE
versus WRITE. Exact composite signature lookup is not viable because only
18.7864% of development operations have a train-seen exact syntax-plus-state
signature.

The audit report file SHA-256 is
`819aa0bdb57b3e46fcf1488323d17e18fd3668db277d4404ed685f76a69b75c5`;
its payload SHA-256 is
`0a5574bf499440fe4a3ca47d80b08d39e9177e76343ecb8f6a3304f0247ea517`.

## Hypothesis

The public operation roles identify *what should be compared*, while exact
typed-state values determine whether that operation is a semantic NONE or
WRITE. Pooled addition can represent each side independently but does not
force their binding. A role-conditioned bilinear lookup should expose the
missing interaction without supervising answers, terminal states, or
programs at inference.

## Architecture

`OperationStateBoundFamilyGatedWriteLinkCompiler` preserves v19's compiler,
recurrent operation boundary, hard NONE/WRITE/LINK gate, and typed payload
rails. It changes only the family arbiter:

1. Re-encode the exact current typed state into one memory vector per slot.
2. Project each public semantic-role anchor into multihead queries.
3. Project current state slots into keys and values.
4. Compute role-to-slot bilinear compatibility and attended state values.
5. Concatenate role, attended state, and their elementwise product.
6. Pool only valid roles and predict exactly one NONE/WRITE/LINK family.

The production compiler has 50,594,556 parameters, 1,576,448 more than v19;
the complete system is 206,533,450 parameters. The user has removed the old
200M ceiling. This increase is nevertheless tied to one measured causal
purpose and is not evidence by itself.

## Information boundary

Inference may read only the autonomous current typed state and public COMMAND
tokens/residuals. It may not read QUERY, answers, target packets, successor
states, transaction programs, assessor traces, candidate scores, or a host
semantic solver. Oracle preceding states are allowed only as family-island
training labels. The immutable evaluator and fixed state algebra remain
unchanged.

## Training isolation

The first v20 arm is family-only. WRITE/LINK count, pointer, value, relation,
and payload-rail parameters are bypassed, and the runtime gradient guard must
prove they receive no gradient. Training remains class-balanced over
NONE/WRITE/LINK. Terminal and causal outputs from random payload rails cannot
promote the family island.

## Fixed route

1. Complete corrected v15 and its independent seed-13 evaluation first.
2. If v15's family accuracy is at least 90%, do not run v19 or v20; route the
   measured payload failure.
3. If v15 family fails, run exactly one v19 family-island fit.
4. If v19 reaches at least 90%, preserve its weights and release joint rails;
   do not run v20.
5. If v19 misses 90%, run exactly one matched v20 family-island fit.
6. If v20 reaches at least 90%, preserve its weights and release joint rails.
7. If v20 misses 90%, reject standalone NONE/WRITE/LINK control and compile
   typed effects through direct latent expert competition. Do not repeat a
   wider, longer, or reseeded pooled/bilinear family classifier.

Only a later joint system that improves both WORLD and COMMAND strict paired
gates on independent source-deleted populations can count as native reasoning.

## Mechanical evidence before GPU admission

- Exact-value counterfactuals change family probabilities with fixed syntax.
- Family loss sends finite nonzero gradients through role queries, state
  keys/values, the multiplicative projection, and typed value embeddings.
- Family-island end-to-end tests leave all WRITE/LINK payload parameters
  gradient-free.
- Train, evaluator, router, and Slurm launcher agree on contract v20 and its
  fixed pass/fail route.

V20 was authorized only after v19 failed and was run exactly once. This
preserved single-writer scientific attribution and the preregistered endpoint.

The immutable Newton runtime is
`scratchpad/shohin_ettr_state_bound_family_runtime_898591c_r1`, bound to source
`898591cf5861cbff2aef2a293a89347c7fea8bcd`. Its 3,658 files pass complete
checksum replay with zero links and zero writable entries. The SHA256SUMS
SHA-256 is
`73b82ec26467de405ccd7d32ff4031cb0fbacd394c2708ebba355d86cec3d484`.

## Measured closure

The finite campaign selected v19 and then v20. Both completed all 1,000
updates in bfloat16 and both failed the independent seed-13 gate identically:
operation-family exactness `464/1856 = 25%`, 100% predicted `NOOP`, exact
terminal packets `0/512`, WORLD strict `0/48`, and COMMAND strict `0/144`.
V19 final compiler SHA-256 is
`9b741d392c149b1e32bc4040621d603d93dd05c5a25759ae7df580150d445457`;
v20 final compiler SHA-256 is
`0aba95f47bc4c0263872dd1e512798fae26c18fbee2f5b92d9295a2f71d1834f`.
The final planner action is `stop`; v21 was correctly not launched.

Postmortem auditing found that the 94.0756% conditional oracle and the neural
fit did not share an equivalent input or optimizer distribution. The oracle
uses an exact symbolic resolver over public COMMAND structure plus exact
preceding-state factors. V20 instead learns from language residuals, pooled
public role anchors, and a learned typed-state encoding; equivalence was never
measured. More decisively, every one of the 100 logged checkpoints in both
fits omits at least one target family, 33% contain exactly one family, and
LINK appears in only 33% of logged batches. At update 499 an all-LINK batch is
fit essentially perfectly; the final batch is 75% NONE, 25% WRITE, 0% LINK,
and the final model emits NOOP everywhere. The exact trace audits are
`artifacts/r12/v19_operation_family_stream_drift.json` and
`artifacts/r12/v20_operation_family_stream_drift.json`.

This does not reopen v20. Standalone family control remains rejected under
the frozen rules. It changes the capability-floor launch requirements: the
new unified trajectory must first prove symbolic-to-neural interface
sufficiency and use deterministic replay-balanced optimizer windows so one
semantic regime cannot erase another.
