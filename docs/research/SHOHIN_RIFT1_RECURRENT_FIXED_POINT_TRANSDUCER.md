# RIFT1: Recurrent Idempotent Fixed-Point Transducer

**Status:** prospectively frozen before output, 2026-08-10  
**Scope:** opened source-disjoint development only; holdout sealed

## Thesis

FRET1 localizes a complementary asymmetry in the frozen DSET-Q35 host. Its
mandatory rewrite has a 99.27% exact content pointer and repairs 99.48% of
fault drafts, but damages clean drafts. Ordinary DSET preserves 99.90% of
clean drafts but misses some faults. RIFT1 turns these two modes into one tied
recurrent edit process rather than training another semantic classifier.

1. **Propose:** the frozen DSET host receives the fixed architecture token
   `<REPLACE_LAST>` and emits a content-addressed edit.
2. **Execute:** a generic deterministic transducer copies the draft and applies
   that model-owned edit.
3. **Re-read:** the exact same frozen DSET host receives source plus the
   executed proposal as its new draft and emits KEEP or one repair.
4. **Commit:** the same generic executor applies that second model-owned action;
   malformed second actions fail closed to the proposal.

The expected fixed point is a trajectory that the model rewrites and then
marks KEEP. No verifier, solver, task route, answer label, hidden host repair,
new parameters, or external semantic decision exists at inference. Weight
tying makes recurrent computation, not additional capacity, the changed
factor.

## Frozen gate

- pinned Qwen3.6-35B-A3B and immutable aligned/hidden DSET checkpoints;
- exact FRET1 aligned/hidden proposals and 1,908-row development board;
- one proposal transition and exactly one commit transition;
- greedy decoding, 32 tokens, seed `2026081011`, eight shards per arm;
- identical hidden control, which cannot attend to draft bytes;
- no public benchmark or holdout.

RIFT1 passes only if all conditions hold:

1. aligned final execution `>=1,874/1,908` and commit-script validity `>=95%`;
2. choice final execution `>=220/256`;
3. clean final execution `>=945/954` and fault final execution `>=859/954`;
4. aligned exceeds ISET1 (`1,838`) by at least 13 answers;
5. aligned exceeds hidden RIFT1 by at least 13 answers;
6. zero decode-limit exhaustion and complete hash/latency/memory receipts.

A pass opens exactly one sealed holdout and one forced-second-action
intervention. A miss closes exact RIFT1 without recurrence-count, prefix,
checkpoint, prompt, decode, threshold, seed, or fallback variants.

## Claim boundary

A pass would support a tied, model-owned recurrent sequence-transduction
mechanism that converts asymmetric propose/preserve behavior into a stable
commit. It would not establish arbitrary multi-edit planning or unrestricted
reasoning, and the DSET adapter primitive is not claimed as novel.
