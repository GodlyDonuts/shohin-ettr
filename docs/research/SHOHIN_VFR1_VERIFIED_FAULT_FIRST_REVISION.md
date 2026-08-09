# VFR1: Verified Fault-First Revision

Status: closed negative after the sole frozen train-only quality pilot.

## Result

Job `747591` completed all 128 deterministic rows in 1,966.17 seconds. Only
`7/128 = 5.47%` outputs satisfied the strict two-block parser and all seven
extracted revisions verified. There were no reference leaks or boxed answers
inside parsed FAULT blocks, but `53/128 = 41.41%` outputs exhausted the 1,024
token budget. The other 121 outputs failed tag cardinality.

A single read-only attribution scored the unparsed raw completions directly
with the original per-row assessors. Only `67/128 = 52.34%` were correct:
math `27/67`, logic/science `40/55`, and code `0/6`. Thus the failure is not
merely strict-format compliance; the teacher outputs also miss the frozen
90% correctness floor by a large margin. The full generation, capability
data, matched fits, and development evaluation were not authorized. Jobs
`747593--747600`, `747602`, and `747603` were canceled before allocation.

Trace SHA-256 is `9e0d7812...8c5d`, trace-report SHA-256 is
`5d9286c0...cbd5`, and quality-report SHA-256 is `96244f9a...38b9`. The
machine-readable summary is `SHOHIN_VFR1_PILOT_RESULT.json`. Close exact VFR1
without a teacher-prompt, decoding, context, parser, seed, or threshold rescue.

## Measured bottleneck

IDR1's 9,655 train presentations contain 3,294 `source_verified_repair`
targets. Their median target length is only 11 characters; 3,245 are exactly a
boxed answer and 3,214 are under 80 characters. Thus 34.1% of the revision
curriculum teaches answer emission after a failed draft rather than a
reasoning process that locates and repairs the failure. In contrast, verified
candidate targets have median length 706 characters.

## Hypothesis

A revision owner should externalize a compact fault state before generating
the replacement trajectory. Because later autoregressive tokens attend to the
fault block, this changes the correction computation rather than selecting
among already completed answers. The target format is:

```text
<FAULT>
earliest decisive error or independently checked no-fault statement
</FAULT>
<REVISION>
complete corrected solution
</REVISION>
```

At inference, only the model owns both blocks. There is no verifier, reference
answer, teacher, task router, or correctness bit. A strict parser extracts one
complete revision; malformed traces fail closed.

## Frozen data-quality pilot

Build exactly 5,824 unique train-only teacher requests from the hash-bound
IDR1 identities. The teacher is pinned Qwen3.5-9B at revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a` with the immutable B1 adapter
SHA-256 `854a7cc44fbc2b54418f4e5bd09b7efeed0da44fc9ce217b0bb6b1997b722971`.
It sees source, model-owned draft, and the verified train-only reference only
while constructing supervision. Assessor fields are never rendered.

The first gate is a deterministic 128-identity pilot, selected from the first
immutable request shard before any full generation. It passes only if:

- at least 95% of outputs parse into exactly one nonempty FAULT and REVISION;
- at least 90% of extracted revisions pass the original answer/code verifier;
- no more than 2% mention a provided/given/verified reference;
- no FAULT block contains a boxed answer;
- at most 10% exhaust 1,024 generated tokens; and
- all source, model, adapter, request, token, memory, and output hashes are
  complete.

Failure closes this exact teacher format before full corpus generation. A pass
allows eight disjoint H100 generation shards, followed by an independent CPU
merge/audit. It does not by itself authorize a capability fit.

## Conditional capability gate

Only after the full corpus passes the same quality conditions may one
treatment and one deterministic within-task/length-stratum shuffled-fault
control train from the same B1 warm start for 256 updates. Both use identical
complete revision text and the same global target-token multiset; only the
source-to-fault assignment differs. Development remains the existing 1,289
source-disjoint identities. Holdout remains sealed.

Treatment promotion requires all of:

- at least `603/1,289` overall;
- at least `+10` answers over the matched shuffled-fault control;
- math at least `223`, logic/science at least `349`, and code at least `17`;
- at least 95% strict two-block parse coverage and zero evaluator leakage; and
- complete matched update, token, FLOP, latency, truncation, and protected-hash
  receipts.

Any miss closes exact VFR1 without teacher-prompt, temperature, seed, update,
rank, context, parser, or threshold rescue.
