# PCF1: Ministral Publication Confirmation

Status: prospectively frozen before any PCF1 data mutation, model load, or
compute on 2026-08-11. This is the only authorized confirmation in this
phase. A completed gate, pass or fail, ends the phase.

## Purpose and immutable boundary

PCF1 tests only the transferable content of the qualified dense Shohin
release:

```text
source -> model-owned draft -> trained same-family revision
       -> learned whole-trajectory commit -> one coherent answer
```

The positive control is the immutable Qwen3.5-9B release whose protected
product result is learned commit `383/538`, trained revision `374/538`, and
unchanged continuation `316/538`. PCF1 does not modify or retest that release.
It does not reopen NDR1, KCR1, VTE1, any natural-language-to-microcode bridge,
the Qwen3.6-35B-A3B edit cascade, a small-OLMoE route, or any closed prompt,
seed, rank, layer, duration, decoding, threshold, or data-mix variant.

The sole new host is the cached dense language path of
`mistralai/Ministral-3-8B-Reasoning-2512` at exact revision
`81eaece1948f3875421d9a45bc55487d10e2d894`. The snapshot must receive a
complete verification of its pre-existing immutable 58-entry manifest,
SHA-256
`46cc9203a18a414e08a53109662c3802b57c046896185ca9ab31875e8167cf1f`,
and pass one no-score mechanics/admission replay before the capability graph
may start. The manifest was created by completed download job `747023`, covers
35,706,515,534 bytes, and verified read-only before this implementation used
the model. No alternate host is a contingency.
Its pinned configuration is the multimodal
`Mistral3ForConditionalGeneration` wrapper with a dense 34-layer
`Ministral3Config` language path; every PCF1 program therefore requires the
explicit established `multimodal` loader and never falls back to a causal
auto-class. Its exact final-four trainable language-layer indices are
`[30, 31, 32, 33]`. The execution environment is likewise pinned to the
existing `product-reasoning-b3a3603-r2` receipt: Python `3.13.13`, Torch
`2.6.0+cu124`, Transformers `5.15.0.dev0`, runtime receipt SHA-256
`277b97fbd6b18760c9789cf3f3372bdb6b40ca87bf84a1df4b41ee3194c4e9dd`,
and package freeze SHA-256
`1d4dfd4a1dc11af9788b0bab072d262278db1814d3fca49465d4df5931b3b87a`.
Every generated MBPP program, including training-only calibration programs,
must execute behind the same pinned Bubblewrap OS-isolation boundary. The
candidate sees only its anonymous read-only program, a read-only minimal
Python runtime, private `/proc` and `/dev`, and an empty temporary filesystem;
it receives a clean environment, no network namespace, no parent process, and
no campaign, source, assessor, model, evidence, or user filesystem. The
Bubblewrap binary is `/usr/bin/bwrap` 0.4.0 with SHA-256
`eb767688b8224d8d3dbe1f8cb30ac3dff9ae8b02ff0452eaec9f94874d4e0011`.
The no-score mechanics gate and every code-scoring allocation must reproduce
a hash-bound escape probe before executing any generated program.
The earlier decision to cancel an unused dense Ministral detour was a priority
decision made during the MoE phase, not a scientific result. The 2026-08-11
publication directive supersedes that priority decision for PCF1 alone.

At inference the runtime sees only the problem and model-owned trajectories.
It receives no target, correctness bit, task label, benchmark route, solver,
verifier, tool, retrieval result, or external proposal model. Every commit
selects one complete candidate byte-for-byte; it never splices fields.

## Frozen lineage and data custody

The draft role is a source-only final-four-token-mixer rank-8 LoRA trained for
256 AdamW updates on the existing verified B1 corpus, SHA-256
`2461d6f70b44a142854d56c24e1fb42d600065e5788a2c4e055ba47b12696549`.
It uses the qualified B1 settings: maximum sequence 1,024, batch one,
accumulation 16, learning rate `2e-4`, alpha 16, model seed `2026080711`, and
data seed `20260802`.

The frozen source universe is the existing CVG1 pair bank and three verified,
evaluation-disjoint source banks:

- pairs: `45f1d66ce5e87dc2a1f4c3594bdde2bae26e9417e879d16eb4eddb228b696afe`;
- math: `e0ede83257e441050a019f59fb13d9c85bd6cba1d6a755ab86fb7129966ddbe5`;
- logic/science: `5a96859fd9088cde598b61da60dd2c6cb7281323ee06c034742a1b4e0e237017`;
- executable MBPP: `0b6d068b4d71f407cb234579b9278dc640df09139ea906dd0f52a6ab71e05398`.

NUL-delimited source identity and split seed `2026080811` remain unchanged.
A CPU-only custodian freeze writes a new nonsealed model-input root containing
only train and development source identities. It writes the 1,289 development
assessors as a distinct write-once CPU-only artifact outside that root; no
confirmation runtime row or GPU report contains an assessor, target,
correctness field, execution result, or per-arm score. The custodian records
only the count and sorted-identity digest of the existing 1,279-row holdout;
it must not copy holdout questions, answers, assessor fields, or prior model
trajectories. The protected 538-row product is not an input to any PCF1
program. After the exact hash-pinned CPU source freeze (whose historical bank
directory happens to be named `product_reasoning`), every model-visible data
path, output path, schema, and requested split fails closed if it contains
`holdout`, `product`, or `public`.
The exact B1 bytes and the unmodified qualified generic trainer receipt retain
their historical control-plane schema names. They are admitted only behind
the pinned B1 hash in a safe path; schema metadata is never a runtime field,
and no PCF1 evaluation, candidate, selection, or result uses a legacy schema.

The draft adapter generates one greedy, no-thinking, at-most-768-token draft
with seed `2026080818`, batch size two, and exactly 16 ordered shards for every
nonsealed identity. The revision adapter
warm-starts from that exact draft checkpoint and receives `source + exact
draft`. Its 5,824 unique training identities retain the qualified presentation
geometry: single-correct outcomes appear four times and both-correct or
both-wrong outcomes once, for exactly 9,655 presentations. It trains for 256
updates with batch one, accumulation eight, context 4,096, learning rate
`2e-5`, final-four-token-mixer rank 8, alpha 16, model seed `2026080815`, and
data seed `2026080814`. Verified complete targets are supervisor-only.

Commit calibration uses only training identities and candidate outputs from
the frozen trained revision and unchanged continuation. The learned commit is
the simpler independent shared-candidate scorer already qualified as the
matched AQC control; antisymmetry is not a PCF1 claim. Calibration is
source-disjoint from the 1,289-row confirmation board. The commit host is the
same pinned Ministral backbone plus the frozen draft adapter. Candidate order
is presented both ways and must be exactly consistent. The qualified
independent-control recipe remains exact: 80/20 calibration split seed
`2026080820`, training seed `2026080822`, 128 updates, accumulation eight,
maximum sequence 3,072, head width 512, backbone LR `2e-6`, head LR `2e-4`,
and tie-loss weight `0.25`.

## Matched arms

All capability arms use the same 1,289 source identities, exact model-owned
drafts, source-plus-draft serialization, maximum 768 final tokens, scorer,
greedy evaluation seed `2026080816`, batch size two, and exactly four ordered
shards per arm. Each of the two training-only commit-candidate calibration
arms, trained revision and unchanged continuation, also uses four ordered
shards; self-refinement is a confirmation control and is not an extra
calibration experiment.
They record
prompt/generated tokens, wall time, peak CUDA,
trainables, and charged GPU-seconds.

1. **Trained revision** — the frozen warm-started revision adapter.
2. **Unchanged continuation** — the frozen draft adapter reads the exact
   treatment prompt and receives the same generation budget.
3. **Self-refinement** — that same draft adapter reads the exact draft under
   one frozen generic review-and-correct instruction and the same budget.
4. **Learned coherent commit** — the frozen independent scorer chooses the
   complete trained-revision or unchanged-continuation trajectory.

There is no long-generation, best-of-K, router, hidden-draft, transaction,
edit, microcode, or alternate-commit arm. An infrastructure failure is
preserved as terminal infrastructure evidence; it is never counted as a wrong
answer and does not authorize a shard replay, retry, requeue, or successor.

## The single falsifiable gate

The 1,289-row development confirmation opens and scores exactly once, in one
CPU process, after all three generated arms, the learned-commit application,
their candidate hashes, scheduler accounting, and the pre-score custody
authorization are immutable. That process is the sole reader of the separate
assessor artifact and scores revision, unchanged, self-refinement, and the
selected whole trajectory atomically. PCF1 passes only if every Boolean below
is true:

1. **Capable host:** unchanged continuation solves at least `387/1289`
   (30.0%) and solves at least one math, one logic/science, and one executable
   code identity.
2. **Causal revision margin:** trained revision solves at least 65 more
   identities than unchanged continuation and at least 39 more than
   self-refinement.
3. **Revision retention:** trained revision has nonnegative math,
   logic/science, and executable-code correct-count deltas against both
   matched controls.
4. **Useful learned commitment:** coherent commit solves at least 13 more
   identities than trained revision.
5. **Conservative commitment:** commit retains at least 95% of the
   revision-correct identities and at least 95% of the unchanged-correct
   identities, and has nonnegative per-domain correct-count deltas against
   revision.
6. **Complete custody:** every arm covers the same `1289/1289` identities;
   candidate and selection outputs are nonempty and well formed; candidate
   assessment truncation and malformed selections are zero; A/B semantic
   order consistency is `1289/1289`; all pinned model, data, runtime,
   execution-environment, code-sandbox/probe, checkpoint, result, and compute
   hashes verify; and the holdout, public, and product access counters remain
   zero.

`PASS = capable_host AND causal_revision_margin AND revision_retention AND
useful_learned_commitment AND conservative_commitment AND complete_custody`.

Any false conjunct is the final PCF1 result. It closes exact PCF1 without a
model, source, schedule, prompt, selector, threshold, seed, or decoding retry.
A pass records that the separately sealed holdout and product may be
considered by a later explicit authorization; PCF1 itself does not open them.

## Execution and stop rule

1. Freeze and test the CPU source firewall, runtime, model manifest, and all
   dispatch receipts without reading a score.
2. After durable storage is within a verified safe quota, run one
   24-presentation mechanics/admission job. It may validate loading,
   gradients, adapter restoration, serialization, masking, commit order, and
   the code-sandbox escape boundary, but may not emit a capability statistic.
3. Run the one dependency-bound graph: draft fit, nonsealed draft collection,
   revision fit, training-only commit calibration, the three label-free
   matched development arms, learned commit application, pre-score custody,
   one atomic CPU score, and the gate.
4. Mirror reports and compute receipts, verify hashes independently, update
   the master/status/runbook ledgers, and stop. Do not submit a holdout,
   product, public, alternate-host, or successor job.

At freeze time Newton had no queued/running jobs and sufficient idle H100s,
but the account reported approximately 612 GiB and 1.326 million inodes over
hard Lustre quota. Therefore no PCF1 remote write or job is authorized until
storage safety is resolved without deleting scientific evidence.
