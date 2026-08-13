# Q36-MTR: MoE Temporal-Revision Handoff

Status: documentation-only handoff after PCF17 terminal-null, 2026-08-12.
No Q36-MTR model, data view, scheduler graph, or score is authorized by this
document. A fresh phase may implement exactly one source-disjoint development
gate after its own admission freeze. PCF17 remains closed and unscored.

Local preparation now includes `pipeline/q36_mtr_contract.py`, which emits and
revalidates the exact no-submit 61-request dependency graph, and
`pipeline/compare_q36_mtr.py`, which implements the five-arm terminal reducer.
The reducer hash-binds the graph, all arm reports, final custody, exact host,
source/data/runtime identity, one assessor read, scheduler accounting, and the
evidence mirror. It always stops after development and never authorizes an
automatic confirmation. These modules are preparation, not a dispatcher.

The role-mechanics milestone is also implemented prospectively. The exact
Q36 source owner, aligned reviser, and draft-hidden reviser contract lives in
`train/q36_mtr_roles.py`; `train/hf_q36_mtr_train_role.py` trains only the
declared final-16 rank-18 shared post-MLP residual and explicitly supplies
full-sequence position IDs before the hidden arm masks draft attention.
`train/hf_q36_mtr_generate_drafts.py` assigns all 7,113 nonsealed identities
to 16 deterministic owner-draft shards. The matching Slurm wrappers request
one H100, disable requeue, retain the exact exclusion set, and require a
future hash-bound phase-authorization receipt. They contain no submission
command. `pipeline/compile_q36_mtr_plan.py` expands the frozen graph into 61
unique single-H100 request records, but is intentionally dry-run-only and
rejects any submit or acquisition authorization. This milestone therefore
closes the model-role ambiguity without launching the MoE experiment.

The data/evaluation plumbing is likewise prospective and no-submit.
`pipeline/merge_q36_mtr_drafts.py` admits only the 16 exact contiguous owner
shards; `pipeline/build_q36_mtr_data.py` creates 9,655 natural-trajectory
revision presentations plus 5,824 train-only calibration and 1,289 label-free
development rows. `train/hf_q36_mtr_evaluate.py` maps the aligned checkpoint
only to revision, the source owner only to unchanged/self-refinement, and the
hidden checkpoint only to draft-hidden. Calibration code scoring remains
inside the qualified Bubblewrap boundary; development emits candidates with
no assessment fields. `pipeline/merge_q36_mtr_evaluations.py` rejects shard
gaps, overlaps, duplicates, reordered identities, hash drift, or labels in a
development candidate. All GPU wrappers are one-H100, no-requeue, exact-node
excluded, authorization-gated jobs; the CPU materialize/merge wrappers have
no GPU request and no dispatch capability. Runtime and model trees are checked
for exact manifest membership before use.

The no-score live admission gate is implemented in
`train/hf_q36_mtr_mechanics.py` and
`train/jobs/q36_mtr_mechanics.sbatch`. It admits only the exact Q36 host,
NF4/BF16 role geometry, 24 deterministic source-only B1 rows, and one H100.
It proves the final-16 rank-18 trainable surface (`1,179,648` parameters),
hashes the exact stored bytes of every protected router/expert tensor before
and after one finite update, checks aligned versus draft-hidden token and
full-position equivalence, and requires byte-exact checkpoint restoration.
It never scores a capability row or reads a development assessor. The
mechanics checkpoint is deliberately tagged with only 24 selected rows, so it
cannot satisfy the 100,000-row source-owner warm-start contract.

The environment/runtime closure is now executable but still no-submit.
`pipeline/package_q36_mtr_runtime.py` packages a sorted, exact-membership
runtime with no dispatcher or model-acquisition capability;
`pipeline/capture_q36_mtr_environment.py` pins the qualified base environment,
`bitsandbytes==0.50.0` manifest
`2201774754fb2e0fdd2208b78d34b803b910d8e34c79a43de49b29d7df3a8355`,
and the Q36 fast-kernel manifest
`dde2adf539302a321afd7322ded3f2f729ac5f96368113a8af82f64efc0b9e8b`
(`flash-linear-attention==0.4.2`, `causal-conv1d==1.6.2.post1`). Every GPU
wrapper replays exact overlay membership and imports only from those roots.
The learned commit path is implemented by
`pipeline/build_q36_mtr_commit_pairs.py` and
`train/hf_q36_mtr_train_commit.py`: calibration pairs are labeled and
development pairs are label-free; one H100 fits the 128-update commit and,
without a second model load, applies it to the already sealed development
pairs. This preserves the 61-request/58.90-H100-hour graph.

The irreversible CPU boundary is now implemented prospectively as well.
`pipeline/build_q36_mtr_custody.py` independently binds the exact model and
runtime trees, source views, owner/aligned/hidden checkpoints and reports,
model-owned drafts, materialized data, both calibration arms, all four
label-free development arms, commit pairs, selections, application, and
environment/mechanics receipts before it can emit a one-shot authorization.
`pipeline/score_q36_mtr.py` qualifies the executable-code sandbox before it
atomically consumes that authorization, then opens the development assessor
board once and scores every generated arm plus the learned whole-trajectory
selection in one process. A post-consumption exception writes a distinct
non-retryable terminal-infrastructure receipt. The pure normalizer and final
comparator turn structurally valid unfavorable observations into the formal
terminal `FAIL`; they reserve infrastructure failure for unreadable,
incomplete, or inconsistent custody. The associated CPU Slurm wrappers are
no-requeue, authorization-gated, and contain no submission command. No
development assessor was opened and no Q36 job was submitted while building
or testing this code.

The PCF17 infrastructure finding is also repaired prospectively: exact-tree
custody accepts either canonical manifest members or the single conventional
`./` prefix emitted by `find .`, canonicalizing before comparison. Interior
dot segments, traversal, absolute paths, duplicate canonical names, symlinks,
extra/missing files, and hash drift remain rejected. This repair cannot be
used to resume or reinterpret PCF17.

## Claim carried forward

The strongest surviving qualified Shohin system is the dense same-family
source -> model-owned draft -> trained revision -> learned whole-trajectory
commit release on `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`:

| arm | exact product score |
|---|---:|
| unchanged second pass | `316/538` |
| trained revision | `374/538` |
| learned whole-trajectory commit | `383/538` |

The MoE question is narrow and falsifiable: can the same role-separated,
model-owned trajectory architecture transfer to one capable MoE while
preserving broad capability, and does informative draft visibility cause the
revision gain?

The host is pinned to
`Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0`.
Every inference and trained role uses this same host. No external proposer,
verifier, answer bit, benchmark router, task label, tool, or second model is
visible at inference.

## Causal boundary already measured

The prior Q35 edit experiments are attribution, not a recipe to retry:

- DSET's aligned final-16 rank-18 post-MLP residual reached `1822/1908`,
  versus draft-hidden `1174/1908`, a `+648` trajectory margin. It established
  that this MoE can learn a strongly draft-causal interface.
- ISET's aligned complete transaction reached `1838/1908`, versus hidden
  `1256/1908` and swapped `3/1908`. It strengthened the visibility/ownership
  result but missed its reliability gate.
- BSOT then changed the `1838/1908` ISET proposal to `1835/1908`, while its
  swapped and hidden commit controls preserved `1838`. The edit-selector
  cascade is therefore closed, not a component of Q36-MTR.

Q36-MTR keeps the useful causal boundary: matched aligned and draft-hidden
revision states, identical source/draft geometry, and equal update/parameter
budgets. It discards the explicit edit language, KEEP/REPLACE selector, and
fixed rewrite cascade. The output is a complete natural-language trajectory;
commit chooses one complete candidate.

## Exact architecture mapping

| dense release role | Q36-MTR role |
|---|---|
| source-only draft adapter | Q36 source-only owner, final-16 rank-18 shared post-MLP residual |
| draft-conditioned revision adapter | separate Q36 aligned reviser, same residual geometry |
| unchanged second pass | frozen Q36 source owner reading the exact aligned draft/prompt |
| self-refinement | frozen Q36 source owner under one task-agnostic review instruction |
| independent draft-hidden control | separately trained equal-budget Q36 reviser with the draft span causally masked, not deleted |
| learned coherent commit | PCF whole-trajectory commit head over complete revision and unchanged candidates, with exact A/B swap consistency |

The owner, aligned reviser, hidden reviser, and commit are distinct role
states. Routers, native experts, embeddings, attention, and language head stay
frozen except for the declared final-16 rank-18 residuals and the bounded
whole-trajectory commit head. NF4 weights with BF16 compute are retained from
qualified Q35 mechanics. The first live mechanics receipt must reprove exact
trainables, layer indices, frozen router/expert tensors, one finite update,
serialization/restore, and peak memory below one H100.

## Data and visibility

Use the PCF source geometry, rebuilt into a phase-owned namespace from the
original hash-pinned banks:

- pairs: `45f1d66ce5e87dc2a1f4c3594bdde2bae26e9417e879d16eb4eddb228b696afe`;
- math: `e0ede83257e441050a019f59fb13d9c85bd6cba1d6a755ab86fb7129966ddbe5`;
- logic/science: `5a96859fd9088cde598b61da60dd2c6cb7281323ee06c034742a1b4e0e237017`;
- executable code: `0b6d068b4d71f407cb234579b9278dc640df09139ea906dd0f52a6ab71e05398`;
- source-only B1 corpus:
  `2461d6f70b44a142854d56c24e1fb42d600065e5788a2c4e055ba47b12696549`.

Freeze with split seed `2026080811`: exactly `5,824` train identities and one
source-disjoint `1,289`-identity broad development board. Generate one Q36
owner draft for all `7,113` identities in 16 ordered shards using seed
`2026080818`, greedy decoding, and 768 new-token maximum. The revision corpus
has exactly `9,655` presentations under the existing 4x single-correct / 1x
both-correct-or-wrong weighting. Both revisers train for exactly 256 updates,
LR `2e-5`, seed `2026080815`, data seed `2026080814`, maximum sequence 4,096.
The hidden reviser receives the same token/position geometry and loss targets;
only informative draft attention is masked.

Calibration is train-only. It may score revision and unchanged candidates to
fit the 128-update whole-trajectory commit (seed `2026080822`, maximum sequence
3,072). Development candidates remain label-free. One authorized CPU scorer
opens the phase-owned assessor board once after every candidate, checkpoint,
runtime, environment, sandbox, scheduler-accounting, and custody hash is
sealed. It atomically scores learned commit, revision, unchanged,
self-refinement, and draft-hidden on identical ordered identities. No holdout
or protected product split is part of this gate.

PCF17 generated candidates are not Q36 training/evaluation data and must not
be reused. Its runtime timings and custody lessons are operational evidence
only.

## One development gate

All clauses are conjunctive; counts use the exact 1,289-row board.

1. Unchanged solves at least `387/1289` and every domain is nonzero.
2. Trained revision exceeds unchanged by at least `65` answers (5 points),
   self-refinement by at least `39` (3 points), and draft-hidden by at least
   `39` (3 points).
3. Revision has nonnegative correct-count deltas versus each of those three
   controls in math, logic/science, and executable code.
4. Learned commit exceeds revision by at least `13` answers (1 point), retains
   at least 95% of revision-correct and unchanged-correct identities, and has
   nonnegative per-domain deltas versus revision and unchanged.
5. Every arm has exact ordered identity coverage, one output per identity,
   zero candidate truncation, zero malformed evidence, and exact A/B order
   consistency. Environment, model, runtime, checkpoint, data, source,
   sandbox, scheduler, and score-consumption custody is complete.
6. Assessor semantic reads equal one; public, holdout, and protected-product
   access equal zero; retries, requeues, duplicate shards, and successors equal
   zero.

Any false scientific clause writes the terminal `FAIL` result. Missing,
tampered, incomplete, or infrastructure-invalid evidence writes a separate
terminal infrastructure receipt. A `PASS`, `FAIL`, or infrastructure terminal
state ends the phase. No rank, layer, seed, prompt, threshold, duration,
quantization, or shard rescue is allowed.

## Single-H100 graph and schedule

Every H100 allocation owns a fixed identity range and a fresh output root.
Jobs are dependency-prestaged so Slurm can admit any ready single independently;
no allocation requests an idle GPU without a bound scientific operation.

| priority | stage | single-H100 requests | expected aggregate H100-hours |
|---:|---|---:|---:|
| 1 | no-score load/train/restore mechanics | 1 | `0.25` |
| 2 | 256-update source-owner fit | 1 | `2.25` |
| 3 | ordered train+development draft generation | 16 | `29.50` |
| 4 | aligned and draft-hidden 256-update fits | 2 | `4.50` |
| 5 | train-only revision/unchanged calibration, four shards each | 8 | `15.60` |
| 6 | development revision/unchanged/self-refinement/hidden, eight shards each | 32 | `6.30` |
| 7 | 128-update whole-trajectory commit fit plus label-free application | 1 | `0.50` |
|  | **total** | **61** | **`58.90`** |

Maximum queued/runnable concurrency is 32 independent single-H100 jobs; the
maximum earlier fan-outs are 16 drafts and 8 calibration workers. With enough
capacity, the dependency-critical wall estimate is 9.6 hours. These are
prospective planning values, not charged results; exact `sacct` GPU-seconds,
prompt/generated tokens, wall time, peak CUDA memory, and trainable counts are
mandatory outputs.

The scheduler exclusion set is exactly
`evc26,evc29,evc31,evc32,evc33,evc37,evc38,evc46` until a fresh no-science
qualification changes it prospectively. Use only `normal`, one
`nvidia_h100_pcie` per request, and `--no-requeue`. As of handoff, August
accounting is `846.141944` H100-hours against the documented 2,000-hour cap;
the development projection leaves `1,094.958056` hours. Lustre usage after
terminal evidence sealing is `850,234,704 / 1,059,061,760 KiB` and
`857,588 / 1,010,000` inodes, leaving `208,827,056 KiB` and `152,412` inodes.
The inode margin is only 2,412 above the 150,000 admission floor. Recheck after
reacquiring the Q36 model/runtime and before creating a run root; do not submit
unless at least 128 GiB and 150,000 inodes remain.

The 32-way development fan-out is the maximum scientific concurrency, not a
capacity reservation. Submit all fixed shards with dependencies, let Slurm
admit them opportunistically, and cancel every dependency-dead or unused job
immediately at the terminal gate. A shard is complete only when its immutable
report proves the exact assigned half-open row range, source/report hashes,
one candidate per identity, and zero foreign identities. Merges reject
missing, overlapping, reordered, or duplicate rows.

If this development gate passes, a later phase may separately freeze a fresh
confirmation. Its compute blueprint is four generative arms x eight
independent single-H100 shards (32 maximum concurrent singles, expected
`6.30` H100-hours), followed by CPU commit application and one atomic scorer.
That confirmation is not authorized here and cannot be submitted
automatically. Its commit fit/application remains one single-H100 operation;
the final assessment is CPU-only and opens the assessor once.

## Reuse, admission, and evidence durability

Reuse these PCF assets after host-generalization and new tests:

- deterministic source freeze/materialization and strict visibility firewall;
- 16-shard owner generation and ordered merge;
- revision/unchanged/self-refinement evaluation and explicit candidate files;
- development-only commit-pair builder, whole-trajectory commit trainer, and
  order-symmetric application;
- Bubblewrap MBPP sandbox and per-allocation qualification;
- pre-score custody, one-shot authorization/consumption, sole scorer,
  normalization, comparator, Slurm accounting, and terminal-failure receipts.

Reuse the immutable 9B release as the dense positive reference and the DSET/
ISET result JSONs as causal priors. Do not reuse Ministral/Qwen9B candidates or
adapters as Q36 model state.

The exact Q36 root remains present after the authorized historical cleanup at
`/lustre/fs1/home/sa305415/shohin/artifacts/external/qwen3.6-35b-a3b-995ad96e`.
The preparation audit reverified its mode-`555` root, exact config hash
`93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99`,
full revision text, and 124-entry manifest hash
`06c9d8d8419244f2d001cb351e164f356718d9d77138e898b13afee35856f56e`.
Formal admission must still replay complete membership and every member hash
before mechanics. A short `995ad96e` label is insufficient custody.

Before submission, the implementation commit must be clean and pushed to the
private GitHub branch; the runtime must contain only bytes from that commit.
Push again at graph-freeze and terminal-result milestones. Never modify code or
runtime under a live graph.

Every irreplaceable checkpoint, merged candidate file, source/data report,
model/runtime/environment manifest, accounting receipt, score authorization,
consumption marker, result, and terminal receipt is hash-manifested in the
phase run root. After each immutable stage, mirror compact artifacts atomically
to a phase-specific read-only local evidence root. Temporary shards may be
removed only after: the merged artifact independently replays exact row
coverage; both primary and mirror SHA-256 manifests match; the downstream
custody report binds that merged hash; and no live job references the shards.
No duplicate writer, duplicate score, unbound GPU allocation, or orphaned
dependency is permitted.

## Explicit prohibitions

Do not reopen the Q35 edit-selector cascade, DSET/GSET/ISET/FRET/RIFT/OCET/
RSOT/BSOT variants, NDR1, KCR1, VTE1, the natural-language microcode bridge,
the Q35 edit cascade, or any small-OLMoE variant. Do not tune on PCF17's
unscored candidates. Do not open public, holdout, or protected-product data.
The next phase either runs this one broad natural-trajectory transfer gate as
frozen or does not run.
