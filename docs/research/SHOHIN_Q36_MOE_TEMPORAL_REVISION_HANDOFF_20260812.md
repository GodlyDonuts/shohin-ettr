# Q36-MTR: MoE Temporal-Revision Handoff

Status: exactly-once execution handoff after PCF17 terminal-null, 2026-08-13.
The pinned Q36-MTR host, source-disjoint data view, and one scheduler graph are
authorized for a single execution. The 61 single-H100 requests, 58.90 expected
H100-hours, arms, prompts, seeds, thresholds, and terminal stop are immutable.
No retry, confirmation, or successor is authorized. PCF17 remains closed and
unscored.

Local preparation now includes `pipeline/q36_mtr_contract.py`, which emits and
revalidates the exact single-execution 61-request dependency graph, and
`pipeline/compare_q36_mtr.py`, which implements the five-arm terminal reducer.
The reducer hash-binds the graph, all arm reports, final custody, exact host,
source/data/runtime identity, one assessor read, scheduler accounting, and the
evidence mirror. It always stops after development and never authorizes an
automatic confirmation. `pipeline/dispatch_q36_mtr.py` is the sole dispatcher;
its literal acknowledgement is `ONE_FROZEN_DEVELOPMENT_GATE_ONLY`.

The role-mechanics milestone is also implemented prospectively. The exact
Q36 source owner, aligned reviser, and draft-hidden reviser contract lives in
`train/q36_mtr_roles.py`; `train/hf_q36_mtr_train_role.py` trains only the
declared final-16 rank-18 shared post-MLP residual and explicitly supplies
full-sequence position IDs before the hidden arm masks draft attention.
`train/hf_q36_mtr_generate_drafts.py` assigns all 7,113 nonsealed identities
to 16 deterministic owner-draft shards. The matching Slurm wrappers request
one H100, disable requeue, retain the exact exclusion set, and require a
hash-bound phase-authorization receipt. `pipeline/compile_q36_mtr_plan.py`
expands the frozen graph into 61 unique single-H100 request records and permits
exactly one dispatcher-bound execution while still rejecting model acquisition.

The data/evaluation plumbing is executable only through that frozen graph.
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

Owner-draft bytes now follow the exact dense lineage boundary. Generation and
the 16-shard merge preserve the tokenizer's decoded completion verbatim,
including outer whitespace and its generated-token accounting. Materialization
alone derives the model-visible draft with the dense recipe's deterministic
Unicode outer-whitespace strip. Every identity carries separate raw and
canonical SHA-256 receipts; independent precompute custody replays the raw
merged draft into all 9,655 revision presentations and both evaluation views,
reconstructs the exact revision prompt, and rejects normalization drift. Raw
draft text never becomes an additional model runtime field.

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

Host admission is exact rather than family-level. The pinned outer config is
`qwen3_5_moe`; the nested causal text config is `qwen3_5_moe_text` with 40
language layers, hidden size 2,048, 256 routed experts, top-8 selection,
512-wide routed and shared experts, and vocabulary size 248,320. The layer
schedule is exactly three linear-attention layers followed by one
full-attention layer, repeated ten times. Transformers' pinned mapping resolves
the outer config through the explicit causal loader to
`Qwen3_5MoeForCausalLM`; the final-16 residuals therefore attach only to layers
24 through 39. Mechanics, every role report, evaluation restore, native-router
receipt, and independent custody all reject any different class, layer count,
expert count, top-k, layer schedule, or controlled index. Synthetic 64-layer
geometry is not an admissible stand-in for this host.

The causal boundary is now executable rather than metadata-only. Role reports
separately bind draft-token byte presence and draft-information availability:
the aligned and draft-hidden fits retain identical prompt/response tokens and
absolute positions, while only the aligned fit exposes draft keys to causal
attention. The no-score mechanics allocation perturbs every draft token while
holding weights, response targets, sequence length, and positions fixed; it
requires a material change in aligned target-facing states and invariance in
the hidden states within a frozen BF16 tolerance. Generation state is also
bound to the exact prepared prompt IDs, preventing a same-shape prompt from
reusing stale hidden-mask or position state. The span parser recognizes the
frozen task-agnostic `Follow the original problem's requested output format.`
boundary as well as the legacy task-specific `Return ...` boundary, and a
regression proves that only the exact internal-draft bytes are masked. This
prevents the Q36 aligned/hidden tokenizer from failing before training while
leaving the frozen prompt text unchanged. Precompute custody replays the
matched sequence receipts and enforces exact checkpoint isolation: aligned
only for revision, source owner only for unchanged/self-refinement, and the
independently trained hidden checkpoint only for draft-hidden. This also fixes
a prospective custody mismatch that previously expected the aligned checkpoint
for the two owner controls and would have rejected an otherwise valid run.
The same counterfactual now requests the pinned host's native router logits at
every MoE layer. Aligned draft changes must produce a material target-facing
router-logit change; hidden changes must remain within the frozen numerical
tolerance and preserve every target-facing top-k assignment. Any aligned top-k
expert change is recorded explicitly but is observational, not required or
silently promoted into a stronger routing claim. Router/expert weights remain
frozen and byte-receipted throughout.

The environment/runtime closure is executable and exact-commit bound.
`pipeline/package_q36_mtr_runtime.py` packages a sorted, exact-membership
runtime with the sole dispatcher and no model-acquisition capability;
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
The commit input projection contains exactly the problem text and the two
complete candidate completions. Identity, task, lineage, correctness,
generation-token counts, exhaustion flags, and assessor metadata are excluded.
Commit fitting, calibration-development validation, and label-free development
application all call that same Q36-owned projection; no generic legacy
evaluation helper can substitute a broader view. Reports and custody bind the
`question_plus_complete_candidate_only_v1` contract and exact three visible
fields.
Its shared scalar scorer is antisymmetric by construction under A/B reversal.
The checkpoint and downstream custody additionally require a nonzero finite
FP32 adapter-state delta, exact final adapter/head state hashes, and immediate
serialization restore. Before label-free development application, the live
adapter and head are zeroed and restored from those written checkpoint bytes;
the restored state hashes must match exactly. A separate receipt requires a finite nonzero task
gradient on the residual surface at every one of the 128 commit updates, so
AdamW weight decay alone cannot satisfy the learned-adaptation claim. The
source aligned-checkpoint file remains byte-exact
even though its loaded residual state is deliberately updated.
The graph's CPU `commit_apply` stage is an independent label-free validator,
not a second model load: it replays the checkpoint, pair, selection,
antisymmetry, truncation/malformed, visibility, and zero-assessor-access
receipts before precompute custody can begin. The compiled execution plan names
an exact existing wrapper for every H100 and CPU stage, closing the handoff
ambiguity without another submit path.

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

Exact execution accounting and durable evidence preservation are now part of
that prospective boundary. `pipeline/capture_q36_mtr_accounting.py` replays
the frozen plan and dispatch receipt against `sacct`, requires every one of
the 61 named single-H100 allocations (including every array cell), and rejects
wrong partitions, GPUs, or nodes; failed exits; restarts; reused job IDs;
missing cells; duplicates; and orphans. Prescore accounting is an explicit
hash-bound input to score authorization. `pipeline/mirror_q36_mtr_evidence.py`
copies and rehashes a fresh, nonwritable snapshot of every irreplaceable
checkpoint, merged draft/candidate file, commit pair/selection, source/data
report, environment/model/runtime manifest, score artifact, normalized arm,
and accounting record without copying or opening the assessor board. The
final comparison CPU job synchronously seals its own `PASS`/`FAIL`, final
custody, graph, and preterminal mirror into the authorized evidence root; it
does not submit an evidence successor. The CPU-only `evidence_mirror`
dependency now sits between final accounting and compute custody, leaving the
scientific 61-request/58.90-H100-hour contract unchanged.

The PCF17 infrastructure finding is also repaired prospectively: exact-tree
custody accepts either canonical manifest members or the single conventional
`./` prefix emitted by `find .`, canonicalizing before comparison. Interior
dot segments, traversal, absolute paths, duplicate canonical names, symlinks,
extra/missing files, and hash drift remain rejected. This repair cannot be
used to resume or reinterpret PCF17.

The final phase-admission transaction and fail-closed submission path are now
implemented. `pipeline/capture_q36_mtr_cluster_preflight.py` requires an
empty user queue, at least 128 GiB and 150,000 inodes of durable Lustre
headroom, at least one eligible non-excluded H100 node, and enough remaining
H100-hour budget for the exact 58.90-hour plan. After exact repository,
private-remote, runtime, model, environment, sandbox, and source-hash checks,
`pipeline/authorize_q36_mtr_phase.py` can mint one write-once authorization
for one fresh run root. `pipeline/jobs/q36_mtr_prepare_phase.sh` performs only
this receipt publication. The dispatcher holds the root, prestages all 33
dependency roots, publishes one immutable dispatch receipt, and only then
releases the root; any failure cancels every submitted dependency. The graph's
first CPU allocation then runs
`pipeline/validate_q36_mtr_live_preflight.py`, which rechecks live quota,
eligible H100 capacity, and accounting before any scientific row can be read;
that receipt is mandatory precompute custody and durable-mirror evidence.
Preparation and authorization do not launch Q36-MTR. No scientific Q36 job was
submitted while producing this execution handoff.

Publication statistics are frozen before any Q36 outcome exists. The sole
scorer now derives exact paired win/loss/tie tables for revision versus
unchanged, self-refinement, and draft-hidden, plus commit versus revision and
unchanged, both overall and by domain. Each table carries the matched risk
difference, paired Wald 95% interval, and exact two-sided McNemar probability.
These fields are hash-bound through normalization and the terminal result but
are explicitly non-gating: they cannot change any threshold or PASS/FAIL
clause. The same report emits plot-ready dense-9B and Q36 arm percentages with
an immutable warning that the two source-disjoint boards do not authorize a
direct absolute-score comparison or a compute-scaling-law claim. This gives a
publication scaling figure honest effect sizes immediately at the terminal
gate without post-result analytic choice.
Four stronger publication labels are also preregistered independently of the
gate: revision over unchanged, revision over self-refinement, causal draft
visibility, and commit over revision. A label is supported only when its
paired effect is positive, its paired 95% interval excludes zero, every domain
has nonnegative direction, and its exact McNemar probability survives a
familywise `0.05` Holm-Bonferroni correction across all four labels. A gate
`PASS` without that evidence remains a mechanism result, not a significance or
breakthrough claim.
`pipeline/render_q36_mtr_publication_figure.py` can subsequently render only
that sealed terminal payload into a deterministic two-panel SVG, scaling-point
CSV, paired-effect CSV, and SHA-256 manifest. It refuses absent or tampered MoE
statistics and is not wired as a scientific successor, retry, or gate input.

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
qualified Q35 mechanics. The `1,179,648` residual parameters are held as FP32
master weights and executed under BF16 autocast. This is required because the
commit-stage `2e-6` learning rate is below one BF16 representable step at
ordinary adapter magnitudes; BF16 master weights could otherwise turn the
declared joint adapter update into an exact no-op. The first live mechanics
receipt must reprove exact master/compute dtypes as well as exact
trainables, layer indices, frozen router/expert tensors, one finite update,
serialization/restore, and peak memory below one H100.
Each role checkpoint also binds the trainable residual names, dtypes, shapes,
and bytes before and after fitting. Both aligned and draft-hidden revisers must
load the exact owner's final residual-state digest, while each begins with an
independently constructed empty AdamW state (`optimizer_restored=false`). After
role fitting, the saved trainable state is hash-checked, the live residual is
zeroed, and the checkpoint is restored before the role report can complete.
Role checkpoints serialize only the 32 residual tensors and metadata: no
optimizer state and zero router/expert tensors. This makes optimizer carryover
and learned native-MoE checkpoint mutation structurally impossible; evaluation
always reconstructs the hash-pinned frozen host and overlays only that residual.
Role warm starts/restores, draft generation, matched evaluation, and
learned-commit fitting share one Q36-only `weights_only` checkpoint loader. It
rejects every payload except the
exact 32 FP32 residual tensors and role metadata, revalidates the role/config,
constructs the pinned causal NF4 host, copies those tensors with exact
name/shape/dtype checks, and rehashes the resulting live residual state.
All Q36 chat prompts are already native-template-rendered strings and are
therefore tokenized with `add_special_tokens=false` in draft generation,
matched evaluation, prompt accounting, merging, and custody. This exactly
matches role training and prevents tokenizer-dependent BOS/control-token
duplication from changing the aligned/hidden causal geometry.
Adapter decoding also uses an explicit generated-token-only sequence contract.
Because the residual path supplies `inputs_embeds` without bookkeeping
`input_ids`, the frozen Transformers runtime must return only newly generated
token IDs. The no-score mechanics allocation proves this with a one-token
generation whose rendered prompt is wider than one token; the shared generator
then rejects any output with the wrong batch geometry, zero width, or width
above `max_new_tokens`. Draft, evaluation, merge, and precompute-custody
receipts all bind `inputs_embeds_generated_tokens_only_v1`, preventing prompt
placeholders from being decoded as answers or generated answers from being
silently sliced away.

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
has an exact `9,655`-presentation pool under the existing 4x single-correct /
1x both-correct-or-wrong weighting. Matching the surviving dense recipe, each
reviser consumes the first deterministic shuffled `2,048` presentations
(`256` updates x accumulation `8` x batch `1`), rather than claiming that all
`9,655` are traversed. Role custody hashes the exact consumed indices, token
geometry, and draft-mask geometry. Both revisers use LR `2e-5`, seed
`2026080815`, data seed `2026080814`, and maximum sequence 4,096.
The hidden reviser receives the same token/position geometry and loss targets;
only informative draft attention is masked.

Calibration is train-only. It may score revision and unchanged candidates to
fit the 128-update whole-trajectory commit (seed `2026080822`, maximum sequence
3,072). Its 8-way gradient accumulation consumes exactly 1,024 deterministic
calibration presentations. The report and precompute custodian independently
replay the task/outcome-balanced index plan and bind every source index,
identity, task, outcome, and presentation position into one SHA-256 receipt;
both calibration-training and calibration-development projections must retain
both complete trajectories without truncation. Development candidates remain
label-free. One authorized CPU scorer
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
phase run root and copied to the authorized durable evidence root. Durable
custody does not trust the mirror
manifest as a self-assertion: final custody and terminal sealing independently
reopen the snapshot, require exact root/artifact membership, reject symlinks,
extras, writable members, byte-count drift, or hash drift, and bind a canonical
artifact-tree digest. A manifest that survives after any mirrored evidence was
removed or changed cannot satisfy the publication gate. After each immutable
stage, mirror compact artifacts atomically
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
