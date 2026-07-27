# R12 ER Dual-Stream Opcode Route: Paused Development Handoff

**Paused:** 2026-07-21 03:58 EDT
**Worktree:** `/Users/sairamen/projects/shohin-er-ordinal`
**Branch:** `codex/er-tt-ordinal-route`
**Current HEAD:** `7d1b936bde617ac61ffc9bb741ecbe51a61f1003`
**State:** dirty, deliberately uncommitted, pre-freeze, pre-seed, no H100 job
**Resume instruction:** read this file and the latest `AGENT_RUNBOOK.md` journal before editing.

## 1. Scientific objective

The active work is a narrow compiler-mechanism qualification, not a claim of
general reasoning. Rejected ordinal-route fresh v1 showed that the inherited
v1.2 route lattice was exact on opcode-first layouts and cardinalities five and
six, but collapsed near 52% when the opcode occupied the middle candidate and
for cardinalities three and four. Retained evidence localized the defect: the
path objective scored retained witnesses but never positively scored the
excluded candidate as the opcode.

The current factorial is intended to separate:

1. coherent path MAP hardening from independent witness-slot hardening;
2. acute inference-time opcode evidence;
3. another equal-budget legacy multitask fit;
4. opcode-coupled fitting;
5. structured exclusion-path NLL fitting; and
6. interactions between training and inference-time opcode scoring.

The executor remains fixed host tensor algebra. A pass would qualify only the
compiler route on already-consumed training semantics and authorize a separate
fresh-board experiment. It would not establish learned execution, autonomous
halt, planning, arithmetic, open-language transfer, or general reasoning.

## 2. Durable evidence and custody

- Closed fresh-v1 development is permanently `1/0`; never rescore it and never
  open its confirmation.
- Fresh-v1 result: route joint `1558/2048 = 76.074%`, state `85.400%`, answer
  `90.527%`, complete witness pointers `50.000%`.
- Qualified parent v1.2 checkpoint is locally available at
  `train/er_dual_stream_ordinal_1790361034717866861/compiler.pt`.
- Qualified parent seed is `1790361034717866861`.
- Parent split hashes are fit `70e84d6c...` and probe `775925da...`.
- The current code now reuses the exact parent split and verifies it against the
  checkpoint, so the probe is held out from parent and current refitting.
- Commit `7d1b936...` and CPU job `695028` proved real-parent reconstruction,
  exact parameters `185,532,296 / 11,129,504 / 14,467,704`, finite structured
  gradients, and frozen-state stability. It is only a wiring result and is
  scientifically superseded before any seed.
- No v1.5 source commit, derived seed, H100 job, development read, confirmation
  read, or score exists.

Newton paths required for a future qualification/launch are recorded in
`AGENT_RUNBOOK.md`. Do not print credentials or `.env` contents.

## 3. Current modified files

Tracked modifications:

- `AGENT_RUNBOOK.md`
- `FRONTIER_AGENT_PLANS_ANALYSIS.md`
- `R12_ER_DUAL_STREAM_OPCODE_COUPLED_PREREG.md`
- `SHOHIN_NATIVE_REASONING_MASTER.md`
- `train/assess_er_dual_stream_opcode_canary.py`
- `train/jobs/er_dual_stream_opcode_canary.sbatch`
- `train/pilot_er_dual_stream_fresh.py`
- `train/pilot_er_dual_stream_opcode_canary.py`
- `train/test_er_dual_stream_opcode_canary.py`

Existing untracked artifacts are not part of this patch and must not be staged,
deleted, or rewritten:

- `train/er_dual_stream_fresh_score_5499768532556522119/`
- `train/er_dual_stream_ordinal_1790361034717866861/`

Do not edit or reset the shared worktree `/Users/sairamen/projects/shohin`.

## 4. Repairs already implemented but not frozen

### Evaluation semantics

- `evaluate_coherent_routes` refuses a model in training mode.
- Every fitted arm is put in `eval()` before evidence generation.
- A regression test covers the train-mode refusal.

### Split identity

- `QUALIFIED_CANARY_SEED = 1790361034717866861` is fixed.
- Producer and assessor reconstruct the original parent split with
  `derived_seed(QUALIFIED_CANARY_SEED, "dual-stream-train-probe-split")`.
- Producer requires exact equality with the split stored in the parent
  checkpoint.

### Target-blind route decoding

- Target exclusion is retained only for scoring.
- Prediction is gated by model-owned active-rule/cardinality fields rather than
  `row.rule_count`.
- Audit this again before freeze; no gold target field may choose a predicted
  route, relation, event, halt, query, or answer.

### Query grounding

- Same-program query-target counterfactuals change only the queried numeral and
  its expected span/answer role.
- Two offsets are evaluated for each raw/structural query mode.
- Raw query semantic and pointer logits are retained and independently checked.
- Query gating follows the selected route's S0/S1 branch.
- Program route and query diagnosis remain separate.

### Nondegenerate source control

- The degenerate all-`z00000` source-free control was replaced for advancement
  by `identity_deranged`, which gives each neutral occurrence a distinct
  same-width token.
- Gate name is `identity_deranged_joint_max`.
- Historical documents may still mention source-free results; new v1.5 prose
  and code must consistently describe the nondegenerate identity control.

### Path and intervention evidence

- Canonical and relocated path evidence are checked.
- Expected active-route count is derived from rebuilt rows.
- Retained target cardinality/rule count must match rebuilt rows.
- Recomputed path argmax must equal the selected exclusion.
- Direct path-derived marginal reconstruction is retained.
- Query logit evidence and two query-target counterfactuals are retained.

### Safe loading and source closure

- Producer-side parent loading now uses `weights_only=True`.
- Assessor checkpoint/evidence loading uses `weights_only=True`.
- `train/pilot_er_dual_stream_fresh.py` was added to the frozen-source set.
- Imports for full model reconstruction were started in the assessor, but that
  implementation is unfinished; the resulting unused imports are intentional
  evidence of the paused state, not a lint-clean result.

## 5. Launch-blocking scientific defects

### A. Checkpoint is not yet bound to retained evidence

The assessor validates copied receipts and trainable-state digests but still
does not reconstruct each complete model, strict-load the arm's trainable
state, recount parameters, and reproduce retained model outputs. A malicious or
buggy producer could provide self-consistent oracle evidence beside unrelated
weights. This is the highest-priority unfinished implementation.

Required repair:

1. Add assessor arguments for repo root and all parent checkpoints/assessments.
2. Verify current clean Git HEAD and every frozen source hash against the source
   manifest.
3. Safely load the qualified parent and reconstruct every arm with
   `initialize_system`.
4. Verify exact parameter certificate, parent/initial/final digests, and strict
   trainable-state loading.
5. Put each model in evaluation mode.
6. Deterministically rescore model-owned anchors on all canonical and relocated
   rows and compare raw logits, pointers, hard fields, selected records, and
   route outputs with retained evidence.
7. Independently derive paths, marginals, relations, event rebinding, state,
   and metrics from those anchors.
8. Release each model before reconstructing the next arm.

The Slurm assessor invocation must pass the same immutable parent paths and
exact `CODE_ROOT`/source commit.

The source-manifest validator is also currently self-attested: it verifies the
shape and internal hash of the supplied manifest, but not that the named commit
and runtime bytes are real. An invented digest map can pass. Runtime Git and
per-file verification are mandatory parts of this repair. If independent
assessment must not import producer initialization code, move reconstruction
into a small frozen shared factory or duplicate the minimal reconstruction in
the assessor; do not retain a false “imports no producer code” claim.

### B. Diagnosis predicates overlap

The current exact-one policy can falsely reject when more than one real
mechanism works. Example: zero-update S0 `.70`, zero-update S1 `1.00`, and
legacy S0 `1.00` makes both acute-opcode and additional-training diagnoses
true. Similar overlaps exist for decoder plus learned/structured and acute plus
learned.

Do not repair this with priority ordering that hides coexistence. Replace the
single-label gate with a causal contrast table or lattice that reports every
supported effect and selects the minimal sufficient intervention using explicit
nested contrasts. Advancement should require a uniquely justified minimal
mechanism, while simultaneous higher-order successes remain reportable rather
than automatically invalid.

### C. “Additional marginal training” is overclaimed

The legacy arm trains the full multitask compiler objective, not only the
marginal witness term. If it succeeds, the valid statement is “additional
legacy multitask optimization repaired the route.” Attributing the effect to
marginal witness supervision requires a matched loss ablation. Rename the
diagnosis now, or add an equal-budget arm that isolates the witness objective.

### D. Path controls need full downstream reconstruction

The assessor now checks path partitions, complements, marginals, and selected
exclusion, but must also independently derive and compare relations, active
events, event opcode pointers, halt, and state for every active controlled
route. The checkpoint-rescore work should make producer-provided transformed
predictions non-authoritative.

Three additional fail-closed checks are required:

- `cardinality_probability` must be finite, nonnegative, normalized within a
  frozen tolerance, and consistent with the model-owned predicted cardinality;
- the expected rule-opcode position must be derived from the semantic rule line
  and opcode token, not defined circularly as whichever candidate is not a
  witness; and
- candidate positions and unaffected model fields must match across matched
  decoder/control branches while scored `pred_rule_opcode_pointer` and
  `pred_event_opcode_pointer` are compared to independently derived targets.

Without these, an all-zero cardinality posterior can manufacture an independent
decoder failure, and an event occurrence can be relabeled as the expected rule
opcode.

### E. Query evidence is not yet fail-closed

The assessor must require every declared query branch and both target
counterfactuals, rather than skipping missing logits. It must validate the
producer's `query_structural_routing` arm metadata and bind `qraw`/`qstruct`
labels to reconstructed model settings. Swapping those artifacts must fail.

### F. Runtime and artifact design are operationally unsafe

The current workload is approximately:

- four arms, three fitted arms, 7,500 optimizer updates;
- 1.28 million row evaluations;
- about 50,000 source-encoder batch passes;
- about 5.45 GiB retained evidence and 5.63 GiB total output.

Memory is safe under 96 GiB RAM and H100 VRAM, but the combined producer and
assessor can take 2.5 to more than 5 hours. The current four-hour allocation is
not safe. The assessor also repeatedly converts entire 8,000-row tensors to
float inside per-route loops, creating an estimated 14.8 PiB of avoidable
memory traffic.

Required source-preserving optimization order:

1. Slice first, then convert only the small fp16 slice to float.
2. Hoist control tensor references/conversions outside row/rule loops.
3. Vectorize route reconstruction by cardinality with fixed gather tables.
4. Split training/evidence production from assessment; run assessment as an
   `afterok` CPU job so it does not retain the H100.
5. Prefer hash-manifested evidence shards by arm/mode over one 5.45 GiB pickle.
6. Losslessly narrow integer evidence to int16/int8/bool.
7. Keep all rows, controls, updates, and scientific thresholds unchanged.

Any performance rewrite requires bit-exact metric and evidence-equivalence
tests against the reference implementation before deletion of the reference.

### G. Documentation is stale relative to code

The preregistration still describes five explanations, source-free collapse,
exactly-one diagnosis, and an older query contract. Rewrite it only after A-E
are settled. Bump schemas from v1.5 to v1.6 if semantics or artifact layout
change; do not silently reuse v1.5 identifiers.

## 6. Current verification snapshot

At pause time:

```text
PYTHONPATH=train:pipeline python3 -m pytest -q \
  train/test_er_dual_stream_opcode_canary.py \
  train/test_er_dual_stream_relation_adapter.py

22 passed in 5.44s
```

Static verification is deliberately not clean because the model-reconstruction
implementation is incomplete. Ruff reports seven unused assessor imports:

- `subprocess`
- `load_trainable_state`
- `trainable_state`
- `byte_batch`
- `_load_canary`
- `initialize_system`
- `release_cuda`

Do not merely delete these imports. Implement the missing independent
reconstruction first, then rerun lint.

The latest broader 56-test pass predates the most recent repairs. It must be
rerun. The known absent-large-artifact integration fixtures should be exercised
in a clean Newton capsule, not bypassed locally.

## 7. Exact restart order

1. Confirm this branch/worktree and inspect `git status --short` without
   touching the two untracked artifact directories.
2. Read `AGENT_RUNBOOK.md`, this handoff, the preregistration, and the two
   Frontier plan documents.
3. Finish checkpoint-to-evidence reconstruction and runtime source verification
   in the independent assessor.
4. Replace overlapping exact-one diagnoses with a preregistered minimal causal
   contrast scheme; rename the legacy multitask effect honestly or add a loss
   ablation.
5. Complete downstream control reconstruction, probability-simplex checks,
   semantic opcode grounding, and fail-closed query branch validation.
6. Optimize assessor conversion/vectorization without changing evidence or
   gates; split GPU producer and dependent CPU assessor jobs.
7. Update schemas, preregistration, runbook, Frontier analysis, and master
   reasoning document to match exact code semantics.
8. Add adversarial tests for:
   - unrelated weights plus self-consistent oracle evidence;
   - dirty/wrong source commit and altered source file;
   - strict state-load failure and parameter-count mismatch;
   - overlapping diagnoses;
   - legacy multitask versus witness-loss attribution;
   - missing rows/routes and altered downstream event/state predictions;
   - zero/NaN/non-normalized cardinality posteriors;
   - an event occurrence substituted for a rule opcode;
   - query-position shortcut and query-logit corruption;
   - omitted or relabeled `qraw`/`qstruct` evidence;
   - evidence-shard substitution, omission, and reordering;
   - reference versus optimized assessor equivalence.
9. Run focused tests, the full 56+ regression set, Ruff, `py_compile`, `bash -n`,
   and `git diff --check`.
10. Run exact real-parent CPU qualification in a clean Newton capsule.
11. Obtain a final adversarial protocol review with no launch blockers.
12. Only then commit/push exact repaired source, derive one post-commit seed,
    run `sbatch --test-only`, and submit one train-only H100 producer job plus a
    dependent CPU assessor job.

Do not open development or confirmation, do not reuse a prior seed, do not
launch from a dirty worktree, and do not treat a wiring pass as a capability
result.

## 8. Frontier-plan guidance retained

The useful synthesis from `FRONTIER_AGENT_PLANS.md` and
`FRONTIER_AGENT_PLANS_ANALYSIS.md` remains:

- ground compilation in source-visible pointers;
- keep compiler, executor, halt, and consumer responsibilities separable;
- delete source before claiming execution;
- use matched controls and component gates before combining mechanisms;
- prefer a small, falsifiable causal change over a bundled architecture story.

Do not add Hopfield memory, VQ, SSM recurrence, RL, checksums, host scheduling,
or a larger latent workspace to this experiment. Those are separate hypotheses
and would destroy attribution. ER-TT is still only a compiler qualification;
the fixed tensor motor is an algebraic ceiling, not a native reasoning claim.
