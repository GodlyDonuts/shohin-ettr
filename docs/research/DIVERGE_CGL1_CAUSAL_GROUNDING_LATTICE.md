# DIVERGE-CGL1: Causal Grounding Lattice

Status: data mechanics frozen after the QTE1 development ceiling and before
any CGL1 neural training. QTE1 confirmation and composition remain independent.

## Hypothesis

GTI1 demonstrates that direct role labels are cheaply fit through renderer
lookup. CGL1 removes those labels from the candidate-visible corpus. It
represents both complete `READ` transactions, executes each against a sealed
two-candidate state, and supervises only the observed terminal answer:

\[
  L_{outcome} = -\log \sum_{j: execute(T_j)=y^*} p(T_j\mid source,state).
\]

Each semantic source appears under three state interventions: two distinct
value assignments that swap the target outcome, and one equal-outcome state
where both transactions produce the same answer. The two clause orders for
each meaning are also present. A semantic transaction must therefore stay
coherent across six records even when the outcome alone is underdetermined.

This is not claimed as novel latent-variable learning. Its purpose is to test
whether downstream consequences plus intervention consistency can train a
small model-owned semantic owner after direct role fitting failed. QTE1 is the
fixed 0.8B capability ceiling, not a runtime teacher in this gate.

## Frozen data contract

The source is the immutable 100,000-row RRG1 QUERY corpus at SHA-256
`2d325c860e707307886f782350e7ec35ae8c23ae275260b0a937bbb738078c1c`.
The deterministic builder emits 300,000 public records and 300,000 separate
outcome-supervisor records: 200,000 distinct-outcome and 100,000 equal-outcome
cases across 50,000 complete semantic pairs.

Public records contain source text, source-owned symbols, anonymous candidate
values, state-orbit identity, and commitments. They contain no target,
distractor, symbol-role, role-order, or gold-transaction field. Supervisors
contain only the committed public identity and terminal answer. Exhaustive
generation verifies every six-record orbit, value swap, equal-outcome case,
clause-order answer invariant, identity, and forbidden-field audit.

The first local/Stokes cross-host build reproduced the public and supervisor
files exactly, but exposed that report v1 serialized absolute host paths. That
receipt format is rejected before neural use. Report v2 stores only canonical
roles and relative artifact names; public and supervisor bytes are unchanged.
The canonical hashes are:

- public: `bc438f793a3ced67a3b5493d70c14cbc39db4c20f3fe0fb50579af6b5f1daea9`;
- supervisor: `affa2cc36412f07f2816a00bbe2abfb06ee93be3b602c79b79b248f4ccf2552d`;
- report: `e9267aadaa1413778d7ac54db6a72a95500da92acee733fccaa655830b9cb1a6`.

Stokes must reproduce all three canonical hashes before neural admission.

## Neural admission boundary

QTE1 is closed and independent Stokes job `767029` reproduces all three data
hashes exactly. The frozen neural interpreter uses each parent model's final
eight blocks with LoRA rank 16 and alpha 32. For each complete candidate it
scores `YES` versus `NO` likelihood for the fixed claim that the candidate is
the requested source and the other is the distractor. The transaction
distribution is trained only through the terminal-outcome marginal. No direct
role or transaction label enters the candidate-visible or supervisor files.

The complete 300,000-row objective is evaluated through exact sufficient
statistics: each of 50,000 semantic pairs has two clause orders, two
informative distinct-outcome copies, and one zero-gradient equal-outcome copy.
The compressed mean multiplies source cross-entropy by `2/3`, exactly matching
the six-row mean. A `0.25` symmetric clause-order consistency penalty aligns
the two distributions by physical mention identity without revealing which
identity is TARGET. Training is one deterministic epoch over 50,000 pairs,
pair batch 32, AdamW `1e-4` cosine decay, seed `2026080702`.

Three independent single-H100 arms run concurrently: protected Shohin,
SmolLM2-135M, and a matched SmolLM2 control whose distinct terminal outcomes
are flipped. All receive identical data, order, updates, objective geometry,
and evaluation. Development uses the source-disjoint CCR1 board at SHA-256
`299237068f436ba33a68487b5300fcd724f8c98bd8bfe6b1916a4ebc7541ebf7`.

A treatment pass requires at least `765/768`, every mode `254/256`, every
renderer `127/128`, mapped mention-swap equivariance `765/768`, a context-
scrub drop of at least 250, bit-exact entity-renaming behavior, and an unchanged
frozen parent. The flipped control must remain at most `430/768`. If both
treatments pass, selection is deterministic by exact count, signed margin,
then SmolLM2. A development pass admits exactly one fresh balanced
confirmation board. A miss closes the exact outcome mechanism without seed,
width, duration, renderer, prompt, or threshold variants.

The confirmation generator is frozen before development results. It uses seed
`2026080703`, 256 new exact TFS1 programs, a disjoint 32-entity bank, six new
query families, and two clause orders per family. Every renderer contains
exactly 64 transaction-0 and 64 transaction-1 queries. Generation runs on
Stokes with exhaustive source/query/entity overlap audits against CGL1
training, the development board, and the already-open PQI/QTE board. The board
remains unopened unless assessor `744568` passes.

The confirmation evaluator is frozen at commit `c08be09`. Its minimal
read-only Newton overlay is
`runtime_overlays/diverge_cgl1_confirm_runtime_c08be09_r3`, with
`SHA256SUMS` SHA-256
`367a85bf08767842983ebf636f38f5bda580c54201f9093ef2dd361a11cf50d7`.
The overlay contains only the confirmation data contract, evaluator, job
wrapper, source receipt, and checksum manifest; all other imports resolve from
the already-qualified CGL1 base runtime. A fail-closed CPU dispatcher may run
only after `744568` exits successfully. It verifies the fixed board, verifies
that the assessment selected exactly one admitted treatment, submits exactly
one H100 confirmation, and records the child job and artifact hashes. It does
not alter an arm, threshold, board, or score.
