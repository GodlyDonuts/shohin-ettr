# DIVERGE-JRB1: Joint Register Binder

**Status:** development FAIL; confirmation closed
**Parent:** confirmed DIVERGE-NCP1 command owner and confirmed DIVERGE-EAL2 temporal owner  
**Objective:** remove exact register-name scanning plus typed initial-state and query-register inputs in one integrated gate

## Capability hypothesis

NCP1 already maps raw commands to ordered operation pointers. EAL2 already maps
natural evidence to BEFORE/AFTER semantics. Their remaining end-to-end path still
uses an exact string scanner to assign each numeric mention to one of two
episode-local registers, receives a typed initial vector, and scores both output
coordinates rather than interpreting a natural query.

JRB1 gives one shared learned owner all three register-binding duties:

1. assign four evidence numbers to the two episode-local registers;
2. assign two natural initial-state numbers to those registers; and
3. select the register requested by a natural late query.

The owner is a permutation-equivariant dynamic pointer. A shared byte encoder
represents the source, another shared encoder represents the two episode-local
register names, and normalized dot products score source mentions or a pooled
query against the table entries. Evidence and initial-state paths share the same
mention projection; the query has one task-specific projection. The model has no
fixed output head for register names.

The qualified EAL2 reader remains bit-identical but no longer receives register
IDs from its exact scanner. Its BEFORE/AFTER logits are constrained using JRB1's
learned register groups. The qualified NCP1 pointer also remains bit-identical.
Exact numeric span extraction, operation-name lookup during law induction, the
bounded law solver, modular executor, and the two-entry register table remain
engineered and are outside this claim.

## Frozen data

- Training seed: `2026080811`
- Development seed: `2026080812`
- Conditional confirmation seeds: `2026080813` through `2026080817`
- Training rows: `100,000`
- Development and each confirmation board: `256` episodes
- Every episode contains three demonstrations for each of eight unseen linear
  operations, sixteen noncommuting programs at held depths 12 through 32, and
  two late natural queries per program.
- Register and operation names are opaque and episode-local.
- Training and evaluation use disjoint compositions of independently recurring
  initial-state and query renderer primitives.
- Training, development, confirmations, and prior EAL2/NCP1 boards must have
  zero source-text hash, opaque-name, and episode-identity overlap.
- Candidate-facing boards contain raw evidence, raw command, raw natural
  initial-state text, raw natural query text, and two dynamic tables. They do
  not contain typed initial states, typed operation sequences, register indices,
  numeric role labels, terminal states, or answers.

## Frozen optimization

Two matched arms start from identical weights and receive the identical row
order and schedule:

- **Treatment:** the dynamic register table is aligned with the targets.
- **Shuffled-table model:** the table is cyclically permuted during training
  while targets remain in canonical coordinates, deleting the only usable
  episode-local name-to-coordinate relationship.

Each arm receives exactly 1,000 AdamW updates, batch 128, learning rate 0.001,
and weight decay `1e-4`. The loss is the unweighted sum of evidence-mention,
initial-mention, and query cross-entropies.

## Frozen evaluation arms

1. **Treatment:** normal register names and normal dynamic table.
2. **Renamed registers:** unseen names consistently replace registers in
   evidence, initial state, query, and table.
3. **Table permutation:** the same semantic table is presented in reverse
   order and predictions are mapped back to canonical identity. This is a
   positive equivariance test, not a negative control.
4. **Register-source scrub:** evidence, initial state, and query use unrelated
   decoy names while the candidate table remains original.
5. **Shuffled-table model:** the independently trained matched control is
   evaluated on the normal board.

The same frozen NCP1 program predictions are used for every JRB1 arm. Programs,
initial states, law packets, and late query choices become typed only after the
learned owners have run; raw source text is not consulted by the executor.

## Conjunctive development gate

JRB1 passes only if every condition is true:

- qualified EAL2 and NCP1 parent hashes and PASS reports match;
- NCP1 program exactness is at least 99%, with at least 95% at every depth;
- treatment evidence-register, initial-register, query-register, complete-role,
  initial-state, terminal-state, and answer exactness are each at least 99%;
- treatment law commitment is at least 99% and terminal-state exactness is at
  least 95% at every depth;
- renamed-register evidence, initial, query, terminal-state, and answer
  exactness are each at least 99%;
- table-permutation semantic equivariance for those same outputs is at least
  99%;
- register-source-scrub terminal-state and answer exactness are each at most
  5%;
- shuffled-table-model terminal-state and answer exactness are each at most 5%;
- treatment/control initialization, data, update count, batch, and learning
  rate match exactly;
- checkpoints match their reports, parent weights are bit-identical, the JRB1
  runtime contains no exact register search, and typed initial/query carriers
  are absent.

The five confirmation boards are opened only after a development PASS. Every
seed must independently pass the unchanged evaluator. A development failure
closes this exact JRB1 rule without width, duration, seed, threshold, renderer,
or loss variants. Attribution may distinguish binding, compilation, execution,
and query failure, but may not rescue the gate.

## Claim boundary

A confirmed PASS would qualify controlled model-owned register binding across
natural evidence, natural initial state, and natural late query, composed with
previously qualified command and temporal owners. It would not establish
open-domain language understanding, unrestricted arithmetic, general reasoning,
or superiority on public benchmarks. A FAIL would show that this shared dynamic
pointer does not remove the three register scaffolds under the frozen budget.

## Development result

Final corpus report SHA-256 is
`d240756fa82052cc204a93596064524dec74ec0a3da8f84199f5106863ecc2dd`;
training SHA-256 is
`2181593d3813636a398cd75090577d3e3aaf60f15269cfac509cef9de7000c38`.
Treatment/control jobs `744822/744823` complete the unchanged schedule from
identical initialization. Treatment reaches `2048/2048` joint fixed-sample
exactness. Its 290,177-parameter checkpoint SHA-256 is
`bc3fc348acac017e0782e09465036640b32819da5245811f418c2b8f3c56552e`.

Frozen development replacement `744828` passes the substantive evidence,
initial-state, and recurrent-execution path:

- parent NCP1 programs: `4096/4096`;
- evidence register binding: `6144/6144` normal and renamed;
- complete transition roles and law commits: `6144/6144` and `256/256`;
- natural initial-state binding: `4096/4096` normal and renamed;
- terminal recurrent state: `4096/4096` normal and renamed, with 100% at every
  depth 12--32;
- natural query binding: `7912/8192 = 96.5820%` normal and
  `7844/8192 = 95.7520%` renamed;
- exact natural-query answers: `7914/8192 = 96.6064%` normal and
  `7846/8192 = 95.7764%` renamed.

The conjunctive result is FAIL. Report SHA-256 is
`df8827dbea25428f843e06751bd722c0f52d362f7e8619550d23ffd9c3473171`;
confirmation remains unopened.

One read-only attribution closes the lane. Query errors are spread across all
eight held renderer compositions rather than one missing primitive; the worst
normal pair is `952/1024 = 92.97%`. Mean-pooling the complete query is the
remaining positive-path bottleneck. Register-source scrub gives
`1395/6144 = 22.71%` evidence, `1029/4096 = 25.12%` initial, and
`4044/8192 = 49.37%` query exactness, which is near the relevant grouping and
binary chance levels. Its downstream state/answer rates (`15.70%/28.25%`)
show that the frozen 5% ceilings were not chance-calibrated.

The shuffled-table model commits all 256 laws but produces only `30/4096`
canonical terminal states and `7811/8192` correct answers. This is the expected
signature of one coherent global coordinate swap: the state tuple is permuted,
while a correspondingly permuted query still retrieves the correct semantic
answer. A fixed table rotation is therefore an equivalent representation, not
an information-deleting control. The preregistered verdict remains FAIL; the
result is not relabeled. The successor must represent state in table-relative
content-addressed slots, focus query evidence token-wise, and break owner
coherence only in causal controls.
