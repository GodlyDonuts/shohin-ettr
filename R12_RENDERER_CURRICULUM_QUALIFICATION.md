# R12 Renderer-Curriculum Qualification

## Decision

The source-deleted anonymous-machine compiler now passes the frozen
five-seed, three-family renderer/law/composition smoke:

- treatment: **120/120 = 100%**
- direction-shuffled control: **65/120 = 54.1667%**
- margin: **+45.8333 percentage points**
- positive seed-fold directions: **15/15**
- candidate-time oracle/search/verifier calls: **0/0/0**

Decision:
`renderer_curriculum_passes_five_seed_three_family_holdout_smoke`.

This is the first retained evidence of one learned mechanism transferring
across unseen laws, longer compositions, an unseen surface renderer, and a
fully held-out law family. It establishes bounded systematic anonymous-machine
compilation. It does **not** establish unrestricted natural-language or
general reasoning.

## Mechanism

The mechanism combines two pieces:

1. **Renderer-neutral structural typing.** The source equality/incidence graph
   identifies anonymous action and state key classes. The sealed machine then
   identifies the late-query start by membership in its state set. No renderer
   grammar or exact transition parser is used at candidate time.
2. **Counterfactual direction curriculum.** Every fitting law is rendered
   once more in target-first order under symbols absent from the held-out
   renderer. A 152,933-parameter shared recurrent byte compiler learns the
   remaining source-versus-target direction.

The matched control receives identical candidate bytes, initialization,
parameters, optimizer, updates, and query labels, but the counterfactual
source/target labels are swapped.

## Parameter and Custody Ledger

- protected Shohin parameters: 125,081,664
- learned compiler parameters: 152,933
- conceptual complete system: 125,234,597
- global limit: 200,000,000
- fitting rows per family-holdout arm: 24 original + 24 counterfactual
- optimizer updates per arm: 300
- preparation exact-parser calls: 360 aggregate
- candidate-time exact-parser calls: 0
- candidate-time oracle/search/verifier calls: 0/0/0
- inference source bytes are deleted after sealing
- no H100s were used for the five-seed replication

## Results

Each fold trains on two families and evaluates only the third family. Every
evaluation includes two unseen-law, two longer-composition, two held-out
renderer, and two joint cases.

| Held-out family | Treatment | Control |
|---|---:|---:|
| affine modular | 40/40 | 25/40 |
| bitwise rotate/xor | 40/40 | 20/40 |
| permutation | 40/40 | 20/40 |
| **aggregate** | **120/120** | **65/120** |

Every treatment seed scores 8/8 in every held-out family. Every cell in every
seed scores 2/2. All 15 treatment-control comparisons are positive.

The preceding all-family smoke scores 24/24 treatment versus 13/24 control.
Structural typing alone scores only 13/24 and 1/6 on the renderer cell, so the
counterfactual direction curriculum is necessary.

## Causal Interpretation

The old `0/6` renderer failure was not a generic inability to execute unseen
laws. It was a factorized representation failure:

1. local three-way role prediction did not infer key type under a new syntax;
2. the held-out renderer introduced a target-first permutation absent from
   training; and
3. query decoding ignored the sealed machine's exact state/action partition.

Structural typing eliminates invalid packets but does not solve direction.
Correct target-first supervision solves direction, while swapped supervision
does not. The 55-case aggregate margin therefore depends causally on the
counterfactual direction labels rather than extra bytes or compute.

## Scope Boundary

This result must not be described as genuine general reasoning:

- every episode exposes a complete finite transition table;
- all three families compile to the same anonymous-machine ontology;
- action count and state cardinalities are fixed and make key incidence
  informative;
- target-first role order is covered during training under different symbols;
- the learned compiler is currently a sidecar and does not consume Shohin
  trunk features; and
- no natural-language theorem proving, latent law discovery, or open-ended
  planning is tested.

The next falsifier must vary action count, state cardinality, incidence
profiles, record completeness, and machine topology so fixed frequency typing
cannot solve the task. It must preserve source deletion, candidate-time zero
oracle/search/verifier access, matched direction controls, and held-out
surface grammars. Only after that survives should the mechanism be integrated
into Shohin and tested on natural-language post-training.

## Artifacts

- independent audit:
  `artifacts/r12/source_deleted_multifamily_machine_board_v1/renderer_curriculum_five_seed_audit.json`
- audit SHA-256:
  `6cbf52ffe48fe79b8bf996d6b70fa244840153fd0b0d293ce47766e987b5179e`
- all-family smoke SHA-256:
  `1b41f27fbc6f402a0a2b9b8e866e181318ee09c14e9c8e0886602eff33fa325d`
- structural-type smoke SHA-256:
  `1b18e02f908fd22ef582e013fbf4e20e497c918655128937b720971d377b1cb5`

The independent audit contains the SHA-256 manifest for all 15 source reports.
