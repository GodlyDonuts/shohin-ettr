# CSDC SmolLM2 Lexical Integration Gate

Status: frozen before implementation and execution on 2026-08-05.

## Question

The matched lexical-backbone gate showed that the frozen SmolLM2-135M
backbone plus an ordinary bidirectional adapter compiles unseen lexical
polarity substantially better than Shohin (`95.947%` versus `77.344%` exact
programs).  This gate asks the next narrower question: can a model-decoded
natural-language challenge actually control CSDC's frozen candidate,
falsification, commit, and late-execution path?

It is not an unrestricted natural-language reasoning or architecture claim.
The observation compiler and CSDC completion reasoner remain the frozen typed
components.  Only challenge records cross the new lexical interface.

## Frozen inputs

- SmolLM2 ETTR parent:
  `train/ettr_smollm2_control_parent_a2026072801_7881d8e/joint-model-final.pt`
- Exact tokenizer:
  `artifacts/external/smollm2_135m_instruct_83212e1e/tokenizer.json`
- Warm-start lexical adapter:
  `train/csdc_lexical_backbone_gate_v1/smollm2/compiler.pt`
  (`abd22528da0d8dc4718c7a89d9c94520540a2f38b7f0b1d9a9e623d0af23cf4d`)
- Frozen learned-PSPA reasoner:
  `artifacts/reasoning/learned_pspa_pilot_8fb5d61/seed47.pt`
  (`c374e3b566808cb317ffcd2725653c9073d2e7aebeb75e93ed7ea2a7e2e27044`)
- Frozen role-copy control:
  `artifacts/reasoning/csdc_role_gated_copy_6359453/seed59.pt`
  (`55b5ef79110625f383f6800ac89a20dba9d0a1420bd554fd928ee70f42fdf956`)

All hashes are checked before training.  The reasoner and language backbone
are inference-only.  The lexical adapter is warm-started and may update with
new record-kind and token-role heads.

## Interface

Each episode is rendered twice from the same latent presented-algebra batch:

1. Frozen typed records feed the learned-PSPA row compiler and produce whole
   completion candidates.
2. Natural-language records feed frozen SmolLM2 plus the lexical adapter.
   The model identifies challenge records and copies source positions for
   START, OUTCOME, and the ordered generator WORD.

State and generator names are episode-local aliases.  The evaluator interns a
copied alias by exact source identity; it does not predict a global class or
read a hidden semantic label.  Alias assignments are independently permuted
per episode.  Held evaluation uses disjoint alias pools and an unseen syntax
template.

The decoded challenge tuples alone select one whole CSDC completion.  The
answer is then read by frozen late execution.  Fields are never averaged
across candidate lineages.

## Fixed training budget

- Seed: `2026080507`
- 1,500 updates, batch 128 (`192,000` generated episodes)
- AdamW, learning rate `3e-4`, weight decay `0.01`
- Gradient clipping at `1.0`
- Query lengths cycle over 1--4 in training
- Training challenge templates: 0, 1, 2
- One H100, one run, maximum wall time one hour

## Evaluation

Six cohorts (three presented-algebra families by query lengths 8 and 12) are
evaluated on each split with 1,024 examples per cohort:

- `development`: unseen episodes, training syntax and alias pools
- `lexical_shift`: unseen template 3 and disjoint state/generator alias pools

The typed challenge oracle is reported on the identical candidates.  Two
causal controls are fixed:

- shuffle decoded OUTCOME fields across episodes before selection
- swap the selected whole candidate table across episodes before execution

## Pass/kill rule

The bridge passes only if all conditions hold:

1. Development learned answer accuracy >=95%.
2. Development exact challenge tuples and selected tables are each >=95%.
3. Lexical-shift learned answer accuracy, exact challenge tuples, and selected
   tables are each >=90%.
4. Every lexical-shift family/length cohort has >=90% learned answer accuracy.
5. Shuffled outcomes reduce aggregate answer accuracy by >=20 percentage
   points on both splits.
6. Lineage swaps reduce aggregate answer accuracy by >=20 percentage points
   on both splits.
7. Typed-oracle aggregate answer accuracy is >=98% on both splits.

Any missed condition closes this exact bridge.  No width, seed, duration, or
loss-weight repair is authorized from the same result.  A pass authorizes the
next expansion: lexicalize observation records and the late query under the
same frozen CSDC core.

## Resource estimate

- CPU smoke/audit: <=10 minutes, no GPU
- H100 training and both evaluations: expected 20--35 minutes, hard limit 1h
- Durable output: one adapter checkpoint plus one JSON report, expected <50MB

