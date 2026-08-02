# Shohin Research Archive

This directory separates exploratory research from the repository's operational
entry points and executable custody records.

## Archive Layout

### Frontier proposals

External proposals and their Shohin-specific reviews live in
[`frontier/`](frontier/):

- [`FRONTIER_AGENT_PLANS.md`](frontier/FRONTIER_AGENT_PLANS.md)
- [`FRONTIER_AGENT_PLANS_ANALYSIS.md`](frontier/FRONTIER_AGENT_PLANS_ANALYSIS.md)
- [`FRONTIER_NOMINAL_GRAPH_REWRITE_MACHINE_PLAN.md`](frontier/FRONTIER_NOMINAL_GRAPH_REWRITE_MACHINE_PLAN.md)
- [`FRONTIER_S9_ARCHITECTURE_PROPOSAL.md`](frontier/FRONTIER_S9_ARCHITECTURE_PROPOSAL.md)
- [`FRONTIER_S9_TO_GENERAL_REASONING_ANALYSIS.md`](frontier/FRONTIER_S9_TO_GENERAL_REASONING_ANALYSIS.md)
- [`GEMINI_PLAN.md`](frontier/GEMINI_PLAN.md)
- [`GEMINI_PLAN_ADAPTED.md`](frontier/GEMINI_PLAN_ADAPTED.md)

### Concept research

Mechanism hypotheses and broad research programs live in
[`concepts/`](concepts/):

- [`LATENT_MEMORY_RESEARCH.md`](concepts/LATENT_MEMORY_RESEARCH.md)
- [`REASONING_ATTACK_PLAN.md`](concepts/REASONING_ATTACK_PLAN.md)
- [`VRWM_RESEARCH.md`](concepts/VRWM_RESEARCH.md)

### Baseline interactions

Qualitative records from the frozen 300k model live in
[`baselines/`](baselines/):

- [`RAW300K_INTERACTION_RESULT.md`](baselines/RAW300K_INTERACTION_RESULT.md)
- [`RAW300K_FREEFORM_INTERACTION_RESULT.md`](baselines/RAW300K_FREEFORM_INTERACTION_RESULT.md)

### Active architecture experiments

New preregistrations that are not bound to legacy root-relative manifests live
here directly:

- [`R12_STICKY_OPCODE_MACRO_RAIL_PREREG.md`](R12_STICKY_OPCODE_MACRO_RAIL_PREREG.md)
- [`R12_REGISTRY_CONSTRAINED_OPCODE_PROJECTION_PREREG.md`](R12_REGISTRY_CONSTRAINED_OPCODE_PROJECTION_PREREG.md)

## Documents Intentionally Kept At Root

The following classes remain at repository root:

- `AGENT_RUNBOOK.md`, `README.md`, and the current plans and operating records.
- `R11_*.md` and `R12_*.md` preregistrations, protocols, theories, results, and
  no-go records.

Many R11/R12 files are loaded by exact root-relative paths, hashed into frozen
manifests, or named in immutable experiment receipts. Moving only some of them
would silently break reproducibility. They should be migrated only after a
single path-registry change updates code, tests, frozen manifests, and archival
receipts together.
