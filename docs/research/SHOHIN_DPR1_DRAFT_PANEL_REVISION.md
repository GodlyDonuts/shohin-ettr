# DPR1: Model-Owned Draft Panel Revision

Status: frozen information-ceiling gate, 2026-08-09. No capability fit,
holdout read, or larger-MoE work is authorized yet.

## Rationale

MPR2's trained owner carries weak useful information but a single aligned
draft cannot improve over source-only computation. Its owner+aligned oracle is
only 265/1,289, below the required 286 even under perfect selection. DPR1
changes the temporal object from one premature commitment to a coherent panel
of eight complete model-owned trajectories. The panel is not averaged
fieldwise; every trajectory remains independently readable.

This is a standard test-time-compute family, not a novelty claim. It is the
fastest falsifier for whether the pinned OLMoE owner contains enough diverse
correct computation to justify training a panel-conditioned reviser.

## Frozen ceiling probe

- exact pinned OLMoE and MPR1 trained source-only owner checkpoint used by
  MPR2;
- exact 1,289-row source-disjoint development board only;
- `K=8` complete samples per source in one fixed run;
- sampling: temperature 1.0, top-p 0.95, top-k 20, no thinking, maximum 768
  new tokens, seed `2026080924`;
- source plus original revision envelope, with the old draft causally hidden
  through the full model exactly as in the owner;
- no verifier, assessor, candidate correctness, or answer label is visible to
  generation.

The offline assessor reports fixed-index accuracy, whole-panel oracle,
domain oracle, normalized completion diversity, generated tokens, exhaustion,
latency, memory, checkpoint hash, and source/runtime hashes.

## Frozen gate

A panel-conditioned training campaign is allowed only if all conditions pass:

1. whole-panel oracle `>=350/1,289`;
2. MATH oracle `>=90/623`;
3. logic/science oracle `>=245/637`;
4. executable-code oracle `>=15/29`;
5. at least 25% of rows contain two or more distinct normalized completions;
6. all eight fixed candidate indices are complete, with no index selected or
   dropped post hoc, and total exhaustion `<=80/10,312`;
7. exact source/checkpoint/runtime hashes and complete reports.

If the gate passes, freeze one panel-conditioned reviser with matched shuffled
panel and hidden-panel controls before scores. If it fails, close DPR1 without
K, temperature, top-p, prompt, seed, or decoding variants. Holdout remains
sealed in either case until a later trained development gate passes.

