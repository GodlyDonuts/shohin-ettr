# R12 Episodic-Generator Adversarial Audit

## Disposition

`reject_architecture_native_shohin_reasoning_retain_neurosymbolic_solver`

Jobs `704786` and `704792` are valid measurements of a bounded
neuro-symbolic permutation solver. They are not evidence that Shohin learned
program induction or reasoning.

## Decisive Findings

1. **Shohin is absent from the runtime.** The runner instantiates only
   `EpisodicGeneratorConstraintCompiler`. The protected 125,081,664-parameter
   checkpoint is never loaded or called. Its parameter count is added only to
   a conceptual ledger. The stored model has 24 tensors and 232,065
   parameters.
2. **The learned component is a direction reader.** The compiler deletes its
   inherited learned transition modules. Exact code identifies supports by
   record count, constructs all 127 binary generator words, scores them,
   seals target tables, and executes queries in Python.
3. **Parsing and search are external.** Regex scanners extract cardinality,
   opaque-action equality, numeric states, record boundaries, and query
   programs. The report counts 386 exact preparation-parser calls. Enumerating
   127 words is exhaustive search even though the report's
   `candidate_time_search_calls` field is hard-coded to zero.
4. **Abstract target programs are not held out.** Training contains all four
   length-two binary words. Fourteen of 22 development target instances reuse
   those words; only the eight depth-three-through-six targets are abstractly
   unseen. Contextual law and concrete-map hashes are disjoint, but those
   hashes include episode-specific supports and do not prove program holdout.
5. **Source deletion is not process-level.** The source and tensor batch
   remain live during evaluation. The packet includes a source-derived digest,
   and deletion checks only exclude literal source/support bytes from packet
   serialization.
6. **The claimed logical intersection is approximate.** The candidate uses a
   finite-temperature softmax and a `0.999` confidence threshold. Every
   program retains nonzero mass and duplicate syntactic words weight maps by
   multiplicity. This is not exact set intersection.
7. **Several controls are weaker than described.** Support recoding only
   flips two already-parsed support tensors, which is tautologically invariant
   when all binary words are enumerated. Observation shifting changes support
   and target endpoints together. Equal-count support selection uses an
   implementation-dependent `topk` tie order.
8. **Evidence is too small and not independently implemented.** Development
   has 11 episodes, one unseen-law row, and two held-out-family rows. The board
   audit imports the generator, compiler, decoder, and executor it assesses.

## What The Result Still Establishes

- A learned byte-level record-orientation classifier transfers across the six
  frozen renderers.
- Given exact parsed records, complete support tables, a fixed permutation
  ontology, and exhaustive bounded word enumeration, the symbolic solver
  answers all 11 development queries.
- Sparse target records are inclusion-minimal inside that closure.
- Ambiguous evidence can be rejected after consensus hardening.
- A coherent alternate world should be executed rather than rejected.

These are useful mechanics and data-design results. They are not a Shohin
reasoning baseline.

## Required Successor Boundary

Any future positive claim must:

1. load the actual Shohin trunk and show causal contribution against zeroed,
   randomized, and swapped-trunk controls;
2. consume raw tokens without exact regex extraction of semantic fields;
3. place compilation and execution in separate processes and destroy source,
   residual, KV, parser, and compiler state before late queries;
4. remove host enumeration, semantic sealing, and domain execution from the
   claim-bearing path;
5. use genuinely disjoint program structures and hundreds of episodes across
   multiple frozen seeds;
6. cover at least three genuinely different ontologies under leave-one-
   ontology-out evaluation; and
7. distinguish contradiction, behavioral ambiguity, and coherent alternate
   worlds with independently audited controls.

The proposed successor is the Endogenous Typed Theory Reactor: one anonymous
typed-transaction architecture evaluated on Horn closure, typed term
rewriting, and guarded resource processes. That gate is specified separately;
this result does not pre-authorize a capability claim.
