# Prompt-Selected Presented Algebra

Status: repaired mechanics pilot passes; learned compiler not yet implemented;
no novelty or general reasoning claim.

## Thesis

Shohin's recent mechanisms can fit source consequences, preserve invariants,
maintain particles, lower contradiction, and solve episode coefficients. None
forces the latent representation to obey an algebra that determines behavior
outside the observed witnesses. The one positive precedent is S7: once a
learned Cayley generator was the only route to the answer, unseen-law recurrent
execution reached 100% while an exact-fit ordinary Transformer failed.

Prompt-Selected Presented Algebra (PSPA) generalizes that principle. A source
does not compile an opaque vector or an arbitrary feature basis. It selects and
fills a finite presentation:

```text
P(X) = (carrier slots C, generator actions {A_g}, relations R, query word q)
```

Each `A_g` is a complete action over anonymous carrier slots. A tied runtime
composes generator words. Prompt relations and source witnesses constrain the
same actions that execute the late query:

```text
relation_energy = sum_(u=v in R) distance(A_u, A_v)
witness_energy  = sum_(x,w,y) CE(A_w one_hot(x), y)
answer          = A_q one_hot(query_start)
```

Whole action tables persist across recurrent completion. Coordinates from
different presentations are never averaged. The query is absent from action
completion and can only execute the completed presentation.

## Architectural separator

PSPA is not a standard Transformer, MoE, LoRA, recurrent-depth block, latent
scratchpad, particle average, kernel regression, or external symbolic tool.
Its defining restriction is that prompt-conditioned reasoning state is a
finite algebraic presentation and every update must preserve the semantics of
generator composition.

Finite groups, semigroups, equivariant networks, differentiable interpreters,
neural theorem proving, and learned Cayley tables are prior art. S7 is direct
internal prior art. Any novelty claim is therefore provisional. The candidate
separator is language-conditioned selection and completion of a presented
algebra with one tied model-owned executor, rather than hard-coding one cyclic
topology or emitting operation embeddings.

## Minimal discriminating gate

The first mechanics board uses anonymous, freshly permuted carriers and three action
families:

1. cyclic/affine actions;
2. noncommuting dihedral or permutation-generator actions; and
3. random multi-generator permutation actions with repeated binding.

Training uses words of length 1--4. Frozen development uses lengths 8 and 12,
new carrier renamings and unseen generator words.

The mechanics harness evaluates:

- PSPA: source-challenge-constrained whole-generator completion and exact tied word
  composition;
- RECURRENT: an unconstrained tied dense state updater;
- TRANSFORMER: an ordinary two-layer sequence model;
- SHUFFLED-CHALLENGE: PSPA with source challenge outcomes exchanged across
  episodes; and
- LINEAGE-SWAP: PSPA with selected whole presentations exchanged across
  episodes before query execution.

The combined harness has 180,054 trainable neural-control parameters. The
structured PSPA runtime itself is deterministic and receives strictly less
trainable capacity than both controls. This first result is therefore a
favorable-control mechanics test, not the final parameter/FLOP-matched learning
claim. A pass authorizes a learned language-to-presentation compiler and its
matched gate; it does not by itself establish a superior learned architecture.

The first seed is a fail-fast mechanics pilot. PSPA advances to a learned
compiler only if it
beats the best non-intervention control by at least five macro points and gains
on every family, reconstructs at least 95% of complete generator tables, and
loses at least five points under both causal interventions. The later learned
compiler's five-seed gate requires:

- at least +10 absolute OOD macro points over the best matched control;
- positive gain on every family in at least four of five seeds;
- at least five points lost under exchanged relations;
- at least five points lost under generator-lineage swaps;
- at least 95% primitive generator fit but no answer supervision leakage; and
- one unchanged-weight confirmation pass after all development gates pass.

Failure of the first seed closes this exact structured compiler/runtime without
width, duration, seed, loss, carrier, or challenge-set tuning.

## Repaired mechanics result

The first immutable runtime exposed three ties among 6,144 development
episodes because six source challenges did not always identify a unique
complete multi-generator presentation. This was an oracle-generation defect,
not a model error. The repaired generator constructs up to eight
whole-candidate challenges and proves that every wrong complete presentation
is eliminated before evaluation. Exhaustive local replay then recovered all
6,144 presentations and answers exactly.

The repaired seed-43 H100 pilot used 1,000 updates, 128 examples per update,
180,054 combined trainable control parameters, and 128,000 charged examples.
It completed in 80.512 seconds at 1,589.829 examples/s.

| Arm or diagnostic | Six-cohort OOD macro exact |
|---|---:|
| PSPA | **100.000%** |
| Tied recurrent control | 9.961% |
| Two-layer Transformer control | 10.840% |
| Shuffled source-challenge outcomes | 52.507% |
| Whole selected-presentation lineage swap | 13.574% |
| Complete selected-presentation recovery | **100.000%** |
| Source-challenge prediction | **100.000%** |

PSPA is 100% exact on cyclic, dihedral, and random-permutation families at
both length 8 and length 12. It clears the fixed mechanics requirements by
89.160 points over the best neural control, with large causal losses under
both interventions. The result file is
`artifacts/reasoning/pspa_presented_pilot_071229a/seed43.json`, SHA-256
`b88aabc9d09ec4dce2790efd7a12814722c132e8d104d580f96468b966842539`.

This is a mechanics result under favorable controls. The structured compiler
still receives anonymous tables and exact challenge tuples rather than
language. The next gate must learn the language-to-presentation map, retain
the tied model-owned executor, remove deterministic candidate enumeration at
inference, and compare against parameter- and total-FLOP-matched neural
controls. A failure there closes PSPA despite this perfect mechanics score.

## Resource envelope

Board generation and oracle/identifiability checks are CPU-only. Eight focused
tests and an end-to-end CPU smoke already pass. At lengths 8/12, all three
families have 100% exact generator reconstruction and answers; shuffled source
challenges score roughly 44--67% and lineage swaps roughly 5--27% in the smoke.

The first full control pilot is one single-H100 job because one checkpoint
trains both favorable controls on identical examples and reports PSPA plus both
interventions. It has a one-hour Slurm ceiling, 1,000 updates, 128
examples/update, and an expected runtime under 15 minutes. Hard allocation
ceiling is one H100-hour.

Only a passing mechanics pilot authorizes implementation of the learned
compiler and the originally specified matched four-arm, five-seed matrix. That
later matrix remains hard-capped at 20 H100-hours. No long pretraining run is
authorized by mechanics or training loss alone.

## Immediate implementation order

1. ~~implement exact anonymous-carrier whole-generator composition~~;
2. ~~prove query-late ownership and no fieldwise action mixing~~;
3. ~~generate and audit the three-family depth-shift board~~;
4. ~~implement recurrent and Transformer favorable controls~~;
5. ~~pass focused gradients and CPU smoke~~;
6. ~~run one combined seed-43 H100 pilot~~; and
7. implement the learned language-to-presentation compiler and
   freeze its parameter/FLOP-matched gate before scoring it.
