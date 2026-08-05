# Prompt-Selected Presented Algebra

Status: next architecture hypothesis after the closed PCDL gate; implementation
not yet score-bearing; no novelty or reasoning claim.

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

The first board uses anonymous, freshly permuted carriers and three action
families:

1. cyclic/affine actions;
2. noncommuting dihedral or permutation-generator actions; and
3. finite transformation-monoid actions with repeated binding.

Training uses words of length 1--4. Frozen development uses lengths 8 and 12,
new carrier renamings, unseen generator words, and held-out relation sets. A
separate confirmation board is generated but not opened by a failed pilot.

Matched arms instantiate and execute the same encoder and action tensors:

- PSPA: relation-constrained whole-generator completion and exact tied word
  composition;
- RECURRENT: an unconstrained tied dense state updater;
- TRANSFORMER: a parameter-matched ordinary sequence model;
- BROKEN-RELATION: PSPA with source relations exchanged across episodes.

The first seed is a fail-fast pilot. PSPA advances to five seeds only if it
beats the best non-intervention control by at least five macro points and gains
on every family. The five-seed gate then requires:

- at least +10 absolute OOD macro points over the best matched control;
- positive gain on every family in at least four of five seeds;
- at least five points lost under exchanged relations;
- at least five points lost under generator-lineage swaps;
- at least 95% primitive generator fit but no answer supervision leakage; and
- one unchanged-weight confirmation pass after all development gates pass.

Failure of the first seed closes this exact compiler/runtime without width,
duration, seed, loss, carrier, or relation-set tuning.

## Resource envelope

Board generation and exhaustive oracle/identifiability checks are CPU-only:
one Stokes job, 32 cores, four-hour wall limit, at most 128 CPU-hours.

The pilot requests four independent single-H100 jobs. Each has a one-hour
Slurm ceiling, 1,000 updates, 256 examples/update, and an expected runtime
under two minutes from the preceding 51k--113k parameter gates. Expected pilot
use is at most 0.15 H100-hours; hard allocation ceiling is four H100-hours.

Only a passing pilot releases the five-seed matrix: 20 single-H100 jobs with
the same one-hour ceiling, expected under 0.7 H100-hours and hard-capped at 20
H100-hours. Confirmation is one additional unchanged-weight H100 job. No long
pretraining run is authorized by mechanics or training loss alone.

## Immediate implementation order

1. implement exact anonymous-carrier composition and relation checking;
2. prove permutation equivariance and no fieldwise action mixing;
3. generate and audit the three-family board and answer identifiability;
4. implement the shared compiler plus all four matched arms;
5. pass focused gradients, equal-parameter, equal-execution, and CPU smoke;
6. run one seed on four single H100s; and
7. apply the fixed pass/kill rule before any extension.
