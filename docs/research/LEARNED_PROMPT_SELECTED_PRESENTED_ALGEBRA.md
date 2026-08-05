# Learned Prompt-Selected Presented Algebra

Status: implementation gate frozen after the deterministic PSPA mechanics pass.

## Capability hypothesis

The repaired PSPA mechanics result proves that one coherent finite
presentation plus tied generator composition can solve long, unseen words
when the presentation is supplied through structured observations. The next
question is whether a neural compiler can recover that executable object from
rendered source statements without seeing the late query or answer.

The treatment compiles a source-only set of randomized symbolic-language
records into complete generator action tables. A Sinkhorn projection couples
all rows and columns of each generator, and inference commits to one whole
permutation. The late query is then executed only by repeated application of
those committed generators.

This is not a natural-language reasoning claim. The first gate uses generated
symbolic language with randomized templates, record order, carrier renaming,
families, and words. It tests whether the architectural bridge from language
evidence to an executable algebra is learnable.

## Fixed arms

1. `PRESENTED`: source-language encoder, whole-generator permutation
   projection, and tied late-query composition.
2. `ROW_SOFT`: identical encoder, parameters, table logits, losses, and tied
   executor, but each generator row is normalized independently. This is the
   primary matched control; it lacks global action closure.
3. `DIRECT`: a favorable source-attention recurrent answer model trained
   directly on short query answers.
4. `SHUFFLED_CHALLENGE`: unchanged PRESENTED weights with challenge outcomes
   exchanged across episodes before rendering.
5. `LINEAGE_SWAP`: unchanged PRESENTED weights with complete compiled
   presentations exchanged across episodes before late-query execution.

The compiler loss uses only source observations and source-challenge
consequences. The late query and answer are absent from both compiler APIs and
compiler losses. DIRECT receives answer supervision and is therefore a
favorable control, not an underpowered straw arm.

## Data and shift

- train query words: lengths 1--4;
- development query words: lengths 8 and 12;
- families: cyclic, noncommuting dihedral, and random three-generator
  permutation systems;
- every episode randomizes carrier identity, record order, renderer template,
  omitted generator rows, source challenges, and query word;
- development seeds and renderer streams are disjoint from training;
- a later confirmation generator remains unopened unless five development
  seeds pass.

## Pass and kill rules

The one-seed implementation pilot advances only if:

1. PRESENTED beats the best control by at least five macro exact points;
2. PRESENTED improves every family;
3. exact complete-generator recovery is at least 95%;
4. source-challenge exactness is at least 95%; and
5. shuffled challenges and whole-lineage swaps each cost at least five points.

If the implementation pilot passes, a five-seed matched gate requires at
least +10 OOD macro points over the best control, positive gain on every
family in four of five seeds, the same causal losses, and one unchanged-weight
confirmation pass. If the one-seed pilot misses, this compiler closes without
width, duration, seed, temperature, renderer, or auxiliary-loss tuning.

## Resource envelope

Focused tests and CPU smoke precede all allocation. The one-seed pilot is one
single-H100 job containing all three learned arms, 2,000 updates, 128 examples
per update, and a one-hour hard ceiling. Expected use is below 0.25 H100-hour.
Only a passing pilot authorizes up to 15 independent single-H100 development
jobs and a total ceiling of 20 H100-hours. No language pretraining or Shohin
integration follows from training loss alone.

