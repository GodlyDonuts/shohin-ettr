# Learned Prompt-Selected Presented Algebra

Status: one-seed gate failed; joint projection compiler closed.

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

## Result

Seed-47 job `739335` trained all arms for 2,000 updates and 256,000 examples.
The matched PRESENTED and ROW_SOFT compilers each have 122,763 parameters;
DIRECT has 130,827. The run completed in 519.701 seconds at 492.591
examples/s.

| Arm or diagnostic | Six-cohort OOD macro exact |
|---|---:|
| PRESENTED | 9.147% |
| ROW_SOFT | **25.798%** |
| DIRECT | 9.749% |
| PRESENTED with shuffled challenges | 8.984% |
| PRESENTED with whole-lineage swap | 8.887% |
| PRESENTED source-challenge exact | 10.227% |
| PRESENTED complete-table exact | 0.000% |

PRESENTED loses the matched ROW_SOFT control by 16.651 points, remains near
chance, and has no causal challenge signal. Its source loss remains near the
uniform floor for roughly the first 1,400 updates and ends at 4.366, while
ROW_SOFT falls to 1.565. This is a decisive miss.

A read-only checkpoint diagnosis shows that ROW_SOFT parses every observed
generator row exactly and reaches 84--91% active-row accuracy. PRESENTED
reaches only 17--34% active-row accuracy and corrupts observed facts. The
failure is therefore the optimization interface of imposing doubly stochastic
closure during learning, not the rendered-language parser.

Report SHA-256 is
`d525fbe5aac6d6622575324775b979b6673638091c12829dbd59eb3df92c11e4`.
Joint Sinkhorn projection closes without duration, width, seed, temperature,
renderer, or loss variants.
