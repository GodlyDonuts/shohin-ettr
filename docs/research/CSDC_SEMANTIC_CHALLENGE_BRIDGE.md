# CSDC Semantic Challenge Bridge

Status: gate frozen; implementation and Newton CPU smoke pass; H100 gate
pending.

## Objective

Confirmed CSDC reasons over typed source challenges. This gate removes that
shortcut. A neural semantic compiler must recover every challenge's start
state, ordered generator word, word length, and outcome from shuffled rendered
records. The frozen CSDC candidate constructor, falsifier, and executor then
consume only those predictions.

The row-local presentation compiler and its seed-47 checkpoint remain frozen.
The semantic bridge receives no query, answer, selected presentation, or
terminal-state supervision. Its only labels are source-record semantics.

## Renderer split

Training uses three randomized source templates with varying field order and
record order. Development uses new episodes and those three template families.
Confirmation uses a fourth held-out field order with unchanged parser weights.
Observation statements retain their existing two templates so the experiment
isolates challenge semantic compilation rather than retraining the local table
compiler.

## Fixed comparisons

- `ORACLE_CSDC`: protected typed-source ceiling;
- `LEARNED_CSDC`: parser-decoded challenge fields;
- `DWPC`: closure without challenge use;
- `SHUFFLED_PARSED_OUTCOME`: learned CSDC with decoded outcomes exchanged
  across episodes; and
- `LINEAGE_SWAP`: learned CSDC with the selected complete presentation
  exchanged before the late query.

## Pass and kill rules

The one-seed bridge advances only if development satisfies all of:

1. at least 95% end-to-end OOD macro and within five points of ORACLE_CSDC;
2. at least 95% exact challenge-record identification;
3. at least 95% exact start, outcome, length, and complete ordered word;
4. at least 95% selected-table exactness; and
5. at least 20-point losses under both causal interventions.

The held-out renderer must then reach at least 90% end-to-end macro, stay
within five points of its oracle ceiling, retain at least 90% exact challenge
tuples, and preserve both intervention directions. Failure closes this parser
without width, duration, seed, template, or loss-weight variants. CSDC itself
remains valid because its typed-source confirmation is independently frozen.

## Resource envelope

The parser is trained in one single-H100 job for 1,500 updates at 128 examples
per update. The same job evaluates development and held-out renderers. Hard
ceiling is one H100-hour; expected use is below 0.25 H100-hour. No Shohin-scale
integration follows from parser training loss alone.
