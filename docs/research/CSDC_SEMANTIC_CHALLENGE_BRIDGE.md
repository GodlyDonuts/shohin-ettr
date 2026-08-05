# CSDC Semantic Challenge Bridge

Status: completed mixed negative; parser closes under the frozen gate.

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

## Result

Immutable job `739385` ran commit `152d42f` on one H100 for 1,500 updates,
192,000 examples, and 408.458 training seconds. The parser contains 75,912
trainable parameters. The CSDC reasoner and seed-47 row-local compiler stayed
frozen.

| Metric | Development | Held-out renderer |
|---|---:|---:|
| End-to-end exact answer | 95.573% | 93.896% |
| Oracle CSDC answer | 99.593% | 99.723% |
| Challenge record | 100.000% | 100.000% |
| Start state | 100.000% | 100.000% |
| Outcome state | 100.000% | 99.996% |
| Word length | 100.000% | 99.994% |
| Complete ordered word | 89.290% | 84.153% |
| Complete challenge tuple | 89.290% | 84.145% |
| Selected presentation | 90.706% | 86.377% |
| Shuffled outcome answer | 57.503% | 54.671% |
| Swapped lineage answer | 13.525% | 13.167% |

The parser clears the answer and causal-use conditions on development, but it
misses the 95% complete-word and selected-presentation thresholds. The held-out
renderer clears 90% answer accuracy but misses the five-point oracle-gap and
90% tuple thresholds. Most remaining error is concentrated in ordered
generator-word serialization: cyclic cohorts are 100%, while the held-out
random-permutation cohorts reach only 86.426% and 83.496% answers.

This is strong evidence that the protected CSDC mechanism can operate from
learned rendered source records. It is still controlled templated language,
not unrestricted natural language. Under the frozen rules this exact parser
is closed without width, duration, seed, template, or loss-weight variants.
CSDC remains the protected typed-source architecture result; this checkpoint
is retained as a diagnostic and possible future systems component, not a
promoted architecture milestone.

Report SHA-256 is
`c463a96a1c67f86e51540fc44352892d9e8c921b5fc22740521f24ff9d115aa8`.
Checkpoint SHA-256 is
`e70a87313f51403bbd408c84c145d76c88739b892f248f8b9e653af3f9cfc77e`.

Decision:
`close_this_semantic_parser_preserve_csdc_and_target_sequence_compilation_as_a_later_integrated_interface_problem`.

## Bounded successor

The role-gated copy successor resolves this boundary without tuning the closed
decoder. It reaches `99.593% / 99.723%` end-to-end exact answers and 100%
complete challenge tuples on development/held-renderer splits by selecting
semantic token roles and copying the ordered source tokens. See
`docs/research/CSDC_ROLE_GATED_COPY_BRIDGE.md`.
