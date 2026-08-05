# CSDC SmolLM2 Lexical Integration Result

Date: 2026-08-05  
Commit: `96217cb`  
Newton job: `739765` on `evc26`  
Decision: exact bridge closed; preserve the end-to-end near-pass and replace
first-subtoken copying with tokenization-invariant model-owned alias grounding.

## Result

The frozen SmolLM2 lexical adapter successfully controls CSDC's complete
hypothesis selection and late execution. It does not pass the stricter exact
interface gate because the combined unseen-template/unseen-alias split does
not recover all eight complete challenge tuples per episode.

| Metric | Development | Lexical shift |
|---|---:|---:|
| Learned exact answers | **99.691%** | **95.915%** |
| Typed-oracle exact answers | 99.691% | 99.463% |
| Exact selected tables | 99.202% | 93.197% |
| All-eight challenge tuples exact | 100.000% | **17.920%** |
| All-eight decoded fields source-valid | 100.000% | 75.472% |
| Shuffled-outcome answers | 52.604% | 53.304% |
| Whole-lineage-swap answers | 13.867% | 13.037% |
| Minimum family/depth answer cohort | 99.414% | 94.824% |

The learned path therefore remains strongly causal. Shuffling decoded
outcomes costs `47.087 / 42.611` points and swapping the selected whole world
costs `85.824 / 82.878` points. Every held family/depth cohort remains above
94.8% exact answers.

The gate still fails exactly as frozen. Twelve of thirteen conditions pass;
`shift_tuple_ge_90` fails (`17.920%`). No threshold, seed, width, duration, or
loss-weight repair is permitted for this exact bridge.

## Work and compute

- 1,500 updates
- batch 128
- 192,000 generated source-only episodes
- 8,604,806 trainable adapter/head parameters
- 134,515,008 frozen SmolLM2 backbone parameters
- 817.266 training seconds at 234.930 episodes/s
- 14:54 total one-H100 allocation including both 6,144-row evaluations
- 6,144 examples per split: three families by lengths 8 and 12

The frozen reasoner, CSDC candidate constructor, selector, commit, and late
executor were unchanged. The parser received no query, answer, selected-table,
terminal-state, or CSDC loss.

## Interpretation

The main capability question receives a positive answer: natural-language
challenge fields decoded by a real pretrained backbone can select a coherent
CSDC world and answer late queries under lexical/syntax shift. This is much
stronger than the prior list-machine lexical gate.

The exact interface question receives a negative answer. The decoder assigns
roles to one tokenizer position per alias and treats that first subtoken as
the copied identity. Under the combined unseen alias/template split, at least
one of eight challenge records is often not completely source-valid. CSDC's
redundant counterexamples tolerate those missing constraints, so table and
answer accuracy stay high while all-eight tuple exactness collapses. This
localization is consistent with a tokenization-boundary grounding failure; it
does not prove which individual field is responsible because the frozen report
records row-level complete tuples rather than post-hoc per-field diagnostics.

## Successor

Do not rerun this parser. The next mechanism is a tokenization-invariant alias
grounder:

1. predict mention boundaries/equivalence independently of semantic role;
2. quotient all subtokens of one source mention into one identity-bearing
   mention state;
3. classify `START`, `OUTCOME`, and ordered `WORD` roles over mention states;
4. copy one whole mention identity, never a privileged first subtoken;
5. keep the same frozen CSDC candidates, falsifier, commit, executor, cohorts,
   and causal controls;
6. compare against this preserved first-subtoken bridge at the same examples
   and update budget.

This is explicit grounding, not a cosmetic parser variant. Only a pass on
exact shifted tuples authorizes lexicalizing observation records and the late
query.

## Artifacts

- Report SHA-256:
  `b3ae0526e9e28ef21e93f2b32bacd7845f40cdb663d8d4b48d74ed9b7cfc05c5`
- Checkpoint SHA-256:
  `12a95731e5be263dce96a3bf13c21d3e28b55167fecae70e714228eb3a5bdcec`
- Immutable runtime archive SHA-256:
  `4328e07470e474d2eed02da14123d9d07c708ee413bd9e3b8a1012cc876e8d02`

