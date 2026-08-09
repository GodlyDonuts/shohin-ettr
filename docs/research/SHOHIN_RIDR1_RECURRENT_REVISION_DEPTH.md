# RIDR1: Recurrent Revision Depth

Status: closed negative on development; holdout remains sealed.

## Hypothesis

The qualified Qwen3.5-9B IDR1 reviser may implement an iterative correction operator rather than a single-use rewrite. Applying the exact same trained reviser to its own completed first revision could repair residual errors without new weights, training data, sampling, tools, or external verification.

## One Changed Factor

Depth one is the immutable IDR1 development result (`589/1,289`). Depth two replaces the original internal draft with the exact depth-one completion for the same identity. Model root, B1 warm start, IDR1 checkpoint, source, prompt constructor, greedy decoding, 768-token budget, evaluator, seed, and batch geometry remain fixed.

Only development is opened. A pass may unlock exactly one depth-two holdout run. A failure closes recursive depth for this reviser; there is no depth-three, prompt, seed, token-budget, or threshold rescue.

## Frozen Gate

All conditions are conjunctive:

- depth-one receipt exactly reproduces `589` overall, `223` math, `349` logic, and `17` code;
- depth two reaches at least `615/1,289`, a gain of 26 answers (2.02 percentage points);
- math, logic, and code are each nonnegative versus depth one;
- at least 98% of depth-one-correct examples remain correct.

The comparison reports repaired errors, newly broken correct answers, persistent errors, preserved correct answers, generated tokens, token exhaustion, peak memory, aggregate GPU time, and critical-path time.

## Interpretation

A pass supports useful model-owned recurrent inference depth for this trained reviser. It does not establish indefinite convergence or architectural novelty. A failure means the learned rewrite is not a stable iterative operator under this exact contract.

## Result

Eight single-H100 shards (`747426--747432`, `747442`) completed the exact
development pass. The first shard-7 allocation (`747433`) had no visible GPU;
two replacement submissions (`747436`, `747440`) failed before model execution
because the manually supplied model-manifest hash had one extra character.
Debug allocation `747441` identified that custody typo. These failures produced
no candidates and did not alter the scientific settings. Merge `747443` and
comparison `747444` completed cleanly.

Depth two scored `539/1,289`, down 50 answers from depth one's `589`. It repaired
15 depth-one errors but broke 65 depth-one-correct answers, retaining only
`524/589 = 88.96%` of correct answers. Every domain regressed:

| Domain | Depth one | Depth two | Delta |
|---|---:|---:|---:|
| Math | 223 | 192 | -31 |
| Logic/science | 349 | 331 | -18 |
| Code | 17 | 16 | -1 |
| Overall | 589 | 539 | -50 |

All capability gates fail. The merged evaluation used 17,035 generated tokens,
625.83 aggregate GPU-seconds, a 106.17-second generation critical path, two
token-exhausted outputs, and 19.877 GB peak allocated memory. Comparison
SHA-256 is
`cccc45c0c517df597c24fb64dc1cfc8aa41b0721806fde61987d013e00989ca1`;
merged report SHA-256 is
`755322624182f0b307f7f21b2bfeb44eb9461a8b99804708cba7d5080554e045`.

RIDR1 is closed without depth-three, prompt, seed, budget, or threshold rescue.
A future multi-stage design must train each later owner on the distribution of
completed earlier revisions and must earn conservative answer retention.
