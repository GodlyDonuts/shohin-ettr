# RIDR1: Recurrent Revision Depth

Status: frozen development gate; no capability output opened.

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
