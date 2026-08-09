# Error-Syndrome Revision (ESR1)

Status: frozen development gate; sealed holdout remains unopened.

## Motivation

SCTR1 showed that commitment is not the current bottleneck on OLMo2-7B.
Always-revise preserves 221 of 222 correct internal drafts and repairs 38 of
1,067 incorrect drafts. The union of draft and revision contains only one
additional correct answer beyond always-revise, so a perfect selector has a
measured ceiling of 0.0776 percentage points. ESR1 therefore targets the
ability to construct a correction, not the decision to keep one.

## Mechanism

The source-only prompt contains the problem and the model's own internal
draft. An eight-step tied recurrent workspace produces 16 model-owned soft
prefix states. Standard causal language-model loss trains the full corrected
response. ESR1 adds one fixed auxiliary target derived only during training:

```text
s = mean(E[verified response tokens]) - mean(E[internal draft tokens])
L_syndrome = 1 - cosine(mean(workspace prefix), s)
L = L_LM + 0.01 L_halt + 0.25 L_syndrome
```

The embedding table is frozen. The exact internal-draft span is located with
token offsets; truncation that removes it fails closed. At inference the
verified response and syndrome target are absent. The workspace receives only
the same problem-plus-draft prompt as always-revise and emits a soft prefix
before one coherent replacement trajectory.

This is a bounded capability experiment, not a novelty claim. It tests whether
explicitly supervising the direction of correction makes a recurrent latent
workspace materially better at revision.

## Matched Arms

1. `syndrome`: recurrent workspace plus fixed syndrome loss.
2. `ettr`: identical recurrent workspace, parameters, recurrent depth, LoRA,
   data, update count, seed, and evaluation, without syndrome loss.
3. `always_revise`: completed standard LoRA reviser from the same OLMo2-7B,
   data, update count, and evaluator.

Both new arms use 9,655 frozen training examples, 256 optimizer updates,
batch size 1, accumulation 8, sequence length 4,096, LoRA rank 8 on four late
layers, learning rate 2e-5, 16 workspace slots, width 512, and eight recurrent
steps. Development has 1,289 identities spanning MATH-500, BBH logic/science,
and executable MBPP. Eight exact evaluation shards are merged before scoring.

## Frozen Gate

ESR1 passes only if all conditions hold:

- syndrome accuracy is at least 5.0 points above always-revise;
- syndrome accuracy is at least 3.0 points above the same-workspace control;
- correct-count deltas versus always-revise are nonnegative in math, logic,
  and code;
- all 1,289 development identities are covered by eight merged shards.

Only a conjunctive pass authorizes the sealed holdout. A failure closes this
exact objective without weight, width, duration, seed, prompt, or threshold
variants.
