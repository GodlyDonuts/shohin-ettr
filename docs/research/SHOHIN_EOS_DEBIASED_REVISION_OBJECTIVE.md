# Shohin EOS-debiased revision objective

Status: code-only preparation; no job submitted and no measured result.

## Causal motivation

The exact Qwen3.5-9B IDR1 training replay contains 9,655 presentations from
5,824 unique sources. Of the executed 2,048 microsteps, 702 are
`source_verified_repair` / `both_wrong` targets. They contribute only 7,008 of
365,028 charged response tokens, yet the terminal EOS token accounts for an
average 13.09% of their within-row loss, versus 0.77% for verified-candidate
targets: a 17.09x imbalance. These answer-only repairs have a median response
length of 11 characters against a 1,776-character model-owned draft.

The prepared `eos_debiased_revision` loss changes one boundary only: for those
answer-only `both_wrong` repairs, it masks the final EOS label while preserving
every answer token, the complete prompt and draft, the EOS token in the input,
the sampled row order, and the charged-token accounting. All other targets
retain standard causal-language-model supervision. The implementation
fail-closes unless the row kind/outcome binding is exact and the tokenizer has
actually appended its pinned EOS token.

## Prospective matched test

The treatment and standard-loss control must start from the same immutable B1
warm start and use the same selected rows, presentation order, seed, data seed,
batch size 1, gradient accumulation 8, 256 updates, optimizer, learning rate,
LoRA geometry, model, prompts, and decoding. The predicted executed difference
is exactly 702 masked terminal labels:

- charged response tokens: 365,028 in both arms;
- supervised response tokens: 365,028 standard, 364,326 EOS-debiased;
- all nonterminal answer-token labels: identical;
- source-disjoint evaluation identities and matched unchanged,
  self-refinement, and draft-hidden controls: identical.

This is not a teacher-trace retry, synthetic-fault retry, aligned-natural-draft
retry, post-hoc selector, or inference-time benchmark router. It directly tests
whether a row-normalized terminal-stop objective caused the observed
answer-only revision collapse. Promotion requires broad source-disjoint
capability improvement without worsening conservative retention. Confirmation
data remains untouched until that development gate passes.

## Evidence boundary

- training-horizon report SHA-256:
  `866e0904199b28e6f121d3d711f1fb93e3872904896a0a8ad56ddfff2e2d37e7`
- IDR1 training data SHA-256:
  `6df3204573ce807db1b5057bce709189366b6674e38e5224ee3d17a3e6f0ac6c`
- original revision report SHA-256:
  `457eccd13a4998510d774cf3e84f347b30480e017a6ad5e0eff4a24b368101c2`
- tokenizer SHA-256:
  `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`
- EOS token / ID: `<|im_end|>` / `248046`
- launch authorized: false

