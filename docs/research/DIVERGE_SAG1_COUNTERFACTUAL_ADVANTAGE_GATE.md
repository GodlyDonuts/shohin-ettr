# DIVERGE-SAG1: Counterfactual Advantage-Gated Reasoning

Status: frozen before any SAG1 capability score. The two-update CUDA canary
passed on 2026-08-07 without opening an evaluation board.

## Hypothesis

QPT1 adds substantial math, science, and logic capability to Qwen3.5-4B, but
its residual is globally available and code falls from `30/40` to `26/40`.
SAG1 tests whether negative transfer can be removed without discarding the
reasoning expert:

1. load the qualified B1 adapter and freeze it bit-identically;
2. train a separate query-pointer transaction expert;
3. compare frozen-base and expert teacher-forced loss per training example;
4. train a prompt-only router against the detached counterfactual advantage;
5. at inference, hard-commit to either the complete base lineage or the
   complete expert lineage;
6. when the router abstains, embeddings and generation are exactly B1.

There is no logit averaging, fieldwise state averaging, external solver,
answer-label access, or evaluator-time task router. The router sees only the
prompt representation. The mechanism differs from QPT1 by adding a protected
null lineage and training the commit policy on measured within-example
advantage rather than exposing the expert residual to every prompt.

## Frozen Development Run

- host: `Qwen/Qwen3.5-4B` revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`;
- protected B1 checkpoint SHA-256:
  `f7354e6a0c4311ad792b73358b4e62d9dbe0ae1bd2d41896cf55482d9ce81feb`;
- V10 data SHA-256:
  `2461d6f70b44a142854d56c24e1fb42d600065e5788a2c4e055ba47b12696549`;
- 256 updates, 16 examples/update, context 1,024, seed `2026080711`;
- AdamW cosine schedule, peak LR `2e-4`;
- QPT geometry: width 512, 8 source/state slots, 4 query slots, 8 tied
  recurrent steps, hard-pointer temperature 0.5;
- router: prompt mean -> 256 hidden -> one logit, bias `-2`, threshold 0.5;
- detached advantage margin 0.02;
- router/risk/sparsity weights `0.20 / 0.50 / 0.01`;
- B1 host and LoRA are frozen; only the expert workspace and router train.

The original qualified B1 is a protected no-harm reference. A separate
equal-exposure control starts from the same protected B1 checkpoint and runs
another 256 ordinary B1 LoRA updates on the same V10 stream, data order,
learning-rate schedule, context, and seed as SAG1. This control is not a new
architecture candidate. It exists to prevent the second 256-update stage from
being mistaken for a SAG1 gain. It is matched for examples and target-token
exposure, but not parameter count or FLOPs; any development pass is therefore
a product gate rather than a parameter-matched causal architecture claim.

One two-update canary may verify exact B1 identity, finite dual-path loss,
checkpoint reload, and memory. It is not a capability score. A canary miss
closes this implementation before the development run.

## Development Gate

The existing 538-example board is now a development board because QPT1 used
it. The original B1, equal-exposure B1 continuation, and SAG1 are evaluated
once with identical prompts, generation settings, and scoring. SAG1 must:

- retain at least `30/40` executable code answers; on this 40-task board even
  one lost answer exceeds the frozen two-percentage-point regression limit;
- exceed the original B1 by at least three five-domain macro points and 15
  solved answers, while improving at least three domains;
- exceed the equal-exposure B1 continuation by at least three five-domain
  macro points and 15 solved answers, improve at least three domains, and
  regress no domain by more than two percentage points;
- leave the frozen B1 parameter hash unchanged;
- show nontrivial but non-universal expert commitment (`5%--95%`), measured
  as the mean over all fixed-interval training-trace samples. Batch size is
  one, so a single final row is necessarily binary and is not a valid rate.

A development pass authorizes one source-disjoint larger-board confirmation
and a transplant to exact pinned Qwen3.5-9B. A miss closes exact SAG1 without
router-threshold, margin, loss-weight, width, duration, seed, or task-label
variants. Existing QPT1 results and controls remain closed.
