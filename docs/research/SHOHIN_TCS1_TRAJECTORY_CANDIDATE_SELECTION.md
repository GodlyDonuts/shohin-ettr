# TCS1: Trajectory Candidate Selection

Status: CPU control complete; one semantic scorer frozen before CUDA results.

## Hypothesis

The qualified 9B depth-one owner, blind depth-two reuse, and independently
trained direct-rewrite owner solve different development identities. Their
fixed union is `615/1,289`: math `228`, logic/science `368`, and code `19`.
TCS1 asks whether model-owned candidate assessment can recover a material part
of that complementarity without a verifier, answer key, task label, external
model, or fieldwise trajectory mixing at inference.

## Frozen sequence

1. Build train-only three-candidate groups from the original base and expert
   attempts plus the actual IDR1 predecessor for 5,824 source identities.
2. Build the source-disjoint 1,289-row development pool from complete depth-one,
   depth-two, and direct-rewrite trajectories. Holdout remains unopened.
3. Fit a CPU shape-only control that cannot read lineage, task labels, gold,
   correctness, or peer outcomes. It may read only candidate text, question
   text, generated length, and exhaustion metadata.
4. Only after the CPU receipt is immutable may one semantic hidden-state scorer
   be frozen. It must use the same candidate pool and whole-trajectory argmax.

## Semantic promotion gate

The future semantic scorer must satisfy every condition:

- at least `603/1,289`, fourteen above depth one and at least 53% of the
  26-answer oracle opportunity;
- at least ten answers above the frozen shape-only control;
- math at least 223, logic/science at least 349, and code at least 17;
- at least eight-answer loss when candidate contents are deterministically
  permuted while questions and candidate positions remain fixed;
- no task/benchmark label, lineage identifier, correctness field, verifier,
  answer key, or external model at inference; and
- complete candidate, parameter, FLOP, latency, generated-token, truncation,
  and protected-checkpoint receipts.

A pass opens one source-disjoint holdout selection. A failure closes this exact
candidate-commit route without prompt, width, seed, threshold, pool, or
duration rescue. The claim is practical coherent trajectory commitment, not a
new reasoning primitive.

## CPU control result and semantic owner

The immutable candidate set contains 17,472 train candidates over 5,824
identities and 3,867 development candidates over 1,289 identities. Train and
development SHA-256 values are `622aa19e...9022` and `a216928d...5048`.
Shape-only job `747538` scores `579/1,289`: it repairs one depth-one error and
breaks eleven successes. Domains are math `214`, logic/science `348`, and code
`17`. It therefore fails every promotion condition and rules out superficial
completion-shape selection as the source of the oracle gap.

Exactly one semantic scorer is now frozen. The pinned Qwen3.5-9B IDR1 adapter
reads the raw source question and each complete candidate independently using
the fixed correctness-assessment instruction. A 512-wide scalar head and the
existing LoRA tensors train for 256 updates, eight groups per update, LR
`2e-6/2e-4`, seed `2026080903`, and maximum context 2,048. Training uses only
the 5,824 train identities and balances task/outcome-pattern strata. Loss is
balanced binary correctness plus within-group positive-over-negative ranking.
Inference takes an argmax over three complete trajectories; scores or fields
are never averaged. The source checkpoint remains immutable.

One one-update mechanics job may run first. If finite, non-OOM, hash-safe, and
fully receipted, it unlocks the single 256-update fit above. There is no
shuffled-label, width, prompt, layer, seed, or duration family. Development is
read once after training; holdout remains sealed behind the frozen gate.
