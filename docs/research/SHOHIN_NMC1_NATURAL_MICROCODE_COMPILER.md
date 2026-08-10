# NMC1: Natural-Language Microcode Compiler

Status: frozen prospective data and development gate

Date: 2026-08-10

## Objective

LAM1 proves that learned finite decimal transitions compose exactly once a
correct postfix program exists. NMC1 tests the missing semantic interface: can
a pretrained language model compile a natural word problem into a compact,
result-free program that the frozen learned microcode executes more reliably
than an equal-update ordinary chain-of-thought SFT control?

This is structurally different from BTT/WGP. The source is a natural word
problem, not an explicit arithmetic expression. A pinned Qwen3.5-0.8B source
compiler emits a register microprogram. The target contains operations, source
constants, and references to earlier computed registers, but omits every
stated equation result and the final answer. At inference the frozen LAM1
microcode executes the generated program. There is no verifier, solver, answer
label, teacher, or host repair.

## Frozen data

- source: immutable GSM8K train JSONL SHA-256
  `e0a44460964649db791a0ced18449b34c34875218b7d21b23241cbdb6ca2a104`;
- overlap reference and later public benchmark: GSM8K test JSONL SHA-256
  `752adc99f23132275139f1f7be57126b08735a55daf0c56859ee6df70abafbac`;
- accept only rows whose complete `<<expression=result>>` annotations parse
  into exact `+`, `-`, `*`, and `/` arithmetic and whose final register equals
  the `####` answer;
- reject rows with no complete equation program or unsupported operators;
- replace an operand by the latest prior register whenever its exact value is
  already available;
- split admitted train identities by normalized-question SHA-256 modulo 10:
  remainder zero is development and all others are training;
- require zero normalized-question overlap between train, development, and
  public test.

The program serialization contains no computed record result and no final
answer field. An independent Fraction assessor and frozen learned-digit
executor must agree on every admitted gold program before GPU training.

## Frozen model and fits

Model is pinned `Qwen/Qwen3.5-0.8B` revision
`2fc06364715b967f1860aea9cf38778875588b17`, locally hash-bound. Both arms use
fresh final-four-layer rank-8 all-projection LoRA, alpha 16, learning rate
`2e-5`, batch 4, gradient accumulation 2, maximum sequence 1,024, exactly
1,024 updates, seed `2026081051`, and data seed `2026081052`. Token admission
must prove that every complete prompt and target, including EOS, fits without
truncation in both arms before either fit starts.

Arms:

1. **NMC1:** source to result-free register microprogram, followed by frozen
   LAM1 execution.
2. **Direct control:** same source and admitted identities, ordinary original
   GSM8K chain-of-thought/final response, with identical model, LoRA geometry,
   updates, batch, optimizer, and decoding budget.

Actual non-padding training tokens, elapsed time, peak memory, trainable
parameters, and generated tokens are reported. The direct arm is allowed to
consume more target tokens; NMC1 receives no compute credit for being shorter.

## Data and mechanics admission

CPU job `749812` admitted 6,333 matched training identities and 666
source-disjoint development identities. They contain 22,846 computation
records and 75,242 actions. Rejections were 95 rows without a complete
equation program, two unsupported floor-division rows, and 377 rows whose
annotation did not end in the final answer. Program/direct train SHA-256 are
`71010bac117d9fbca2a28e7f9a63f24456b519a7b4bc172ddd772ff27e851512` /
`b78fa53081114f1b5c08f34c0f26df641dfd975a4975130a6b0c8098a0944757`;
development SHA-256 is
`981b83016d9a895af3016b3629c5852b868c80e364dbe97e3901a8d3e8ced4bd`.

Token audit job `749813` retained every row under the frozen 1,024-token
limit. Maximum complete sequence is 363 tokens for NMC1 and 574 for direct
CoT. Mechanics job `749814` obtains exact Fraction/LAM parity on all
`6333/6333` train and `666/666` development programs with zero normal
invalidity. On gold development programs, carry reset scores `116/666` and
opcode permutation `13/666`. Both prerequisite gates pass.

## Development gate

Before any public-test score is opened, all conditions are conjunctive on the
source-disjoint development split:

1. gold-program Fraction/LAM parity is 100%, with zero invalid or overflow;
2. NMC1 syntax validity is at least 90%;
3. NMC1 execution validity is at least 85%;
4. NMC1 final-answer accuracy is at least 60%;
5. NMC1 beats the matched direct control by at least 3 absolute points;
6. same-family/depth source shuffle scores at most 10%;
7. carry reset loses at least 10 points on multi-digit normal-correct rows;
8. opcode permutation loses at least 30 points overall; and
9. no admitted target serializes any stated equation result or final answer.

Failure closes exact NMC1 without prompt, rank, layer, duration, seed, split,
parser, decoding, or threshold variants. A pass opens one unchanged full
public-test evaluation of both arms. The test is a standard public benchmark,
not a sealed project holdout, and will be labeled accordingly.

## Claim boundary

A development pass would show that a natural-language model can own semantic
program selection while learned recurrent microcode owns arithmetic execution.
It would not establish broad mathematical reasoning, novelty of program-aided
language models, or generalization beyond the admitted arithmetic language.
The changed factor is the result-free register interface plus learned local
execution, measured against ordinary matched SFT.
