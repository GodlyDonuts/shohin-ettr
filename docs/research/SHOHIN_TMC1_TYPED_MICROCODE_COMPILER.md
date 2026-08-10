# TMC1: Typed Microcode Graph Compiler

Status: CPU representation/mechanics gate frozen; no neural output exists

Date: 2026-08-10

## Causal hypothesis

NMC1 failed before arithmetic: free-text generation lost register causality,
commit identity, and stack grammar. TMC1 tests whether semantic planning is
present but cannot be serialized reliably. It replaces textual program output
with a model-owned typed computation graph. A generic executor alone enforces
grammar and causality; it does not select operations, operands, constants, or
answers.

Each graph instruction contains one operation (`ADD`, `SUB`, `MUL`, `DIV`,
unary `NEG`, or identity `COPY`) and typed operands. `COPY` only materializes
identity/alias equations as state and performs no semantic repair. An operand
is either a pointer set over equal numeric spans in the source, one strictly
prior state pointer, or a signed rational literal represented by fixed digit
fields. Repeated equal source mentions remain an equivalence set rather than
receiving an arbitrary owner. The final answer is a pointer to one computed
state. No instruction contains an intermediate result or final-answer label.

## Stage 0: exact CPU gate

Lower every immutable NMC1 gold register program into binary typed
instructions. The train-derived maxima for source spans, instruction count,
and literal digit widths become the only neural schema. Development may not
increase them. The gate requires:

1. exact Fraction execution on all 6,333 train and 666 development rows;
2. exact frozen LAM1 execution on every row;
3. every final owner is a computed state;
4. zero missing source/state/literal owners; and
5. development fits the train-derived schema without truncation.

A miss closes this representation before GPU training.

## Conditional neural canary

Only a CPU pass may freeze the concrete tensor dimensions. The semantic owner
is the immutable matched direct-CoT checkpoint from NMC1 (`8a2b6550...0b53`),
which scored 267/666. Its Qwen3.5-0.8B weights and LoRA remain frozen. A
separate recurrent typed decoder cross-attends its source token states and
emits operation, operand-kind, source-pointer-set, causal state-pointer, and
literal-digit heads. Hard masks make forward state references and invalid
instruction shapes impossible. Teacher forcing is allowed only during
training; development is fully autonomous. Frozen LAM1 executes the graph.

The first fit is bounded to one architecture, seed, and schedule frozen after
the CPU receipt and before neural output. Development must satisfy all of:

- at least 45% exact answers and at least +5 absolute points over the frozen
  40.090% direct owner;
- at least 80% operation and operand-owner accuracy;
- at most 10% exact answers under same-geometry source shuffle;
- at least 10-point carry-reset loss on multi-digit normal-correct rows;
- at least 30-point opcode-permutation loss; and
- zero invalid graphs, causal-reference violations, or slot exhaustion.

Failure closes exact TMC1 without width, layer, duration, seed, loss, pointer,
or schema variants. A pass opens one unchanged public GSM8K test evaluation.

## Claim boundary

TMC1 is a structured program compiler test, not a broad-reasoning claim. The
generic grammar mask and executor are explicit scaffolds. A pass would show
that model-owned natural-language semantics can populate a result-free causal
graph that learned arithmetic microcode executes. It would not make typed
decoding, pointer networks, or program execution novel.
