# TMC1: Typed Microcode Graph Compiler

Status: CPU mechanics passed; neural development gate failed; exact TMC1 closed

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

Job `749894` passed in seven seconds. Exact Fraction and learned-LAM execution
are `6333/6333` train and `666/666` development. Train maxima are 15 source
spans, 15 instructions, ten numerator digits, and three denominator digits;
development maxima are 11/12/5/3. Report SHA-256 is `f2b71afd...ab08`.

The semantic owner is the immutable matched direct-CoT checkpoint from NMC1
(`8a2b6550...0b53`), which scored 267/666. Its Qwen3.5-0.8B weights and LoRA
remain frozen. A 24,864,055-parameter causal typed slot decoder uses width 512,
two source-encoder layers, four causal decoder layers, and eight heads. It
cross-attends the frozen source token states and emits instruction count,
operation, source-equivalence/state/literal references, and literal digits.
Hard masks make forward state references and invalid instruction shapes
impossible. There is no teacher-forced instruction stream; development is
fully autonomous. Frozen LAM1 executes the graph.

The sole fit uses 4,096 updates, batch 32, AdamW (`betas=(0.9,0.95)`, weight
decay 0.01), learning rate `2e-4`, maximum source prompt 512 tokens, seed
`2026081061`, and data seed `2026081062`. Length, operation, left reference,
right reference, and aggregate literal-digit losses each contribute one
normalized component. No result/value/answer loss exists.

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

## Neural result

Fit `749895` completed all 4,096 updates in 859.33 seconds over 131,072
presentations / 12,651,233 source tokens. Final aggregate loss was 0.8836;
checkpoint SHA-256 is `a8742fca...1336`. Frozen evaluations `749923/749924`
completed in 22/20 seconds.

Normal development reaches `44/666 = 6.6066%` answers, `26/666 = 3.9039%`
exact graphs, `290/666 = 43.5435%` instruction counts, `1203/2646 = 45.4649%`
operations, and `1735/5285 = 32.8288%` operand owners. One graph produces an
invalid normal execution. Same-geometry source shuffle falls to `5/666`,
opcode permutation to `2/666`, and carry reset retains only `8/36` multi-digit
normal-correct rows. Thus the typed compiler is source- and executor-causal,
but remains far below the frozen direct owner at `267/666 = 40.0901%`.

The public test remains unopened. Exact TMC1 closes without width, layer,
duration, seed, loss, pointer, or schema variants. The evidence localizes the
next change: grammar is no longer the bottleneck; a static question-only hidden
state does not expose the autoregressive semantic plan. A successor may use an
exact model-owned draft trajectory as a second source stream, with aligned,
shuffled-draft, and source-shuffled controls.
