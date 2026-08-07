# DIVERGE-QPT1: Qwen3.5-4B Pointer-Transaction Gate

Status: frozen before any QPT1 CUDA score.

## Objective

QST1 improved math, science, and logic on Qwen3.5-0.8B but failed broad
promotion because its unconditional 20-token workspace prefix regressed code
and produced only a `+1.007` five-domain macro lift. Its learned halt also
closed almost completely by the end of training, reducing effective recurrent
depth.

QPT1 tests a structurally different mechanism on exact pinned
`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`:

1. query-conditioned straight-through pointers select complete prompt
   observations into eight source slots;
2. each of eight tied recurrent steps reads one whole source slot and writes
   one whole state slot;
3. fixed recurrent depth replaces learned early halting;
4. four query-owned outputs are injected into existing terminal prompt tokens
   through a zero-initialized gated residual, preserving sequence length and
   positional geometry;
5. the final-four-layer rank-8 LoRA path remains identical to the matched B1
   control.

The model owns every pointer, transaction, gate, and answer. There is no
external solver, tool call, answer router, host arithmetic, or teacher at
inference.

## Frozen Development Budget

- model revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`;
- V10 data SHA-256:
  `2461d6f70b44a142854d56c24e1fb42d600065e5788a2c4e055ba47b12696549`;
- 26,387 selected rows, data seed `20260802`;
- model seed `2026080702` for both QPT1 and B1;
- 256 optimizer updates, 16 logical examples/update;
- maximum sequence length 1,024;
- AdamW, cosine schedule, peak learning rate `2e-4`;
- final four text layers, LoRA rank 8, alpha 16;
- QPT1 width 512, 8 source/state slots, 4 query slots, 8 recurrent steps;
- one H100 per arm; batch and accumulation may change only to fit memory while
  preserving 16 logical examples/update.

A two-update canary may determine memory-fit batch geometry. It is not a
capability score. QPT1 and B1 then run once at the frozen 256-update budget.

## Development Decision

Evaluate checkpoint 256 identically on the existing GSM8K-100, MATH-100,
HumanEval-20, MBPP-20, GPQA-198, BBH-100, and AIME-30 boards. The five-domain
macro treats HumanEval plus MBPP as one 40-example code domain.

Promotion requires all of:

- at least `+3.0` five-domain macro points over matched B1;
- at least `+15` additional solved examples across the five-domain board;
- improvement in at least three domains;
- no domain regression greater than two percentage points;
- finite training, protected-weight hash identity, and no evaluator change;
- after a score pass, packet swap and state reset must materially reduce the
  gained answers, while release-off must recover the matched LoRA path.

A miss closes exact QPT1 without pointer-temperature, width, duration, seed,
loss-weight, or threshold variants. A pass authorizes one larger-data
continuation and a fresh benchmark milestone; it does not by itself establish
general reasoning.
