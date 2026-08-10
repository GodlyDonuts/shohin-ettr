# DTMC1: Draft-Conditioned Typed Microcode Compiler

Status: draft generation and prospective gate frozen; no DTMC1 fit exists

Date: 2026-08-10

## Hypothesis

TMC1 proved that a typed source compiler is causal but recovered only 45.46%
of operations and 32.83% of operands from one frozen question encoding. The
same direct owner solves 40.09% only after autoregressive generation. DTMC1
tests whether that generated trajectory externalizes the missing semantic plan
in a form a typed result-free compiler can use.

The system has two model-owned stages:

1. the immutable NMC1 direct-CoT owner greedily generates one exact draft from
   the source; and
2. a fresh typed graph compiler reads source plus that complete draft and emits
   the same result-free causal graph used by TMC1. Frozen learned LAM1 executes
   it.

At inference there is no gold chain, verifier, solver, answer label, host
repair, or task router. Candidate source pointers remain restricted to the
source segment; the draft supplies semantic trajectory context, not pointer
answers. The direct draft may contain arithmetic results, but the typed target
and compiler output contain no result field or final answer. LAM1 still owns
execution.

## Frozen draft corpus

Generate one greedy, no-thinking, maximum-512-token draft for each of the 6,333
training identities using direct checkpoint SHA-256 `8a2b6550...0b53`, pinned
Qwen revision `2fc06364...8b17`, seed `2026081053`, and the same system/user
prompt used by the frozen direct development evaluation. Eight identity-modulo
shards are allowed only for wall-clock parallelism. Every row, including an
exhausted or wrong draft, is retained. Merge requires exact identity coverage,
zero duplicates, immutable shard hashes, and no public-test access.

Development uses the already generated direct report SHA-256
`234a029a...a49`; drafts may not be regenerated or selected.

## Frozen fit

DTMC1 reuses the exact TMC1 24,864,055-parameter compiler geometry and frozen
semantic owner. Only the source representation changes from `source` to
`source + exact model-owned draft`, with segment custody preserving numeric
pointer masks over source only. Initialization seed, 4,096 updates, batch 32,
LR `2e-4`, optimizer, loss components, data order, and LAM executor remain
identical to TMC1.

The immutable input envelope is `PROBLEM:\n{source}\n\nMODEL-OWNED
DRAFT:\n{draft}` under the same system prompt and no-thinking chat template.
The context ceiling is 1,024 tokens and truncation is forbidden. This is the
smallest increase from TMC1's 512-token question-only ceiling that can retain
the frozen 512-token drafts. Token offsets for candidate numeric owners must
intersect the `PROBLEM` source bytes only; numeric strings in the draft are
never pointer candidates.

## Development controls and gate

Evaluate exactly once on all 666 existing source-disjoint identities:

1. aligned source plus its exact owner draft;
2. source plus a within-register-depth shuffled owner draft;
3. same-depth shuffled source plus that donor's aligned draft; and
4. frozen direct owner (`267/666`) and question-only TMC1 (`44/666`) as fixed
   references.

All gates are conjunctive:

- aligned answers at least `301/666` (45.195%), which exceeds direct by at
  least five absolute points;
- operation and operand-owner accuracy each at least 80%;
- aligned exceeds shuffled-draft by at least ten points;
- source+draft shuffle scores at most 10%;
- carry reset loses at least ten points on multi-digit aligned-correct rows;
- opcode permutation loses at least 30 points; and
- zero invalid graphs or executions.

Failure closes exact DTMC1 without draft count, sampling, rank, width, layer,
duration, seed, loss, or prompt variants. A pass opens one unchanged public
GSM8K test evaluation of direct and DTMC1.

## Claim boundary

DTMC1 is temporal program compilation, not proof of general reasoning or a
novelty claim for draft-and-revise systems. A pass would specifically show
that a model's own autoregressive trajectory provides causal semantic state
that a typed learned-execution path can convert into improved exact answers.
