# RSOT1: Role-Separated On-Policy Transduction

**Status:** prospectively frozen after OCET1 development failure, before RSOT1 output, 2026-08-10  
**Scope:** opened source-disjoint development board; holdout sealed

## Evidence and hypothesis

OCET1 closes at `1690/1908`. Its aligned trained owner generates only 1,304
correct mandatory-rewrite proposals, then improves those same proposals to
1,690 after the commit transition. The on-policy objective therefore learned a
causal repair policy (`+386` within the aligned trajectory), but sharing that
adapter between proposal and commit damaged the previously qualified proposal
role. The controls remain far lower: permuted `940`, hidden `623`.

RSOT1 tests one structural correction, not a training variant: preserve the
immutable qualified DSET proposal owner and assign the OCET-trained owner only
to commit. The two roles share the same frozen Qwen3.6-35B-A3B base but use
separate 1,179,648-parameter adapter states. No new weights or updates are
authorized.

## Frozen execution

1. Reuse the exact eight immutable aligned FRET1 proposal shards from the
   qualified DSET owner. They cover all 1,908 development presentations.
2. Run three commit owners over the byte-identical proposals:
   - aligned OCET owner;
   - within-pair label-permuted OCET control;
   - exact draft-hidden OCET control.
3. Use the same prompt, generic edit executor, greedy decode, 32-token budget,
   source board, tokenizer, and evaluator as OCET1.
4. Report proposal, commit validity, final correctness, choice/numeric,
   clean/fault, exhaustion, hashes, wall time, and compute.

The runtime receives no verifier, answer label, solver, or host repair. The
generic executor only applies the model-emitted KEEP or REPLACE_LAST script.

## Frozen gate

All conditions are conjunctive:

- aligned final `>=1874/1908`;
- choice final `>=220/256`;
- clean final `>=945/954`;
- fault final `>=859/954`;
- valid commits `>=95%`;
- aligned is at least 13 answers above ISET1 (`1838`), permuted, and hidden;
- zero aligned decode exhaustion.

One pass opens one sealed holdout. A miss closes this exact role-separated
system without adapter rank/layer/LR/update/seed/prompt/decode variants.

## Claim boundary

Separate proposal and verifier/editor roles are not novel. RSOT1 is a practical
architecture test of whether Shohin's model-owned on-policy repair capability
becomes operational once destructive cross-role parameter sharing is removed.
