# BSOT1: Best-State On-Policy Transduction

**Status:** prospectively frozen after RSOT1 development failure, before BSOT1 output, 2026-08-10  
**Scope:** opened source-disjoint development board; holdout sealed

## Evidence and hypothesis

RSOT1 separates proposal and commit ownership and proves a causal aligned
commit effect (`1700 -> 1798`) over byte-identical proposals, while swapped
and hidden controls remain at `1700`. It nevertheless finishes 40 answers
below the existing ISET1 owner (`1838`) and fails choice and clean gates.

BSOT1 performs one final no-training composition test. It replaces only the
proposal artifact with the immutable best independent ISET1 model-owned
trajectory for each identity. The aligned, swapped, and hidden OCET commit
owners then receive the same byte-identical proposal. This tests whether the
learned repair owner becomes useful above a proposal-quality floor. It is not
an adapter, prompt, checkpoint, seed, duration, or decoding retry.

## Immutable inputs

- ISET1 aligned merged proposals: 1,908 identities, `1838` exact, SHA-256
  `99623156d88d49265c6d9be363718c1c3c659f6da6f35f05c5e12c6e65c33e10`.
- Aligned OCET checkpoint SHA-256:
  `29e642b77265272633aa72c478cda4c540c8bd65d824006059db7a6784be9a5b`.
- Swapped OCET checkpoint SHA-256:
  `221c1edf6c7e52c2fedc0fd972bc134d9a06f9b8603b9ce4e23791df60230e87`.
- Hidden OCET checkpoint SHA-256:
  `05135963b0033f594dfbe9b61cd6ef10313fa6c6753ab74947690f08bd7d5c30`.
- Diagnostic data/report SHA-256:
  `62f9d3c52427d91913ee69a0f87ca2d9230e314b8fe795fcc11773f4fe06c445`
  and `f371d65fa9d973bf5c03083338803f348533b7e3e5122905b0c6b5dda92fe9f1`.

## Frozen execution

1. Validate exact ISET schema, development-only status, data/report hashes,
   1,908 unique identities, and a complete executed trajectory per identity.
2. Run aligned, swapped, and hidden OCET commit owners over byte-identical
   ISET trajectories, eight shards per arm.
3. Preserve the same prompt, generic edit executor, greedy decode, 32-token
   budget, tokenizer, source board, and evaluator used by RSOT1.
4. No verifier, answer label, solver, host repair, candidate selection, or
   fallback exists at inference.

## Frozen gate

All conditions are conjunctive and unchanged from RSOT1:

- aligned final `>=1874/1908`;
- choice final `>=220/256`;
- clean final `>=945/954`;
- fault final `>=859/954`;
- valid commits `>=95%`;
- aligned is at least 13 answers above ISET1 (`1838`), swapped, and hidden;
- zero aligned decode exhaustion.

One pass opens one sealed holdout. A miss closes this proposal/commit cascade
without proposal, owner, prompt, decode, seed, threshold, or recurrence
variants.

## Claim boundary

A pass would establish a practical model-owned multi-owner transducer, not a
novel edit primitive. A failure means the OCET commit owner does not add enough
reliable value even when handed the strongest existing model-owned proposal.
