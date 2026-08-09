# Shohin RCR1 Revision-Conditioned Routing Gate

## Decision boundary

MTR1 is closed on development at `204/1,289`, only `+13` answers over the
unchanged second pass. Its final-four-layer shared-attention adapter changed
the frozen OLMoE route distribution by only `0.002018` mean L1. RCR1 tests a
different causal hypothesis: revision may require changing which frozen
experts process the model-owned draft, rather than adding capacity only to
shared attention.

## Frozen intervention

The host, revision data, split, prompts, target responses, optimizer, seed,
256-update budget, and exact development evaluator are inherited unchanged
from MTR1. The host is pinned
`allenai/OLMoE-1B-7B-0125-Instruct@b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`.

Treatment replaces each of the last four frozen top-k routers with

```text
logits' = logits + (alpha / rank) * tanh(B(A(h)))
```

where rank is 8, alpha is 16, `A/B` are the only trainable tensors, and the
base router and all experts remain frozen. Initial `B=0`, so update zero is
exactly the base router. The residual is bounded per expert logit.

The matched control uses rank-1 LoRA on the final four shared attention mixers.
It has 65,536 trainable parameters versus 67,584 for routing, a 3.13% parameter
difference. Both arms receive the same 342,896 target tokens and generation
budget. No external solver, verifier, answer router, or teacher runs at
inference.

## Development gate

Development contains all 1,289 frozen MTR1 identities. Holdout remains sealed.
RCR1 passes only if all conditions hold:

1. router treatment exceeds unchanged second pass by at least 5 percentage
   points;
2. router treatment exceeds the matched rank-1 attention control by at least
   3 percentage points;
3. correct counts do not regress on MATH, logic/science, or executable code;
4. both arms cover all identities and retain exact protected parameter rules.

A miss closes this exact router intervention. No seed, rank, layer, alpha,
duration, prompt, or threshold rescue is authorized. Only a pass may open the
existing holdout and authorize transfer to a larger MoE.

