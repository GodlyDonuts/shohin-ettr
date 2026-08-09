# Shohin RCR1 Revision-Conditioned Routing Gate

Status: **closed negative on development**, 2026-08-09. Holdout remains
sealed. A static final-four-layer router residual is not sufficient.

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

## Frozen result

| Arm | Correct / 1,289 | Accuracy |
|---|---:|---:|
| revision-conditioned router residual | `194` | `15.0504%` |
| matched rank-1 shared-attention control | `191` | `14.8177%` |
| unchanged second pass | `191` | `14.8177%` |
| prior rank-8 shared-attention MTR1 | `204` | `15.8262%` |

The router treatment adds only three answers / `0.2327` points over both the
matched control and unchanged pass, and remains ten answers below MTR1. Broad
domain deltas versus unchanged are math `+2`, logic/science `+1`, and code
`0`; retention passes while both magnitude gates fail.

RCR1 rules out this narrow mechanism: a small, token-local residual on late
router logits cannot by itself create a useful revision operator when all
experts remain frozen. It does not distinguish whether the missing factor is
persistent draft-level state, earlier routing, expert-side revision capacity,
or the active-capacity boundary. That attribution is the next read-only step.

Comparison SHA-256 is
`eb1b304adbf634e141bd497c5cf4b3d60d8616655ec1a674b7c556e399ae4c35`.
No full larger-MoE campaign is authorized from this result.
