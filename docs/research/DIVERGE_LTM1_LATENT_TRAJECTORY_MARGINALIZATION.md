# DIVERGE-LTM1: Latent Trajectory Marginalization

Status: frozen successor gate after JET1. No result yet.

## Decision

JET1 failed before generalization: several simultaneously unlearned hard
straight-through interfaces drove source grounding and algebra below chance.
LTM1 changes the optimization boundary rather than repairing JET1. It trains a
small factorized set of **complete latent reasoning trajectories** with an
exact smooth sequence-level marginal objective. It never averages candidate
state fields. Deployment commits to one complete trajectory before decoding.

LTM1 is a real-language development gate on the existing verified
math/code/science/logic stream. It is not another register-board experiment,
not a public product route, and not yet a DIVERGE architecture claim.

## Architectural thesis

Let `b` binary source-conditioned fault lines define `K = 2**b` sticky
assignments. For prompt `x`, each assignment `a` initializes one complete
latent state:

```text
z[a,0] = shared(x) + sum_j guarded_delta[j, a_j](x)
z[a,t+1] = T_theta(z[a,t], x)
```

`T_theta` is tied across recurrent steps and shared across assignments. The
same assignment remains attached to a lineage for the entire trajectory.
Every trajectory produces its own soft prefix and therefore its own complete
response likelihood. No opcode, pointer, token, or state field is selected
independently from another trajectory.

The training response is split deterministically into ordered contiguous
reasoning chunks. Frozen token embeddings give each chunk a semantic target.
Every latent step predicts the corresponding target, supplying dense smooth
credit before any discrete commitment.

For complete-trajectory response energy `E_a` and source-only prior `pi_a`:

```text
E_a = mean_token_NLL(y | x, z[a,T])
      + lambda_trace * ordered_trace_distance(z[a,1:T], y_chunks)

L_marginal = -logsumexp_a(log_softmax(pi)_a - E_a)
```

The log-sum-exp is over complete trajectories only. Hidden states are never
averaged. At inference, `argmax pi_a` selects one prefix before autoregressive
generation. No teacher response, answer label, verifier, host program, raw
source reread, or external model is present at inference.

This differs from:

- JET1: no straight-through source/program/state interfaces;
- ordinary recurrence: several sticky complete trajectories rather than one;
- soft particle aggregation: no fieldwise or hidden-state mean at inference;
- best-of-N decoding: one model-owned latent lineage is chosen before text is
  generated;
- ordinary MoE: fault lines create a product of guarded state patches and the
  same assignment persists through recurrent computation.

The possible contribution is the conjunction of factorized sticky latent
lineages, ordered trace-state supervision, and whole-sequence marginal credit.
Every ingredient has adjacent prior art; novelty is unclaimed until matched
transfer evidence exists.

## Frozen implementation

- Backbone: pinned `Qwen/Qwen3.5-0.8B` revision
  `2fc06364715b967f1860aea9cf38778875588b17`.
- Backbone training: rank-8/alpha-16 LoRA in the final four text layers; all
  non-LoRA tensors frozen.
- Fault lines: 2 binary variables, exactly 4 complete trajectories.
- Latent width: 384.
- Slots per trajectory: 8.
- Tied recurrent steps: 8.
- Attention heads: 8.
- Feed-forward multiplier: 2.
- Training-time trajectory mixture: exact log-sum-exp over all four complete
  candidates.
- Inference: source-prior MAP candidate, then one ordinary greedy decode.
- Trace targets: at most eight balanced contiguous response-token chunks;
  target vectors are detached means of the frozen input embeddings.
- Loss weights: trace `0.25`, batch posterior-balance `0.01`, and monotone
  halting `0.01`.
- Optimizer: fused AdamW, LR `2e-4`, betas `(0.9, 0.95)`, weight decay
  `0.01`, cosine decay, and gradient clipping at `1.0`.
- No MEI/MQB/QTG/JET/HSC weights are loaded.

## Frozen staged gate

### Stage 0: mechanics

CPU tests must establish:

1. all `2**b` assignments occur exactly once;
2. assignment identity is sticky across all recurrent steps;
3. prefixes and trajectory probes have stable geometry;
4. gradients reach shared, guarded, recurrent, prior, and output parameters;
5. sequence marginalization is finite and permutation-consistent;
6. inference selects one exact stored lineage, never a mean;
7. reset and lineage-swap controls change only the declared prefix;
8. checkpoint metadata reconstructs exact geometry.

Then run one two-update H100 smoke. Infrastructure-only faults may be repaired
without changing the scientific contract.

### Stage 1: bounded real-language fit

- Data: existing hash-audited V10 verified-priority stream
  `v10_tokenbalanced_35m20c10s10p25t_4m_verified_r1.jsonl`.
- Identical deterministic reservoir and row order for treatment and baseline.
- One seed: `2026080601`.
- 100 updates, 16 selected rows, context 1,024.
- One LTM1 arm and one exact LoRA-only B1 control.
- Both arms receive the same 1,016 logical prompt/response token budget. LTM1
  appends eight latent slots to reach 1,024 backbone positions; B1 receives no
  hidden prefix. Final NLL is compared with identical per-token weighting.

LTM1 qualifies for broad training only if:

- final token-weighted response NLL is no worse than B1;
- all 16 examples improve from update zero under teacher forcing;
- selected-trajectory trace cosine similarity is at least 0.90;
- at least two of four trajectory IDs are selected across the 16 prompts;
- every gradient and tensor remains finite;
- non-LoRA backbone tensors remain unchanged.

Failure closes LTM1 without a new seed, width, bit count, recurrent depth,
trace weight, loss, schedule, layer count, or longer fit.

### Stage 2: matched broad development

Only after Stage 1 passes:

- train one LTM1 and one B1 arm for 200 updates on the same V10 stream;
- match selected rows, row order, target tokens, context, optimizer, LR,
  LoRA geometry, and update count;
- charge LTM1's additional candidate FLOPs and wall time explicitly;
- evaluate both identically on the frozen 538-example development board:
  GSM8K 100, MATH-500 100, HumanEval 20, MBPP 20, GPQA 198, and BBH logic
  100; code enters the five-domain macro as the HumanEval/MBPP mean.

Promotion requires all of:

1. at least +3.0 absolute five-domain macro over matched B1;
2. at least 15 additional solved examples;
3. improvement in at least three of five domains;
4. no domain regression greater than two points;
5. resetting the selected latent prefix loses at least three macro points;
6. forcing the lowest-prior complete lineage loses at least two macro points;
7. non-LoRA backbone hashes remain unchanged.

If treatment qualifies, run one parameter/FLOP-matched dense recurrent control
and one unopened fresh verified board. If treatment fails, close LTM1 and do
not reinterpret train fit, posterior entropy, or trajectory geometry as
reasoning.

## Stop rule

LTM1 is one bounded test of whether smooth complete-trajectory credit and
trace-aligned recurrent state solve the optimization boundary exposed by
JET1. It is not authorization for long continuation pretraining, public score
routing, more candidates, annealed hard selection, alternate chunking, or a
nearby schedule/loss variant. A negative result requires a different state or
learning substrate.
