# DREM1: Draft-Conditioned Recurrent Expert Modulation

Status: preserved unmatched upper-bound diagnostic, superseded by ECR1 on
2026-08-09. No DREM1 capability training or holdout access is authorized.

Independent review found decisive confounds: the controller is static across
generated tokens, draft masking only zeroed pooled draft state, the controller
requires an extra frozen-backbone pass, its approximately 6.84M trainables are
unmatched to MTR1, and the 1,024-token default risks truncation. Pending job
`747085` was canceled before allocation. The identifying successor is
`SHOHIN_ECR1_EXPERT_CONDITIONED_REVISION.md`.

## Evidence boundary

The host remains pinned
`allenai/OLMoE-1B-7B-0125-Instruct@b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`.
The exact 9,655-row revision training set, 1,289-row source-disjoint
development board, model-owned drafts, prompts, targets, decoding, and strict
assessor remain those used by MTR1 and RCR1. Holdout remains sealed.

Completed controls establish:

- unchanged second pass: `191/1,289`;
- MTR1 rank-8 shared-attention revision: `204/1,289`;
- RCR1 direct router residual: `194/1,289`;
- matched RCR1 rank-1 shared-attention control: `191/1,289`.

MTR1 changed mean route counts by only `0.002018` L1 over all layers. Read-only
transition attribution found 16 strict repairs and three breaks. Conservative
semantic audit certifies at least 11 of the 16 repairs as serialization-only,
leaving at most five possible semantic repairs against three semantic breaks.
That report is not a rescore; it bounds how much of MTR1's apparent gain can
represent new reasoning. Its SHA-256 is
`4cea4a85e02c524db69e267effa242620a4237c3213ee5fe0c8d30e94f9543f1`.

Per-row route attribution job `747082` is read-only and must close before the
one-update DREM1 mechanics job opens. Its result may refine interpretation but
may not alter the architecture, arms, or thresholds below.

## Capability hypothesis

Static shared-attention adaptation mostly learned answer serialization, while
token-local router perturbation lacked a persistent diagnosis and could only
select unchanged experts. A useful MoE reviser needs one source-and-exact-draft
state that persists across sparse layers and controls both expert choice and a
small revision-specific computation inside the selected expert lineage.

## Architecture

Let frozen prompt features be `H`, exact draft mask be `m_d`, and ordinary
source mask be `m_s`. DREM1 computes disjoint pooled features

```text
s = mean(H[m_s])
d = mean(H[m_d])
c0 = LN(SiLU(Wc [s; d; d-s; d*s]))
c(l,j+1) = GRU(c(l,0) + e_l, c(l,j))
```

using one tied recurrent cell for four steps at each of the final four sparse
layers. The resulting `c_l` is fixed for the complete second-pass trajectory.
For token state `h`, frozen router logits `r`, rank `q=8`, and expert `e`:

```text
r' = r + tanh(Wro((Wrh h) * tanh(Wrc c_l))) / q
I,W = top8(softmax(r'))
u = (Weh h) * sigmoid(Wec c_l)
expert_residual = sum_(e in I) W_e * U_e u
```

The frozen expert-bank result and the selected-expert residual are added. No
pretrained router or expert tensor is trainable. Incompatible expert outputs
are never fieldwise averaged; the actual top-8 lineage determines the adapter
mixture.

The full arm has approximately `6.84M` trainable parameters: approximately
`2.49M` in the tied controller and `4.34M` across the four sparse blocks. It
uses a second frozen prompt-analysis pass, so latency and total FLOPs must be
reported rather than hidden behind active-parameter counts. It begins at exact
base behavior because router-output and expert-up projections are zero
initialized.

## Frozen arms

All learned arms use 256 AdamW updates, the same 342,896-target-token geometry
as MTR1/RCR1, seed `2026080901`, batch one, accumulation eight, maximum sequence
length 1,024, and learning rate `2e-5`.

1. **DREM1 full:** recurrent state controls router and selected-expert adapter.
2. **Router-only recurrence:** identical controller; expert adapter disabled
   and frozen.
3. **Expert-only treatment:** identical controller and selected-expert adapter;
   router residual disabled and frozen.
4. **Draft-masked full:** identical full parameterization, but exact draft
   pooled state is zeroed before recurrent control.
5. **Equal-parameter shared attention:** ordinary final-four-layer attention
   LoRA sized as closely as possible to full DREM1 trainables, with no sparse
   intervention. Its exact rank is fixed from the mechanics parameter receipt
   before any capability result.

Unchanged, MTR1, RCR1, generic self-refinement, long generation, best-of-two,
and independent commitment remain completed references and are not rerun.

## Safety and accounting

- Base router and expert weights must be bit-identical and absent from every
  trainable checkpoint.
- Report total/trainable/active parameters, charged target tokens, wall time,
  generated tokens, peak memory, estimated inference FLOPs, and latency.
- Report per controlled layer: route-probability L1, top-1 route-change rate,
  all-expert counts, selected-expert entropy, and probability entropy.
- A normalized route entropy below `0.80`, fewer than 48 used experts, NaN,
  OOM, missing draft span, or checkpoint mismatch fails closed.
- Holdout, answer labels, external models, verifiers, and benchmark-specific
  routing are unavailable to the model and optimizer.

## Development promotion gate

Promotion is conjunctive. Full DREM1 must:

1. score at least `256/1,289`, a gain of at least 65 answers (`+5.04` points)
   over unchanged and 52 answers (`+4.03` points) over MTR1;
2. score at least the unchanged domain counts: MATH `40`, logic/science `145`,
   and MBPP `5`;
3. exceed every newly trained matched control by at least 26 answers
   (`+2.02` points);
4. produce at least 25 repairs not certified as serialization-only, with
   possible-semantic repairs minus semantic breaks at least 20;
5. beat draft-masked full by at least 13 answers (`+1.01` points);
6. change at least 1% but no more than 35% of top-1 controlled-layer routes,
   retain normalized route entropy at least `0.80`, and use at least 48 of 64
   experts.

Only a complete development pass authorizes one sealed holdout evaluation of
the full arm and unchanged control. Holdout requires at least +5 points,
nonnegative deltas in every domain, and the same semantic-repair and routing
conditions. Failure closes exact DREM1 without rank, width, duration, seed,
layer, loss-weight, parser, or threshold rescue.

## Implementation receipt

Core implementation is `train/drem1_moe_revision.py`; training entry point is
`train/train_drem1_product.py`; autonomous loading and draft-mask alignment are
integrated into `train/hf_product_reasoning_eval.py`. Focused controller,
base-parity, intervention, ablation, route-receipt, generation-mask, and
evaluator tests pass `32/32` before a real-model mechanics allocation.
