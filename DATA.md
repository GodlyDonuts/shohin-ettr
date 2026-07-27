# Data, teacher, and the reasoning-distillation recipe

> ⤴ **Superseded by [MASTER_PLAN.md](MASTER_PLAN.md) §4 (tokenizer) + §6 (data plan)** (the plan of record).
> Kept as background; the master plan adds the Reasoning-Gym procedural corpus, concrete token counts, and the
> verifier.

## Current ETTR architecture initializer (2026-07-27)

The active data build is `R12-ETTR-IL-v3-initializer`, a mechanism-specific
initializer for the completed ETTR architecture. It is not ordinary language
pretraining and is not claimed to be a complete general-reasoning corpus.

The frozen selected target is **62,500 semantic cores**:

| split | semantic cores |
|---|---:|
| train | 40,000 |
| development | 5,000 |
| sealed confirmation | 5,000 |
| train reserve | 10,000 |
| development reserve | 1,250 |
| sealed confirmation reserve | 1,250 |

Each core is independently reconstructed and replayed as an exact 2x2
WORLD-by-COMMAND causal rectangle, rendered in four token-native views, and
expanded to 64 fixed-geometry training rows. The train split therefore
contains **2.56 million expanded rows** and **1,351,680,000 charged token
positions**. Horn, guarded-resource, and bounded local-rewrite ontologies are
balanced across compiler grounding, atomic transitions, query
counterfactuals, dependent composition, and closed-loop invariance curricula.

Quality is enforced as a sequence of fail-closed gates:

1. deterministic generation from a commit-pinned source freeze;
2. split ownership from the semantic WORLD before rendering or labels;
3. exact primary/replay oracle agreement;
4. receiver-backed qualification through the real architecture materializer;
5. deterministic coverage-weighted selection with global semantic dedup;
6. four-view token-width and tensor-geometry admission;
7. global leakage, replay, uniqueness, and main/confirmation separation audits;
8. private Hugging Face publication followed by a fresh remote hash round trip.

Raw generation is complete at **324/324 candidate cells**. The first direct
selection attempt exposed an important quality gap: some generator-valid
Horn/resource rows were prefix-independent or contained a generic operation
with no state effect. The receiver correctly rejected them. The pipeline now
qualifies every overproduced raw candidate with that exact receiver before
selection; admission is never weakened and insufficient post-qualification
quota fails closed. Main and sealed-confirmation qualification use physically
separate roots, and only the main materialized corpus may enter the private
Hugging Face dataset repository.

## Pretrain mix — reasoning-tilted (~100–200B tok, WSD)

A 130M model has no tokens to waste on trivia. The pretrain buys a **language floor + a reasoning substrate**,
tilted far harder toward math/code than a generalist would.

**Stable phase:** language floor from filtered web (DCLM-baseline / FineWeb-Edu / Nemotron-CC-HQ) blended with
a **heavy math+code diet from the start** (OpenMathReasoning problems, FineMath4+ / MegaMath, Stack-Edu). Target
mix ≈ **~45–55% web / ~25–30% code / ~20–25% math** — math and code weighted far above a generalist's ~15/7.

**Decay phase:** shift almost entirely to quality reasoning — math/code/logic + short-CoT traces + a slice of
instruction data. Linear LR decay to ~0.

## Tokenizer — compact, reasoning-tuned (a decisive early call)

**Decision: a compact ~32k BPE — English + code + math, with single-digit number tokenization — NOT the
teacher's 128–151k vocab.** *(Judgment call; veto if you disagree — it changes the arch.)*

- **Why:** our decisive lever is *trace* distillation (SFT on the teacher's generated **text**), which does
  **not** require a shared tokenizer — so we're free to drop the giant vocab. At d_model ≈ 576, a 151k tied
  embedding ≈ **87M params (~2/3 of a 130M budget)**; a 32k embedding ≈ **18M**. Every param not spent on an
  embedding table is spent on **reasoning depth**. This is a concrete edge over MobileLLM-R1, which carried the
  128k Llama vocab and paid for it in layers.
- **Single-digit numbers** materially help arithmetic (a known math-reasoning trick).
- **Trade-offs we accept:** (a) no vocab-level logit-KD — fine, logit-KD is demoted; (b) no same-tokenizer
  reference model; (c) slightly longer sequences on number-heavy text — worth it for arithmetic.

## The teacher — current-best AND close (2026 refresh)

Llama-405B (via OpenMathInstruct-2, 2024) is retired as our ceiling — it's dated *and* a 0.03% far-teacher.
The 2026 sweet spot is a **current strong ~3–9B reasoner generating SHORT, verified traces**: newer *and*
~50–200× closer to the student than 405B. Because we **trace-distill** (SFT on the teacher's text, re-tokenized
with our own 32k vocab), the **teacher's vocabulary is irrelevant** — the giant Gemma-4 (262k) / Qwen3.5 (248k)
vocabs are a non-issue for us.

| teacher | params | license | short-trace fit |
|---|---|---|---|
| **Nemotron Nano 2 9B** | 9B | NVIDIA Open (permits distillation) | **natively short** via a hard *thinking-budget* cap → correct short traces without discarding 95% of a long-CoT set |
| **Ministral 3 8B Reasoning** | ~8B | Apache 2.0 | top small-model reasoning scores, fully permissive |
| **Gemma 4 E4B (thinking)** | 4.5B | **Apache 2.0** | current, close to student scale, native `<think>` toggle |
| **Qwen3.5-4B (thinking)** | 4B | Apache 2.0 | current best <5B; long-by-default → cap generation |
| **SmolLM3-3B** | 3B | Apache 2.0 | closest scale to a tiny student; fully open |

**Retire Llama; don't pick blind — run a cheap 3-way bake-off** (it *is* cheap at 135M): (A) small/close
(Qwen3.5-4B or SmolLM3-3B), (B) strong/short (Nemotron Nano 2 9B or Ministral 3), (C) Mix A+B ≈1:4
(Mix-Distillation, 2502.12143). The teacher-size question is genuinely **open at 135M** — the two most relevant
2026 papers disagree (2502.12143: smaller for tiny students; 2604.08880: stronger is fine) — but our
**≤256-token filter neutralizes the main gap mechanism** (long-CoT stylistic drift), so a strong 4–9B
short-filtered teacher is a real contender. The A/B decides it, not argument. *(Teacher inference is GPU-gated.)*

**Regenerate short traces — do NOT filter long ones down.** Evidence (2606.21704): natively-short training beats
compressing long traces, and it sidesteps most teacher-license traps. **Math is the riskiest domain to shorten**
(it needs precise intermediate values) — keep math traces on the long side of the budget and verify hard.

## The decisive lever — short-CoT reasoning distillation + rejection sampling

1. **Generate** verifiable reasoning traces with the teacher over math/code/logic problem banks (NuminaMath,
   GSM8K/MATH train, code tasks, logic templates).
2. **Rejection-sample: keep only correct traces** (answer-checked / unit-tested). Cheap, huge quality lever;
   universal in MobileLLM-R1 / Phi-4-mini / OpenMathReasoning.
3. **Keep them SHORT:** enforce a tight token budget and **compress** long R1-style traces (Compress-Distill
   style) so a 130M model can actually represent them. **Measure trace token lengths locally — don't assume.**
4. **Mix short:long ≈ 4:1** ("Mix Distillation", 2502.12143) — short-heavy beats either alone.
5. **Curriculum:** easy→hard, with gradually growing trace length (MobileLLM-R1's proven move).
6. **SFT** the student on this corpus with standard next-token loss (no logit matching needed).

## RL (RLVR / GRPO) — optional polish, not the engine

Run *only* as an honest A/B on top of a strong SFT'd base, on verifiable rewards (answer-checkable math/code).
**Expect little or no gain at 130M** — Meta: SFT 74.0 vs RL 57.0 at 950M; verified RLVR floor ~0.5B; reward
sparsity worsens as params shrink (a 130M base rarely samples a correct trace to reward). If flat, drop it —
the story is distillation, not RL.

## Distillation-dataset vetting — the 10-second rubric

1. **Real?** verifiable teacher, reputable uploader (not "god-level seed" / "mythos" / synthetic).
2. **Verifiable domain?** math/code/logic with checkable answers — **not** agentic tool-use or roleplay.
3. **Short enough / compressible?** the make-or-break filter at 130M.
4. **Correct?** rejection-sample; never train on unverified traces.
5. **Clean license?** Apache/MIT/CC-BY and ToS-compatible (we ship weights).

Ready-made trace sets (OpenR1-Math-220k, OpenThoughts-114k, OpenMathReasoning, OpenCodeReasoning-2) are all
**long-CoT** — usable only **compressed + rejection-filtered**. Expect to **generate our own short-CoT traces**
as the primary source; published avg-trace-lengths are unreliable, so measure before feeding.

## Bottom line

The win is **short, correct, verifiable reasoning traces from a close ~1–2B teacher**, on a reasoning-tilted
base with a compact vocab. Data quality (short-CoT + rejection sampling) is the **dominant** lever; RL is a
maybe; logit-KD and long-CoT are out.
