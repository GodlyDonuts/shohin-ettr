# Shohin 72-Hour Product Reasoning Execution Queue

Status: active. Start: 2026-08-02 21:26 EDT. Owner: Codex.

## Live Scoreboard (2026-08-02 22:15 EDT)

| Result | Status |
|---|---:|
| frozen Qwen, GSM8K deterministic thinking, 100-row hash subset | 17/100 |
| frozen Qwen, MATH-500 deterministic thinking, same subset size | 4/100 |
| `B1` LoRA two-update H100 mechanics | pass; 0.902M trainable, 2.37 GB peak |
| `T1` recurrent workspace two-update H100 mechanics | pass; 6.690M trainable, 3.62 GB peak |
| `C1` dense control capacity match | implemented; workspace is 0.825% smaller than `T1` |

Transcript inspection shows the deterministic thinking score is strongly
affected by decode behavior: many GSM8K completions calculate the right value
inside a long plan, then exhaust the 768-token limit before producing the
requested final answer. Frozen no-thinking controls and official sampling are
therefore running. This is a decoding diagnosis, not permission to count an
unemitted answer as correct.

## Objective

Produce the first credible, directly measured ETTR-versus-baseline reasoning
delta on a capable pretrained backbone. The primary development backbone is
`Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`;
`HuggingFaceTB/SmolLM3-3B@a07cc9a04f16550a088caea529712d1d335b0ac1`
is the capacity fallback and confirmation arm.

This campaign optimizes answer quality. Synthetic packet accuracy, native
causal claims, immutable runtime ceremony, and architectural novelty are not
promotion criteria. They remain useful only when they diagnose a concrete
answer-quality failure.

## Product Scoreboard

Every compared arm uses the same backbone revision, examples, example order,
token budget, optimizer-update budget, decoding configuration, and evaluator.

| Board | Development iteration | Sealed milestone |
|---|---|---|
| grade-school math | held-out GSM8K training problems | full GSM8K test |
| competition math | held-out MATH training problems | MATH-500 and AIME 2024/2025 |
| code | held-out executable training tasks | LiveCodeBench plus standard code tests |
| science | held-out verified science questions | GPQA Diamond |
| logic | held-out generated and public logic tasks | frozen in-house logic board |

Primary metrics are pass@1 exact answer or test pass rate, per board and as an
unweighted five-domain macro average. Generation is greedy for the first gate.
Sampling and verifier-guided search are reported separately and cannot replace
pass@1. Every report records solved counts, prompt count, maximum generation
length, tokens/second, peak VRAM, wall time, and exact model/data revisions.

### First promotion gate

The integrated ETTR treatment advances when all conditions hold on the frozen
development board:

1. At least **+3.0 absolute macro percentage points** over the same-data LoRA
   baseline, or at least **+10% relative additional solved examples** when the
   baseline macro score is below 30%.
2. More solved examples in at least three of five domains.
3. No domain regresses by more than two absolute points.
4. The gain survives transcript inspection: final answers must follow from
   the generated computation rather than extraction quirks or malformed text.
5. A capacity-matched non-ETTR adapter must not clearly exceed the treatment.

The first result is evidence of a useful mechanism, not proof that ETTR alone
caused general reasoning. The full sealed board is opened only after the dev
gate passes.

## Compared Systems

| Arm | Purpose |
|---|---|
| `B0` frozen/pretrained | establishes native checkpoint capability and evaluator health |
| `B1` LoRA SFT | ordinary same-data post-training baseline |
| `T1` integrated ETTR + LoRA | practical recurrent-workspace treatment |
| `C1` dense adapter + LoRA | capacity/compute control without tied recurrent workspace |

`T1` is an end-to-end language model, not a composition of separately fitted
compiler, reactor, and reader checkpoints:

1. The backbone encodes the entire problem prompt.
2. Learned queries compress prompt hidden states into 16 workspace slots.
3. One tied gated recurrent cell updates those slots for 8 internal steps
   while cross-attending to the prompt. A learned STOP head is trained, but
   the first bounded gate also reports fixed-step behavior.
4. The final workspace becomes learned soft-prefix state before rationale and
   answer tokens.
5. Workspace, prompt projection, STOP/readout, and backbone LoRA parameters are
   optimized jointly from rationale and final-answer losses.

`C1` receives the same prompt features, parameter budget, update count, and a
matched number of dense transformations, but its transformations are untied.
If `C1` matches or beats `T1`, recurrence/ETTR has not earned inclusion even if
both beat `B1`.

## Data Contract

The immediate short run uses only audited, frozen sources already in Shohin.
They are seed material for a product experiment, not a sufficient final
post-training corpus.

| Domain | Immediate source | First token-mix target |
|---|---|---:|
| general instruction | audited broad instruction mix | 25% |
| verified math | reasoning-v2/RG and answer-verified concise OpenMath | 35% |
| executable code | unit-tested code rows | 20% |
| science | answer-checked science rows | 10% |
| logic/procedure | generated tasks with deterministic solvers | 10% |

Rows are balanced by charged target tokens, not raw row count. Training data
must be deduplicated before arm assignment and scanned against all development
and milestone prompts. Live teacher-writer files are never direct training
inputs. The existing roughly 25,000 frontier-teacher rows are seeds only.

In parallel, Stokes builds the larger post-training corpus. Admission requires
source license/attribution, exact-answer or execution verification where
possible, semantic and near-duplicate removal, benchmark decontamination,
length/format validation, domain accounting, and reproducible manifests. The
72-hour target is 0.5--2.0 million admitted examples; the subsequent target is
several million. RLVR starts only after a trained model generates a nontrivial
positive rate under deterministic reward checkers.

The first external seed is pinned OpenThoughts3. Its nominal 1.2M rows are 16
annotations over roughly 75,000 unique questions, so the acquisition job keeps
one deterministic best trace per normalized prompt rather than miscounting
annotations as independent data. It then applies math/code/science caps,
length and repeated-line checks, exact and 13-gram benchmark filtering, and
marks every surviving response `teacher_trace_unverified`. NVIDIA
OpenCodeReasoning-2 and OpenScienceReasoning-2 are audit candidates for the
next stage; code is admitted only after its judgement/pass-rate and executable
solution are verified, and science labels are answer-checked before use.

## Execution Queue

### Hours 0--6: make the backbone and measurements real

1. Finish already-running jobs `729556`, `729564`, `729565`, `729767`, and
   `729768`; record their terminal result but launch no successor in that
   mechanistic family.
2. Job `729773` downloads and loads the exact Qwen revision on an H100, emits a
   manual reasoning transcript, and records package support, parameter count,
   VRAM, throughput, and model class.
3. Freeze the development and milestone benchmark manifests. Run evaluator
   answer-extraction unit tests and a small known-answer fixture before model
   scoring.
4. Run `B0` on the compact development board and inspect at least 25
   transcripts across correct, near-miss, and malformed cases.
5. Audit the actual Newton paths, hashes, row counts, token counts, verifier
   flags, and domain balance of the immediate training sources.

Deliverable: a baseline scoreboard, transcript diagnosis, and admitted short-
run data manifest.

### Hours 6--18: build the end-to-end treatment

1. Implement one architecture-agnostic soft-workspace wrapper around the
   Qwen text path, plus checkpoint/save/resume and greedy generation.
2. Implement `B1` and `C1` through the same trainer and batch builder so the
   treatment is the only meaningful variable.
3. Run one 128-example overfit test and one held-out generation smoke for each
   arm. Required gates are finite loss, decreasing training loss, exact resume,
   no OOM, and coherent generated completion.
4. Profile batch size, sequence length, tokens/second, and peak H100 VRAM. Use
   independent one-H100 jobs for small arms; use data parallelism only after a
   measured throughput gain.

Deliverable: three trainable arms that generate answers from identical prompts.

### Hours 18--36: matched short training

1. Train `B1`, `T1`, and `C1` on the same frozen mixed stream for the same
   charged target-token and update budgets.
2. Reserve separate H100s when available so wall-clock queue differences do
   not become training differences.
3. Save periodic checkpoints and run a small non-sealed monitor board. Select
   checkpoints by development loss plus aggregate monitor score, never by one
   benchmark.
4. Run the frozen five-domain development board identically on all arms.

Deliverable: the first ETTR-versus-baseline product delta with solved counts,
throughput, VRAM, and transcripts.

### Hours 36--48: forced decision

| Result | Action |
|---|---|
| `T1` clears the promotion gate and beats `C1` | scale Qwen data/update budget and run sealed milestone |
| `T1` beats `B1` but not `C1` | retain the practical adapter gain; remove unsupported ETTR claim |
| all trained arms improve equally | improve data/post-training; architecture is not the current lever |
| `T1` fails while `B1` improves | redesign workspace-to-backbone injection once |
| all arms fail to learn | fix data formatting/trainer/evaluator before any model conclusion |
| corrected Qwen treatment remains flat | move the identical campaign to SmolLM3-3B |

No seeds, widths, or longer durations are launched merely to avoid a negative
decision.

### Hours 48--72: scale only the winning lever

1. Expand the winning Qwen arm or execute the SmolLM3 capacity fallback.
2. Grow the verified reasoning corpus on Stokes while Newton trains.
3. Open the sealed milestone board only for an arm that cleared the dev gate.
4. If supervised generation yields enough verified positives, launch a small
   RLVR pilot paired with an identical no-RLVR control.
5. Produce a final 72-hour scoreboard and a go/repair/retire decision.

## Compute Layout

Small-model jobs use one H100 each unless measurement proves distributed
scaling helps. Parallelism is across informative arms and evaluations:

| Concurrent use | H100s |
|---|---:|
| Qwen baseline evaluation | 1 |
| `B1` training | 1 |
| `T1` training | 1 |
| `C1` training | 1 |
| independent evaluation workers | 2 |
| SmolLM3 fallback/profile | 2--4 |

This can consume 6--10 H100s without paying small-model DDP overhead. Extra
H100s are used for data generation only when CPU verification is the measured
bottleneck. Stokes remains the default home for CPU curation and verification.

## Stop Conditions

- Do not launch a trillion-token scratch run from this campaign.
- Do not count best-of-N, host arithmetic, answer leakage, or benchmark
  contamination as native model improvement.
- Do not continue separate compiler/reactor/reader fitting.
- Do not repeat the closed layer-tap/byte-rail family after its current result.
- Do not promote an ETTR claim when an ordinary matched adapter performs as
  well or better.
- Preserve the protected Shohin 125M checkpoint as a baseline; it is not
  assumed to be the final-capacity product.

## 72-Hour Success Definition

Minimum success is a reproducible positive Qwen or SmolLM3 ETTR-vs-baseline
development delta with real solved examples and no evaluator artifact. Strong
success is a dev-gate pass plus improvement on the sealed public milestone.
If neither occurs, the campaign still ends with a hard capacity/interface
decision rather than another open-ended synthetic experiment.
