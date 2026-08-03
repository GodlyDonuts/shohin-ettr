# Shohin 72-Hour Product Reasoning Execution Queue

Status: active. Start: 2026-08-02 21:26 EDT. Owner: Codex.

## Live Scoreboard (2026-08-03 00:22 EDT)

| Result | Status |
|---|---:|
| frozen Qwen, GSM8K deterministic thinking, 100-row hash subset | 17/100 |
| frozen Qwen, MATH-500 deterministic thinking, same subset size | 4/100 |
| frozen Qwen, AIME-2024 | 0/30 |
| frozen Qwen, BBH logic hash subset | 13/100 |
| frozen Qwen, HumanEval executable subset, evaluator v2 / 1,024 tokens | 6/20; 30% |
| frozen Qwen, MBPP executable subset | 5/20; 25% |
| frozen Qwen, GPQA-Diamond thinking mode | 20/198; invalid as capability estimate because 198/198 cap-exhausted |
| `B1` exact 16-row/100-update fit | NLL 1.232 -> 0.555; -54.9%; 16/16 improved |
| `T1` soft-prefix exact fit | NLL 1.240 -> 0.796; -35.8%; 16/16 improved; reject unchanged scaling |
| `C1` dense-prefix exact fit | NLL 1.225 -> 0.695; -43.2%; 16/16 improved |
| `T2` gated residual mechanics | pass; 6.690M trainable; 181.1 charged tok/s |
| `C2` gated dense residual mechanics | pass; 6.642M trainable; 202.5 charged tok/s |
| `C2` exact 16-row/100-update fit | NLL 1.154 -> 0.457; -60.4%; 16/16 improved |
| `T2` exact 16-row/100-update fit | NLL 1.168 -> 0.455; -61.0%; 16/16 improved |
| `B1` matched short train | complete; 200 updates; 429,658 charged target tokens; 1,133.9 charged tok/s; 9.10 GB peak |
| `T2` matched short train | complete; 200 updates; 429,658 charged target tokens; 401.2 charged tok/s; 19.78 GB peak |
| `C2` matched short train | running as job `729856`; identical data/order/budget |
| `B1` matched GSM8K | 32/100 after answer-extraction rescore |

Transcript inspection shows the deterministic thinking score is strongly
affected by decode behavior: many GSM8K completions calculate the right value
inside a long plan, then exhaust the 768-token limit before producing the
requested final answer. Frozen no-thinking controls and official sampling are
therefore running. This is a decoding diagnosis, not permission to count an
unemitted answer as correct.

The original T1 interface is not advancing. It learned the bounded fit set,
but less efficiently than both controls and incurred severe score-time
latency. The one permitted redesign keeps the tied recurrent core while
reading frozen token embeddings and adding a near-zero gated residual to the
existing prompt positions. This removes the second full Qwen prompt pass and
the 16 random soft-prefix positions. C2 completed its exact gate and now leads
the bounded fit comparison at a 60.4% NLL reduction, ahead of LoRA's 54.9%.
The original T2 job `729837` encountered an `evc33` NVIDIA-driver lock before
writing a checkpoint and was canceled as hardware-invalid evidence. Its empty
output was preserved with an aborted-job suffix. Unchanged replacement
`729849` excludes `evc33`, with scorer `729850`; no duration or width variant
follows a negative result.

T2 therefore clears the mechanics/fit release by narrowly beating C2 and
materially beating B1. This is authorization for a held-out comparison, not a
reasoning claim. Matched 200-update jobs are B1 `729854`, T2 `729855`, and C2
`729856`; jobs `729858--729875` bind the same six development boards to their
resulting checkpoints.

Evaluator schema v2 now records per-example token usage and explicit
max-token exhaustion. The first HumanEval run exposed both real algorithm
errors and mid-function truncation, so code milestone scores are rerun at
1,024 generated tokens before promotion (`729851/729852`). Official
GPQA-Diamond has been
normalized from exact source commit
`56686c06f5e19865c153de0fdb11be3890014df7`: 198 rows, deterministic balanced
answer permutation, and two disclosed duplicate-distractor rows.

The first full GPQA thinking run is a decoding failure: every one of 198
responses exhausted 768 tokens, and its `20/198` exact score is largely driven
by accidental final option letters in unfinished text. No-thinking control
`729877` establishes the usable frozen reference. Evaluator-v2 HumanEval at
1,024 tokens rises from `3/20` to `6/20`; MBPP remains `5/20`, proving that
truncation explained some HumanEval misses but not the broader code deficit.

The optional fast-kernel runtime remains rejected for use. Its H100 parity
canary `729847` landed on the same unhealthy `evc33` node, ran more than five
minutes against the base runtime's 20.7 seconds, produced no report, and was
found beside a stuck `nvidia-smi` process. It was canceled as hardware-invalid
rather than interpreted as a kernel benchmark. All product jobs continue on
the known-good base environment unless a future isolated canary proves both
output parity and lower wall time.

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

`pipeline/aggregate_product_reasoning_campaign.py` enforces report completeness,
decode/data comparability, the five-domain numeric gates, and the dense-control
comparison. It deliberately leaves transcript coherence as a required manual
gate rather than inferring it from aggregate accuracy.
`pipeline/rescore_product_reasoning_report.py` replays answer extraction over
saved completions, so evaluator fixes never require or alter model generation.
The first use corrected one GSM8K false negative (`2.00` versus `2`) and one
truncation-driven false positive (`1 dollar 40 cents` parsed as `1`); B1's net
score remains `32/100`.

## Compared Systems

| Arm | Purpose |
|---|---|
| `B0` frozen/pretrained | establishes native checkpoint capability and evaluator health |
| `B1` LoRA SFT | ordinary same-data post-training baseline |
| `T2` integrated ETTR residual + LoRA | practical recurrent-workspace treatment |
| `C2` dense residual adapter + LoRA | capacity/compute control without tied recurrence |

`T2` is an end-to-end language model, not a composition of separately fitted
compiler, reactor, and reader checkpoints:

1. Frozen token embeddings expose the entire problem prompt to the workspace.
2. Learned queries compress those embeddings into 16 workspace slots.
3. One tied gated recurrent cell updates those slots for 8 internal steps
   while cross-attending to the prompt. A learned STOP head is trained, but
   the first bounded gate also reports fixed-step behavior.
4. The final workspace is projected back into existing late prompt positions
   through a near-zero initialized residual gate; it adds no sequence tokens
   and does not run the full backbone twice.
5. Workspace, prompt projection, STOP/readout, and backbone LoRA parameters are
   optimized jointly from rationale and final-answer losses.

`C2` receives the same prompt features, parameter budget, update count, and a
matched number of dense transformations, but its transformations are untied.
If `C2` matches or beats `T2`, recurrence/ETTR has not earned inclusion even if
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

Jobs `729807 -> 729848` completed this seed path. The first pass selected
64,938 unique prompts (`53,004` math, `6,243` science, `5,691` code); expanded
replay against 59,512 evaluation prompts removed 19 more 13-gram overlaps.
The frozen 64,919-row derivative has SHA-256
`d9daa5720f7d27ed9c49a24be5fecf3f9db80bdd9dc12d648f11907e12928d90`.
It remains seed material because upstream exposes no answer, test, or judge
field. A new two-pass consensus lane recovers quality signal from the sixteen
independent annotations: math/science rows require at least eight extracted
answers, eight votes for one exact-normalized answer, at least 60% agreement,
and a three-vote margin. Code is excluded from this lane until execution can
verify it. Stokes job `761159` runs from private commit `e55412f`; its runtime
manifest SHA-256 is
`f515310809570f251709e9b09f8aae2986e18db1cf643a120dd4e6f50059b8b7`.

Two independent expected-answer lanes run from private commit `12f3b0c`.
Stokes job `761160` reads pinned
`nvidia/OpenScienceReasoning-2@174b02c9cdf231f220765b2a1d5ece4550921894`
and admits only rows whose normalized generated final answer exactly matches
the provided expected answer. Job `761161` applies the same rule to pinned
`nvidia/OpenMathReasoning@d3d08664755704f422af97d43a7ff0ded4bd95df`,
restricted to `problem_type=has_answer_extracted`. Both replay the complete
local benchmark inventory before atomic output. A source label or upstream
teacher score alone cannot admit a row.

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

1. Train `B1`, `T2`, and `C2` on the same frozen mixed stream for the same
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
| `T2` clears the promotion gate and beats `C2` | scale Qwen data/update budget and run sealed milestone |
| `T2` beats `B1` but not `C2` | retain the practical adapter gain; remove unsupported ETTR claim |
| all trained arms improve equally | improve data/post-training; architecture is not the current lever |
| `T2` fails while `B1` improves | move to the backbone fallback; the one injection redesign is spent |
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
| `T2` training | 1 |
| `C2` training | 1 |
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
