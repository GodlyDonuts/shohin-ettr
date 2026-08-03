# Shohin 72-Hour Product Reasoning Execution Queue

Status: active. Start: 2026-08-02 21:26 EDT. Owner: Codex.

## Live Scoreboard (2026-08-03 05:20 EDT)

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
| `C2` matched short train | complete; 200 updates; 429,658 charged target tokens; 410.9 charged tok/s; 19.68 GB peak |
| `B1` matched GSM8K | 32/100 after answer-extraction rescore |
| `B1` matched MATH-500 | 18/100 under Math-Verify 0.9.0; raw exact-string report was 15/100 |
| `T2` matched GSM8K, explicit EOS/turn stop | **44/100**, zero leaked role turns; +12 over B1, +3 over C2 |
| `T2` matched MATH-500, stopped-prefix semantic score | 30/100; B1 18, C2 29 |
| `T2` matched BBH logic, stopped-prefix score | 43/100; B1 30, C2 32 |
| `T2` matched GPQA, original decode | 34/198; B1 16/198; 105/198 T2 cap-exhausted and explanations remain unreliable |
| `T2` matched executable code | HumanEval 0/20; MBPP 0/20; broad promotion blocked |
| exact-turn GPQA B1/T2/C2 | 16/34/30 of 198; T2 +18 over B1 and +4 over C2 |
| frozen concise no-thinking GPQA | 3/198 |
| exact-turn raw MATH B1/T2/C2 | 15/24/27; provisional until common Math-Verify rescore |
| strict five-domain macro B1/T2/C2 | 18.31% / **26.33%** / 22.93% |
| strict solved examples B1/T2/C2 | 95 / **148** / 128 of 538 |
| corrected promotion decision | fail only on code regression: T2 0/40 versus B1 2/40 |
| balanced B1 training / GSM8K | 521,327 target tokens at 1,260.0 tok/s; 40/100 GSM8K |
| balanced T2/C2 training | both complete on 521,327 tokens; 450.0 / 451.6 tok/s; boards live |
| balanced strict B1/T2/C2 | **26.42% / 34.55% / 31.03% macro**; 131 / 185 / 162 solved of 538 |
| balanced promotion gate | **numeric pass**; T2 +8.13 points/+54 solved vs B1, +3.51/+23 vs C2; transcript gate mixed |
| balanced code B1/T2/C2 | 7/40 / 7/40 / 6/40; prior ETTR code regression repaired |
| strict balanced AIME-2024 B1/T2/C2 | 0/30 / **1/30** / 0/30; one genuine ETTR invariant solve, not yet robust |
| optimized B1 throughput | BS16/ACC1: 3,109.7 target tok/s, 60.24 GB peak |
| optimized T2/C2 throughput | BS8/ACC2: 1,090.5 / 1,095.8 target tok/s, 73.37 / 73.04 GB peak |
| verified-priority V10 data | 4,000,967 target tokens; 2,308 full-test code + 767 answer-checked math/science rows |
| V10 matched u1000 | training `730036/730037/730038`; strict aggregate `730057`; AIME `730058--730060` |
| balanced V8 u1000 B1/T2/C2 | 25.72% / 29.73% / 29.84% macro; 135 / 161 / 162 solved; ETTR loses dense by one |
| verified V10 u1000 B1/T2/C2 | 22.90% / 22.90% / 22.00% macro; 106 / 104 / 104 solved; ETTR math gain erased by logic/science collapse |
| SmolLM3 u1000 B1/T2/C2 | 41.72% / 40.91% / 43.21% macro; 203 / 194 / 199 solved |
| SmolLM3 strict AIME B1/T2/C2 | 0/30 / 2/30 / 1/30; one ETTR-only transcript is a valid complete derivation |
| current ETTR disposition | reject general scale; retain hard-math specialist evidence only |
| production scale candidates | SmolLM3 B1 LoRA and C2 dense residual on V11; no T2 continuation |
| verified-row full retention, 2048 vs 4096 | OpenMath 16.06% -> 53.25%; OpenScience 17.38% -> 49.10% |
| SmolLM3 B1 4096 throughput | BS4/ACC4 6,124.8 target tok/s at 12.54 GiB; selected over BS1/2/8/16 |

## Decisive U1000 Result

The first short Qwen result was real but transient. At 1,000 updates on the
balanced V8 stream, T2 retains a four-point gain over LoRA but ties the dense
control within one solved example and fails the frozen regression rule. On
verified-priority V10, T2 specializes sharply into competition math while
destroying logic and science. It therefore has not earned a Qwen scale run.

SmolLM3 establishes the higher practical capability floor. Same-data LoRA
reaches 93/100 GSM8K, 55/100 semantic MATH-500, and 203/538 solved overall.
Dense has the best five-domain macro because it raises executable code to
15/40. T2 reaches 58/100 MATH and 2/30 AIME, including one fully coherent
ETTR-only 204-minute rate-problem solution, but its general macro and solved
count both fall below LoRA. This is evidence of useful specialist computation,
not a general adapter win.

The immediate product decision is `scale-dense-or-lora`. V11 will compare B1
and C2 only. A future ETTR successor must be a prompt-conditioned expert that
defaults exactly to the general model outside its measured hard-math region;
it may not retrain a globally active workspace and repeat the observed
regression. The static per-domain expert ceiling from existing reports is
45.52% macro and 213/538 solved, which makes routing a concrete opportunity,
not a claim that the router has already been learned.

## Post-Gate 72-Hour Queue (2026-08-03)

The first objective has been answered: T2 produces a real short-run Qwen lift,
but it does not survive the longer matched gate and also loses the SmolLM3
general comparison. The queue now advances the strongest practical adapters
and treats ETTR as a measured hard-math specialist. No new layer-tap,
byte-rail, compiler/reactor/reader, or globally active ETTR duration/width
variant is part of this queue.

### Hours 0--8: close the two active causal questions

1. Finish V8 exposure-control jobs `729992/729993` and their exact six-board
   chains through strict aggregate `730013`. This decides whether 200 to 1,000
   updates improves T2 relative to both B1 and C2, rather than merely fitting
   the balanced stream harder.
2. Finish verified-priority V10 jobs `730037/730038`, exact six-board chains
   through `730057`, and strict AIME rescoring through `730063`. This decides
   whether independently checked code/math/science examples improve hard
   answers and rationale quality.
3. Inspect every new T2-only AIME/code win and a stratified sample of
   T2-only MATH/BBH/GPQA wins. A correct option with a false explanation is
   recorded as answer accuracy, not sophisticated reasoning.

Go rule: T2 must remain at least three macro points above B1, beat C2 by at
least two macro points or ten additional answers, improve at least three
domains, and avoid a domain regression greater than two points. If only C2
improves, recurrence has not earned the next run. If V8 improves but V10 does
not, exposure is the lever; if V10 improves at matched exposure, verified data
is the lever.

### Hours 8--24: build the next admitted training stream

1. Complete the 9,000-candidate TACO all-test audit and admit only programs
   that pass every supplied test.
2. Exact retention selects 4,096 tokens: it preserves 53.25% of OpenMath and
   49.10% of OpenScience rows versus 16.06% and 17.38% at 2,048. V11 job
   `761194` is frozen at 4,096; context length is increased for retained
   reasoning supervision, not unused VRAM.
3. Build V11 as a charged-token-balanced, no-replay stream with at least
   8 million targets: 35% math, 25% executable code, 15% answer-checked
   science, 10% solver-checked logic/procedure, and 15% broad instruction.
   Independently verified rows are consumed before teacher-only traces.
4. Freeze data hash, tokenizer revision, source counts, verifier counts,
   truncation counts, license inventory, and benchmark-overlap report before
   any V11 fit.

### Hours 24--48: scale one Qwen lever or measure the capacity floor

V10 T2 failed the go rule and the SmolLM3 comparison confirms that current T2
should not scale. Train matched B1 and C2 SmolLM3 arms for 5,000 updates on
V11, saving 1,000-update checkpoints. B1 uses measured BS4/ACC4 at 4,096;
C2 uses its independently measured safe geometry. Run the compact five-domain
board at each checkpoint and select one checkpoint by macro score plus
validation loss, never by one benchmark.

If V10 T2 fails the dense-control comparison, do not create another Qwen
workspace variant. Run the now-generic trainer on pinned SmolLM3-3B:

1. frozen baseline and manual transcript sample;
2. two-update generation smoke and 100-update bounded fit for B1/T2/C2;
3. one matched 1,000-update campaign only if T2 learns the fit board at least
   as well as C2 and generates valid autonomous answers.

The trainer records whether the backbone is loaded through the multimodal
Qwen text path or a standard causal-LM path. Data, update order, charged
targets, decoding, and evaluators remain identical across arms.

### Hours 48--72: public milestone and verified-reward decision

Open the full milestone only for the best development arm that passed the go
rule. Report full GSM8K, MATH-500, AIME 2024/2025, HumanEval, MBPP, a pinned
LiveCodeBench slice, GPQA-Diamond, and the frozen logic board with identical
decoding. Publish solved counts, pass rates, throughput, VRAM, wall time, and
at least 50 manually classified transcripts.

Start RLVR only if the selected SFT model produces at least a 5% deterministic
positive rate on an independently verified math/code pool. The first RLVR arm
must have an identical no-RLVR continuation control. Best-of-N or verifier
selection is reported separately and cannot replace greedy pass@1.

The 72-hour decision is one of: `scale-ettr`, `scale-dense-or-lora`,
`move-to-smollm3-capacity-floor`, or `repair-data/evaluator`. It is never
`repeat-a-similar-synthetic-architecture-arm`.

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
For MATH-500, the v3 lane uses isolated Hugging Face Math-Verify `0.9.0`.
B1 rises from raw exact-string `15/100` to `18/100` because three predictions
are mathematically equivalent to their references. The backend and version
are embedded in each rescored report.

The first treatment board exposed an adapter-generation defect before any
promotion decision. Qwen's tokenizer EOS token was not passed explicitly to
`generate`, so a completion could emit the correct `<|im_end|>` delimiter and
then continue into another synthetic chat turn. Those later turns were decoded
as literal `assistant`/`user` text and sometimes contradicted an earlier
correct result. A saved-prefix audit showed that removing the invalid suffix
raises T2 GSM8K from 42 to 44 and leaves MATH/BBH at 30/43. Exact H100 job
`729881` confirms T2 GSM8K `44/100` with zero leaked role turns. Runtime commit
`2ee1f7b` passes EOS explicitly and stops on exact turn delimiters. The fully
matched corrected campaign is `729918--729935`; every B1/T2/C2 board must use
the same recorded stop-token IDs before final aggregation.

Manual behavior remains part of the gate. Eleven GSM8K cases are solved by
T2 that neither control solves, with explicit multi-step arithmetic for rates,
inventory, proportions, and totals. Conversely, several T2-only GPQA answers
select the correct option while giving dimensionally or chemically incorrect
explanations. Current evidence is therefore a credible elementary-math and
logic foothold, not sophisticated general reasoning. The immediate corpus
repair is verified science rationale and executable code, where V8 is plainly
underrepresented.

The first full strict decision is now closed. V6 rejects implicit fallback
answers from cap-exhausted traces unless the model emitted a boxed or labelled
final answer. T2 reaches `26.33%` macro and `148/538` solved, versus B1
`18.31%` and `95/538`, and dense C2 `22.93%` and `128/538`. Relative to B1,
T2 adds 11 GSM8K, 12 semantic MATH-500, 12 BBH logic, and 20 GPQA answers. It
beats C2 by 3.41 macro points and 20 total
solved examples, so tied recurrence has earned a practical foothold rather
than merely matching extra dense capacity. The numeric gate remains false
only because code falls from B1 `2/40` to T2 `0/40`. Direct code transcripts
show that T2 frequently identifies the requested operation but answers in
prose or emits malformed Python. Balanced jobs `729936/729943/729950` began
immediately and automatic semantic decision `729983` follows their exact
boards. This is a targeted repair of one failed domain, not an attempt to
erase the positive four-domain result.

The strict aggregate does not clear the transcript gate unconditionally.
Many T2-only GSM8K and BBH solutions contain coherent multi-step arithmetic
or truth chains, but the manual sample also contains one correct arithmetic
result with the wrong unit and several correct final options preceded by a
contradictory explanation. GPQA rationales remain especially unreliable.
Accordingly, v6 is evidence of broad practical answer improvement, not yet a
claim of consistently sophisticated reasoning.

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

The immediate code-balance repair is already materialized rather than waiting
for the first campaign to fail. `pipeline/build_balanced_product_reasoning_mix.py`
selects the largest exact-weight subset possible without replaying or
duplicating any prompt. From frozen V8 it produces 36,250 rows:
`12,688 math / 7,250 code / 7,250 procedural / 9,062 teacher`, SHA-256
`aebf832278b8b0792cdde423b87f187808b918f5b6dc84631fde81e63a0b7fee`.
The artifact is hash-matched on Stokes and Newton. Code is now 20% rather than
1%, while every selected row remains unique. Balanced 200-update B1/T2/C2
jobs `729936/729943/729950` and their identical corrected board chains
`729937--729956` are dependency-held behind the current full campaign, so the
data-mix repair begins without another approval or scheduling gap.

This immediate artifact is balanced by rows, not by charged target tokens.
Exact Qwen tokenizer audit `729980` measures the effective target-token mix as
`47.14% math / 28.60% code / 7.33% procedural / 16.93% teacher` over
6,094,439 charged targets. It also records 94 response-truncated and 1,488
prompt-truncated rows. The arm comparison remains matched because every arm
sees the same stream, and the 28.6% effective code exposure makes this a
strong test of the data-exposure hypothesis. Future promoted mixes must be
balanced by charged tokens and reduce long-example prompt truncation.

OpenScienceReasoning-2 job `761160` completed with 500,000 unique
expected-answer-matched rows selected from 1,600,812 raw rows after quality,
duplicate, and benchmark-overlap filtering. The full output SHA-256 is
`e11e1923d237e1986725a7148503219e8871523649072cb38c835176854a5caa`.
A deterministic 10,000-row pilot has zero duplicate questions, zero replay,
and SHA-256
`eaca4020fc5dceab1cff41d5bae94e5308949773ee262a9153ee767deec89173`.
It is hash-matched on Stokes and Newton and stored with its report and
CC-BY-4.0 attribution in private dataset
`Godlydonuts/shohin-ettr-reasoning-data`. Expected-answer agreement is an
admission filter, not a guarantee that every intermediate explanation is
flawless. OpenMath `761161` is complete: 46,006 unique
expected-answer-matched rows survive from 3,201,061 raw rows, with output
SHA-256
`aeb373e8fb4fedc746527653e09e3d98e73d9749cd34e5dc628f9845de125e55`.
OpenThoughts consensus `761159` completed with 15,201 admitted rows
(`14,192` math and `1,009` science), SHA-256
`fb2af5eeaa0e2823355f7927997993b21ea35ea4adfdb13dfcc59e6112974346`.

The first tokenizer-exact 4M-token build failed closed for the correct reason:
the 10k OpenScience pilot provides only 155,899 nontruncated science response
tokens, below the 400,000 target. The 1,009 consensus-science rows are too long
to add any usable capacity at sequence length 1,024. A provenance-bound 50k
prefix from the already deterministic/hash-shuffled 500k corpus is now
materialized at SHA-256
`37a9fa96931ecfe719ceddc94c024ab298496f8a6e105ce4f78814a1c9e58937`.
Token mix job `761173` uses that pool; verified-priority successor `761175`
adds the code repair below.

The code bucket also required a provenance repair. All 7,250 V8 code rows lost
their verification metadata during mix construction. Their direct
CodeContests ancestor was bounded-tested on three cases, but that linkage is
not available in the training rows and the bucket includes completion
derivatives that cannot be scored as fully verified. A stronger TACO artifact
had already replayed every supplied test but was accidentally omitted from V8.
Commit `4910549` restores exact metadata only when candidate and verified
response bytes match. The resulting 2,936-row corpus covers 239,533 passed
tests and has SHA-256
`960ebf7dbcefa92bf44b71738e510bb259cb3fab681a817b6fb3bd749760c5da`.
Commit `048dd8e` then makes the token selector consume verified rows before
weaker fallbacks within a domain. Pinned Stokes jobs `761176 -> 761178` scale
this route toward 9,000 candidates and a 48-CPU all-tests audit.

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
