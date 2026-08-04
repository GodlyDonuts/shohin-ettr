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
| admitted Smol V11i | 8,005,985 targets; 22,828 unique rows; 42m/25c/23s/4p/6t; SHA-256 `597293b6...c2cc423` |
| live V11i B1 scale | job `730161`; 5,000 updates; BS4/ACC4; ~5.38k target tok/s on real mixed lengths |
| live V11i C2 scale | job `730177`; 5,000 updates; BS2/ACC8; ~3.14k target tok/s; original BS4 job OOMed before update 1 |
| V11i matched public boards | B1 `730163--730169`; C2 `730178--730184`; seven boards per arm |
| V11i 1k/3k/5k B1 macro | 34.42% / **40.03%** / 37.53%; 178 / **207** / 197 solved |
| V11i 1k/3k/5k C2 macro | 37.12% / **42.44%** / 40.84%; 186 / **226** / 212 solved |
| selected V11i checkpoint | 3,000 updates; 16,783,669 charged targets; about 2.10 corpus passes |
| expanded full-board B1/C2 macro | 39.960% / **42.104%** over 3,930 tasks; 1,989 / 2,007 solved; includes development subsets |
| expanded-board C2 gains | GSM8K +58, MATH +41, GPQA +14 answers |
| expanded-board C2 regressions | executable code -19, BBH logic -76, AIME -1 |
| non-overlapping remainder B1/C2 | 46.168% / 47.149% four-domain macro; 1,782 / 1,781 solved of 3,392 |
| promotion decision | C2 is best single arm but **fails broad gate**; unopened solved count does not improve |
| static domain-routing ceiling | expanded 43.894% / 2,102 solved; unopened remainder 49.586% / 1,877 solved; neither is a model result |
| direct manual composition B1/C2 | 5/12 / 8/12; both still fail four combinatorial/state-composition prompts |
| source-label learned router | 92.807% source-held validation, but expanded 40.280% / 1,997 solved and unopened remainder 46.668% / 1,790 solved; not promoted |
| dev-calibrated global threshold | dev 44.240%; unopened remainder 47.013% / 1,771 solved; overfits and loses both frozen arms in solved count |
| router disposition | **closed**; no threshold/feature variant; future gating needs disjoint paired-outcome supervision or joint training |
| fresh paired math B1/C2 | 6/100 / 8/100; oracle 10/100 |
| fresh paired science B1/C2 | 26/100 / 7/100; oracle 28/100 |
| paired outcome aggregate | B1 32/200, C2 15/200, oracle 38/200; 23 B1-only / 6 C2-only; **close routing** |
| pinned TACO verifier | Stokes `761240`; 9,000 exact candidates; local source SHA-256 `d0593d49...92287e` |
| V12 data candidate | 16M unique targets; 46m/18c/33s/1p/2t; job `761253` after TACO |
| held V12 model gate | standard B1 `731325` vs 8x-wider LoRA `731326`; release only after V12 audit |

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

### Learned Prompt-Gate Result

The expert complementarity did not translate into a successful learned
product router. The preregistered prompt-only classifier uses 50,000 hashed
word and bigram features and 17,976 V11 prompts labelled only by ordinary
training source: math/science/arithmetic procedure selects C2, code and
non-arithmetic procedure selects B1, and teacher rows are excluded. It sees no
benchmark outcomes. Although source-held validation is 92.807%, the learned
score distribution sends almost every public prompt to B1. It reaches 40.280%
macro and 1,997/3,930 solved on the expanded board, and 46.668% macro with
1,790/3,392 solved on the previously unopened four-domain remainder. That is
only eight more remainder answers than B1 and remains far below the static
ceiling.

One bounded calibration repair enumerated a single global score threshold on
the already-used development reports, maximizing five-domain development
macro subject to no domain falling more than two points below B1. It selected
threshold `-133.58884639320723` and reached 44.240% development macro, but the
apparent gain did not generalize. The calibrated router reaches 41.907% and
2,003/3,930 solved on the expanded board, then 47.013% and 1,771/3,392 solved
on the unopened remainder. It routes nearly all math, MBPP, and BBH prompts to
C2 and loses both frozen arms in broad solved count. Manual composition is
5/12 before calibration and 8/12 after calibration because every manual
prompt is routed to C2.

This router family is closed. Source identity is not a sufficient proxy for
per-prompt expert advantage, and a global threshold calibrated on a small
development board overfits. No further threshold or feature variants are
authorized. A future gate must be trained on a large disjoint corpus with
paired frozen-expert outcomes, or learned jointly with the experts, and must
beat the best single arm on an untouched board. Until then the deployable
result is C2 as the strongest single arm, not the oracle ceiling.

### Paired Outcome-Supervision Closure

The one allowed stronger routing test also fails. Two new boards were built
from answer-verified OpenMath and OpenScience rows after exact exclusion of
every V11 training question. The first 100 rows of each were evaluated by the
frozen selected 3k B1 and C2 adapters under the same model revision, decoding,
and evaluator. Math is B1/C2 `6/100 / 8/100`, with only two B1-only and four
C2-only wins. Fresh science is `26/100 / 7/100`, with 21 B1-only and two
C2-only wins. Across 200 prompts, a perfect per-prompt selector raises B1 only
from `32/200` to `38/200`; C2 contributes six exclusive answers and loses 23.

This misses the outcome gate fixed before reading results: oracle lift at
least five percentage points and at least 5% exclusive wins for each arm. The
remaining 1,900 rows per board will not be spent on inference, and no
outcome-label classifier will be trained. The result changes the practical
interpretation of C2: it is a narrow in-distribution math specialist whose
fresh science regression is severe, not a broadly complementary expert.

The next product gate is data plus adapter capacity, not routing. After the
9,000-program TACO audit, V12 attempts a unique 16M-target stream at
`46m/18c/33s/1p/2t`. Matched one-pass 3,000-update jobs compare the proven
1.679M-parameter LoRA with an approximately eight-times-wider LoRA. Both are
held until the V12 capacity/hash/truncation report passes. Identical public and
fresh boards decide whether verified-data breadth plus adapter capacity raises
real solved counts without C2's code/science/logic tradeoff.

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
   49.10% of OpenScience rows versus 16.06% and 17.38% at 2,048. Context is
   increased for retained reasoning supervision, not unused VRAM.
3. The Smol-tokenized capacity audit rejects the draft mix: procedural has
   only 352,609 fully retained targets and teacher seed data only 807,714.
   The admitted no-replay stream has 8,005,985 targets at 42% math, 25%
   executable code, 23% answer-checked science, 4% procedural, and 6% teacher.
   Independently verified rows are consumed before teacher-only traces.
4. Freeze data hash, tokenizer revision, source counts, verifier counts,
   truncation counts, license inventory, and benchmark-overlap report before
   any V11 fit.

### Hours 24--48: scale one Qwen lever or measure the capacity floor

V10 T2 failed the go rule and the SmolLM3 comparison confirms that current T2
should not scale. Matched B1 and C2 5,000-update jobs are now live on V11i.
B1 uses BS4/ACC4 at 4,096. C2's short BS4 canary did not cover the longest V11
batch and OOMed before update 1; BS2/ACC8 preserves the exact 16-example update
and is the real-stream safe geometry. Seven identical public boards per arm
are already dependency-bound to the final checkpoints.

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

The original audit `761178` timed out after six hours with zero checked rows:
its implementation first scanned the entire streaming source and only then
submitted matches to the 48 verification workers. This is not a data-quality
failure. Commit `c6febaa` changes the audit to overlap deterministic
source-order verification with discovery and fsync resumable output every 100
checked candidates. Replacement `761230` runs the identical 9,000-candidate
input, pinned source revision, and all supplied tests with a 12-hour ceiling;
final mix `761231` is held after successful completion.

### V12 scale gate and V13 contingency (2026-08-03)

The exact-code bottleneck is closed. An 18-way, 576-CPU TACO replay admits
8,919 of 9,000 candidate programs; the merged output SHA-256 is
`4fe6e92737ef190423aff8e53246f77f213bb1fe0651c548a008b106fb04bfb4`.
The resulting V12 corpus has 25,139 unique rows and 16,003,044 charged target
tokens at `46% math / 18% code / 33% science / 1% procedural / 2% teacher`.
There are zero selected prompt truncations, response truncations, or duplicate
questions. Its SHA-256 is
`98527177e6e2abad364112659aaf11c71313babfde7f7031c635cdd1dc9ce5ab`,
matched on Stokes, transfer, and Newton.

Three one-H100 V12 arms are released: B1 `731325`, wide LoRA `731326`, and
late-two-layer release `731919`. Twenty-four independent fixed-board workers
are dependency-staged. This maximizes useful cluster occupancy without paying
small-model multi-GPU synchronization overhead.

V11 wide LoRA closes at a 39.2% five-domain development macro after 3,000
updates. It improves selected math/code cases but does not resolve the
science/logic tradeoff. The strongest direct-interaction result is late-layer
update 500 at 10/12 hand-written compositions under a 1,536-token generation
budget, versus 7/12 at 768 tokens. Truncation is a real evaluation/deployment
constraint, but harder public domains remain weak even at 1,536 tokens.

V13 is the predeclared repair if V12's broad verified mix still loses
composition. It substitutes 13% execution-verified procedural traces and uses
`42/16/27/13/2` math/code/science/procedural/teacher weights. Only B1 and the
manual-transcript-leading late-layer arm are staged (`732353/732354`), with
sixteen independent benchmark workers. It will release only after exact
hash/composition/truncation gates pass.

One deployment-only diagnostic is permitted alongside training. For a trace
that exhausts its budget without an explicit final marker, the same frozen
model receives the original problem and its own draft once, then has 64 greedy
tokens to emit only a boxed final answer. This performs no search or host
calculation and leaves the historical evaluator unchanged by default. It is
measured separately because it repairs answer emission rather than underlying
reasoning.

V13 subsequently cleared the gate with 62,874 rows, 16,003,742 charged target
tokens, zero selected truncation/duplicates, and SHA-256
`7df4f35d15d925b3f1a039f7cd877b1a887a942dd050b70abe5d500dc1f05621`.
Both staged arms are released. V12 B1 and wide are already training at about
6.76k and 5.77k charged target tok/s respectively.

The completed late-500 long-decode subset is materially stronger than the
locked 768-token view: MATH `46/100` versus `26/100`, GPQA `21/100` versus
`16/100`, science `46/100` versus `44/100`, AIME unchanged at `2/30`, and
manual composition `10/12` versus `7/12`. GSM8K, code, and BBH long-decode
shards are required before computing a consistent macro.

Those shards now complete. The consistent 1,536-token late-500 board is
GSM8K `79/100`, MATH `46/100`, HumanEval `2/20`, MBPP `4/20`, GPQA `21/100`,
and BBH `47/100`. The five-domain macro is `41.6%` when code is the mean of
HumanEval and MBPP percentages, versus `36.6%` for identical weights under
the locked 768-token budget. This promotes 1,536 tokens for deployment trials
without changing the historical experimental protocol.

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

## Live Product Scoreboard (2026-08-03 16:18 EDT)

| System | Decode | Five-domain dev macro | Key result |
|---|---:|---:|---|
| V11 late-500 | 768 | 36.6% | prior locked leader |
| V11 late-500 | 1,536 | 41.6% | same weights, longer reasoning |
| V11 late-500 + finalizer | 768 + bounded 64 | 42.0% | efficient answer-commitment repair |
| V12 late-1,000 | 768 | 38.2% | 10.22M verified charged targets |
| **V12 late-1,000** | **1,536** | **45.4%** | **current product leader** |
| V12 B1-2,000 | 768 | 35.9% | broad adapter below leader |
| V12 wide-2,000 | 768 | 33.5% | closed negative |
| V13 late-1,000 | 768 | 35.7% | underexposed at 4.11M targets |

The macro is GSM8K, MATH, code mean, GPQA, and BBH. V12 late-1,000 at 1,536
tokens scores `81/48/30/16/52`, with science `30`, AIME `0`, and manual
composition `10/12`. Expanded full-board jobs `733119--733126` are active or
independently pending. Token-matched V13 jobs `732605/732606` and their result
fan `732607--732624` test whether its equal-update loss was only target-length
underexposure. V12 late-layer 2,000-update job `733201` and evaluators
`733212--733220` test the single direct dose-response question.

At peak this campaign used 14 concurrent single-H100 jobs. This is deliberate:
the small models scale poorly with DDP, while independent training arms and
domain evaluators scale nearly linearly across allocations. Every current H100
request has a distinct product decision attached to it; unhealthy nodes
`evc33/evc43` remain excluded.

### V13 token-match decision

The matched late-layer V13 run completes at 10.11M charged targets with
GSM8K 85, MATH 21, HumanEval 25%, MBPP 25%, GPQA 14, BBH 50, AIME 0, and
manual composition 7/12. Its five-domain macro is 39.0%. This closes V13 as a
leader: the extra procedural exposure recovers 3.3 macro points over the
underexposed checkpoint but does not beat V12. No further V13 late-layer dose
or width variants are authorized.

The active positive-direction queue is now:

- `733119--733126`: full 1,536-token V12 late-1,000 board;
- `733201` plus `733212--733220`: V12 late-layer 2,000-update dose response;
- `733293--733299`: V12 late-1,000 deterministic 768+64 finalization path;
- `732605` plus `732607--732615`: token-matched B1 completion and closure.

## Final Rollout-Replay And Selection Result (2026-08-04)

The protected generator is the SmolLM3-3B V12 late-two-layer checkpoint after
1,000 broad updates plus 400 verified rollout-replay updates:

`artifacts/product_reasoning/smollm3-3b-a07cc9a/baseline_late2_v12u1000_rollout_replay_u400_r1/checkpoint_0000400.pt`

Its SHA-256 is
`34c82454e0c53609bc1ac6a9f127437080e431f147e28ae63b4080c413d9a82e`.
Further fresh-SFT and preference-training corrections did not clear the fixed
regression gate, so this checkpoint remains immutable.

The final practical system uses greedy decoding for GSM8K, HumanEval, MBPP,
GPQA-Diamond, and BBH. It generates four autonomous candidates and applies the
disjointly trained completion-shape reranker only for MATH; science uses the
same K=4 reranker and is reported separately. The reranker reads answer-vote,
truncation, explicit-final, repetition, length, and response-shape features,
but never gold answers or correctness labels at inference. Its artifact
SHA-256 is
`b64b2f4a185e41891000123b8d0dc0bf1d5e43b8397b45a8cdd8016427d97160`.

| System | GSM8K | MATH-500 | Code mean | GPQA | BBH | Macro | Solved |
|---|---:|---:|---:|---:|---:|---:|---:|
| corrected greedy leader | 79.303% | 51.000% | 27.948% | 17.172% | 61.600% | 47.404% | 2,292/3,930 |
| **routed K=4 system** | **79.303%** | **67.000%** | **27.948%** | **17.172%** | **61.600%** | **50.604%** | **2,372/3,930** |

The routed system adds 3.200 macro points and 80 primary-board solves with no
primary-domain regression. On the separate 500-row held-out science board it
scores `227/500 = 45.40%`, versus greedy `156/500 = 31.20%`, adding 71 solves.
On AIME-2024 it scores `4/30`, versus greedy `0/30`; this remains a small,
high-variance signal.

MATH uses corrected exact-equivalence labels. Across 2,000 K=4 candidates,
numeric fallback repairs 29 false negatives; first/modal/shape/oracle become
`263/323/335/369` of 500. The original greedy MATH report changes from
`249/500` to `255/500`. These six evaluator repairs are not model gains.
Corrected candidate-bank and shape-report SHA-256 values are
`17ec3a3f2a8baa8b85e1197d0148dc20d1080e54672b3b8ab0788064316b59cf`
and
`4dea0532d1fc853ad70cfb12ea230a512259b4b3f8ee939f8cfab151cb11b6fe`.

Two tempting routes are explicitly closed. Modal self-consistency scores only
`706/1250 = 56.48%` on full BBH, below greedy `770/1250 = 61.60%`; the fixed
100-row gain did not generalize. A counterbalanced zero-shot semantic verifier
also loses to the cheap shape reranker on every non-tied routed domain. The
remaining useful target is a correctness model trained on a disjoint candidate
corpus: current shape-to-oracle gaps are 34 MATH answers and 57 science
answers. This is an inference-selection opportunity, not evidence of native
ETTR reasoning or a reason to mutate the protected generator.

### Supervised late-state correctness head

The one justified successor is also complete and negative. The 8,192-prompt
fresh candidate bank is first rescored with the corrected numeric matcher,
repairing 160 of 32,768 trajectory labels. Sixteen independent H100 jobs then
extract three pooled representations from each of backbone layer offsets
`-1/-2/-4/-8`, yielding 24,576 hidden features plus 32 label-blind shape
features per candidate. A 3,206,497-parameter pairwise correctness head is trained
only on within-prompt correct/wrong pairs while holding out complete prompt
identities.

On 766 held-out prompts the head selects `170`, versus `168` for the existing
shape reranker. The per-domain split is MATH `41/389` versus `44/389` and
science `129/377` versus `124/377`; only science earns a full-board test. On
the 500-row science board the apparent gain reverses: neural `215`, modal
`220`, shape `227`, oracle `284`. The head is therefore closed without a
public MATH run or any pooling/layer/width/seed variant. The generator and
promoted route remain unchanged.

Corrected source, neural model, and full-science selection SHA-256 values are
`20a496867c1afc46d094a1ee2762cc553bd0460bd2915cc7e60d9c53025aa816`,
`aa11215843ef810008f86dbd864459557ccba25ce30424d17dca076783529838`,
and
`780e72b0d94d5cdab01361476ca8c70e7deb38962391779766112b76cc6f901d`.
This result says that candidate correctness is not reliably exposed by these
pooled frozen states. Further progress must come from process-level checking
or stronger generator training, not another static-state pooling variant.

### End-to-end process verifier

The distinct process-level successor is implemented in private commit
`d17a1fd`. It reads the entire problem and candidate trajectory, combines the
final contextual state with the fixed label-blind shape vector, and updates an
isolated copy of the leader's late two layers plus LoRA. The protected
generator is never modified. Prompt-identity hashing reserves 6,539 train,
794 development, and 859 untouched final prompts; training alternates MATH and
science correct/wrong pairs.

Job `736193` completes 300 updates and 2,400 pair presentations in 1,469.25
seconds. At the development-selected update 200, process selection reaches
`203/794` versus shape `194/794`: science gains nine answers and MATH ties.
The untouched final fold reverses the result: process `185/859`, shape
`190/859`; science `142/419` versus `143/419`, MATH `43/440` versus `47/440`.
Model/report SHA-256 values are
`bd4cd0e0030c2cde8599b744e5a030b7697cfdb8723dd0575ff639decdf1cea4`
and
`ddab0c3375492e0b2fcd263123197eaf01ad6ff29beb8c049b517d1cf3c7ddf4`.
The static and end-to-end verifier directions are therefore closed on this
candidate bank.

The immediate replacement is task-specialist routing. Existing fresh-replay
update 200 scored fixed MATH/science `53/38` versus leader `51/31` but was
discarded for unrelated-domain regressions. Full-board jobs `736338/736339`
now test it only as a routed MATH/science expert; no other domain or generator
lineage changes unless those 500-row gates improve.

That gate is now running alongside a stronger domain-only test. Hash-bound
MATH and science corpora contain respectively 5,624 and 4,550 unique prompts,
each split evenly between fresh verifier-correct trajectories and disjoint
in-domain V12 replay. Matched 400-update warm starts finish in 17 minutes.
MATH checkpoints score `52/54/56` on the fixed 100-row board; science scores
`35/31/35`. The science branch is rejected. MATH update 400 advances to one
full 500-row K=4 generation/shape-selection gate, while the earlier mixed
update-200 checkpoint is measured on full MATH and science boards in
parallel. Promotion requires beating the existing corrected routed results,
not merely the old fixed subset: MATH `335/500` and science `227/500`.

### Routed MATH specialist promotion

The mixed fresh-replay update-200 checkpoint clears that scale gate. It has
SHA-256
`baf4623755d26ae13b4a8de8b304c07ae197d95a51f0119e0c67f62a0aa139b5`
and is used only as the MATH candidate generator; every other primary domain
stays on the protected leader. Across the full 500-row MATH board its K=4
candidate bank scores first/modal/shape/oracle `273/337/353/371` after exact
numeric rescoring. Shape selection is `70.60%`, 18 answers above the former
protected-leader K=4 route at `67.00%`.

| Routed system | GSM8K | MATH-500 | Code mean | GPQA | BBH | Macro | Solved |
|---|---:|---:|---:|---:|---:|---:|---:|
| former protected-leader K=4 route | 79.303% | 67.000% | 27.948% | 17.172% | 61.600% | 50.604% | 2,372/3,930 |
| **mixed-u200 MATH specialist route** | **79.303%** | **70.600%** | **27.948%** | **17.172%** | **61.600%** | **51.324%** | **2,390/3,930** |

This is a real generator-plus-selection gain: numeric rescoring repairs 25
candidate labels but does not account for the 18-answer delta because both
routes use the same corrected evaluator. Corrected candidate and shape-report
SHA-256 values are
`c21cabd81d809d778e6369d607449389dc17226cc3fe032463a412085b05803a`
and
`662fafa746aa92e38e54d3f28232afbde155dc94afb4a1235e16fcfd79a301d3`.

The science fan and domain-MATH update-400 fan remain one-shot gates. Their
new floors are `227/500` science and `353/500` MATH. Completed GPU candidate
shards are immutable; downstream selector failures are repaired from their
aggregates rather than by rerunning inference.

The science result clears its floor by two answers. Corrected
first/modal/shape/oracle scores are `186/220/229/308` of 500, so the mixed
update-200 generator plus the unchanged shape selector becomes the science
route at `45.80%`. This is only a 0.40-point final-answer gain over the prior
route, but its oracle rises from `56.80%` to `61.60%`. The generator has
created 24 additional solvable candidate groups that the current selector
does not capture. Corrected-candidate and shape-report SHA-256 values are
`4bdc0e304f1fc3dfd07ad04fa00114b9a74c660ebce4319873f7fcc828f1faf5`
and
`559cfb8235cb84979ba8cd8bbe9d03ebfca7ee3ed3c806418b573621bb472156`.

`pipeline/merge_product_candidate_sets.py` supports the resulting bounded
K=8 diagnostic. It merges independent draws only when identity sets,
questions, gold values, task labels, training groups, and per-source sample
indices agree exactly, then reindexes the supplement contiguously. Additional
inference is promoted only if unchanged shape selection beats the K=4 route;
oracle-only gains are diagnostic.

The domain-only MATH update-400 branch fails its full-board decision despite
the positive fixed subset. Its corrected first/modal/shape/oracle vector is
`266/330/352/367`; shape is one answer below the mixed update-200 route at
`353`. Domain-only MATH is closed. The mixed specialist also fails to transfer
to AIME-2024: first/modal/shape/oracle are all `2/30`, below the retained
protected-leader route at `4/30`. This is an oracle-level generator failure,
so AIME receives no additional sampling.

The mixed specialist's full greedy results are `257/500` MATH and `171/500`
science, confirming that its improvement exists before selection. The final
bounded inference gate is K=8 on MATH, where the accepted K=4 result still has
an 18-answer oracle gap. Jobs `736767--736776` add four independent draws and
`736777` performs the validated merge and unchanged selectors. Promotion
requires shape selection above `353/500`; oracle-only lift does not count.

In parallel, the next training-data wave is frozen without touching public
evaluation labels. After excluding V12, V13, and prior rollout banks V1/V2/V3,
4,096 prompts are selected from 34,571 still-admissible answer-verified
OpenMath rows. The bank SHA-256 is
`e8a03468d8b229ad06d84e15931ab6daa0e6f90e763780847b858fefd6719ead`.
The initial under-batched jobs `736832--736863` are canceled before allocation.
Replacement jobs `736893--736924` search the same exact slices with four
trajectories per prompt at the previously validated prompt batch 64
production shape (`674.42` generated tok/s, 49.20GB peak); `736925` admits
the ledger only after complete coverage and at least 512 verifier-correct
identities. This is data production, not a model result. Any continuation
waits for the ledger hash and a frozen mix.

### Choice-equivalence evaluator correction

Transcript review finds a correct hyperbola derivation whose bare option `E`
was scored against `\\text{(E)}` as wrong. Commit `d69e9e6` adds exact
single-letter A--E equivalence to the MATH matcher; all 27 focused evaluator
and selector tests pass. This correction is reported separately from model
gain.

With the same corrected matcher, protected-leader K=4
first/modal/shape/oracle is `264/326/338/371`, while mixed-update-200 K=4 is
`275/339/355/372`. The mixed specialist therefore retains a 17-answer,
3.40-point generator-plus-selection advantage. The current primary route is
`79.303 / 71.000 / 27.948 / 17.172 / 61.600`, macro `51.405%`, and
`2,392/3,930` solved. Mixed corrected-candidate and shape-report SHA-256
values are
`ab06c170c4e819f771eaed76de95cee8dc652e69e33e7c4bbc65e329fa01a41c`
and
`70382e8ebac85e2daea00e81facd8859e72deb1ac870cf4b3227b8806ee852b7`.
The live K=8 gate must beat `355/500` under this evaluator.

The corrected immutable Newton rollout runtime is
`product_rollout_runtime_d69e9e6_r1`. V4 jobs are replaced before allocation
as `736941--736972`, with aggregate `736973`; this avoids dropping valid bare
choice labels from the future verified-positive ledger.

### K=8 MATH promotion

Four additional independent candidates per MATH identity are merged with the
immutable K=4 bank under the validated contiguous-index contract. With the
choice-aware evaluator, K=8 first/modal/shape/oracle is
`275/360/366/398` of 500. Shape reaches `73.20%`, 11 answers above K=4 at
`71.00%`.

| Current routed system | GSM8K | MATH-500 | Code mean | GPQA | BBH | Macro | Solved |
|---|---:|---:|---:|---:|---:|---:|---:|
| **mixed-u200 MATH K=8** | **79.303%** | **73.200%** | **27.948%** | **17.172%** | **61.600%** | **51.845%** | **2,403/3,930** |

Corrected-candidate and shape-report SHA-256 values are
`f4a12477883ac5ae8c0ad6ce06d284a32652fe55870c38b3a56b44fd2f88b90e`
and
`8551baa43471bbbd60ed69d23fc85e0a8bb86f7f682d19cf29690db8abf968d9`.
No K=16 extension is authorized: K=8 already leaves a 32-answer
shape-to-oracle gap, so the next bottleneck is generator/data quality rather
than candidate availability. Released H100s immediately backfill the V4
verified OpenMath rollout jobs.

### V4 target-quality correction

Manual inspection catches a defect before V4 training: the previous rollout
positive path could validate an answer mentioned inside a max-token-exhausted
draft, then store a response that ended mid-derivation. Correct-answer
verification did not imply a complete training target. Commit `668eb01`
repairs future candidate ranking and adds a hash-bound join to the original
answer-verified DeepSeek-R1 response for every immutable V4 prompt.

All 4,096 bank identities join exactly. Tokenizer-exact 4,096-context
filtering then retains 2,172 complete, unique traces totaling 5,002,781 target
tokens, with zero selected prompt or response truncations. Every retained
trace has balanced reasoning tags and an explicit final answer. The admitted
data SHA-256 is
`d44b93aaddca5aac44a1e944b128d2487e4bdf2413a7e5fc615c64f5b6600f56`;
its report SHA-256 is
`66635a11b3598337f1ce6d01100975dbdea892db2006fd64cce7a173fa307893`.

One conservative warm continuation, job `737104`, starts from the promoted
mixed update-200 specialist with LR `1e-6`, BS4/ACC4, and checkpoints at
100/200 updates. Source and both checkpoints are compared immediately by
same-runtime fixed MATH jobs `737115--737117`. This is a data-quality
intervention, not another width, seed, or sampling variant.

That continuation is now closed as a negative. It completes 200 updates and
7,385,868 charged targets in 531.9 seconds at 13,885.4 target tok/s. Under the
repaired scorer, which replays each saved finalization rather than silently
discarding it, source/update-100/update-200 score `73/69/71` on the identical
fixed 100-row MATH gate. The intermediate `61/60/61` rescore is tooling-invalid
and must never be cited: it rescored drafts without the saved finalization.
The fair update-200 delta is therefore `-2`, so neither checkpoint advances to
K=8. Training-report and update-200 checkpoint SHA-256 values are
`3e3d2942ec3d89c2d43abb8f7308d58047f2433de3442f323611729e56835f7f`
and
`cf65e9111d47ea9fb00bc57b425402f430de73f56aeea9361e653ec5afc4fb9b`.

The completed V4 search remains useful as a capability-frontier map. Four
trajectories for each of 4,096 disjoint prompts produce 16,384 candidates,
5,007 correct candidates, and 2,176 prompt identities with at least one
correct answer (`53.125%`). Candidate, positive-ledger, and aggregate-report
SHA-256 values are
`cc8c3895fc4d65d6c2e9ead0fada3a03fedae4d08d28b3ba41c3f4ce2ea4aefa`,
`3e2adaae6ee47d25563188e3b94d4de8eddf1c44610977d167b933b6db029358`,
and
`0bf3dec0282d25dc96212dc44e0c2e00a1cc71ab847a11f0e9a3d9250a33b2d6`.

The sole remaining nonredundant curriculum test uses those 2,176 identities
only as a selection mask, while restoring the complete verified teacher
response for every selected prompt. The joined data have SHA-256
`b001068dbba1fce95002b7a7c42729a1afeb38a1f75faf31e6bb200d9f0e2d02`.
Tokenizer-exact filtering retains 1,734 complete rows and 3,705,995 available
targets; the frozen training subset contains 1,684 rows and 3,600,762 targets
at SHA-256
`34380cd46602d54185be45d85859d2da0b4985922eef32593f2b08cc2a2c1580`.
Job `737170` completes one bounded 100-update curve from the same promoted
specialist in 267 seconds at 14,450.0 charged target tok/s. Jobs
`737171/737174` test updates 50/100 against the unchanged fixed gate. Their
repaired scores are `66/67`, both below the source floor of `73/100`.
Rescore-report SHA-256 values are
`a4d9ae0fe973a3be3af9a376f7f1d069e89f0bd62673c8a1220e1e425e088cb4`
and
`9a59a7f5ae6fb49fae0b091377142a68197fcb75c26917413878e7b05e2c9061`.
Teacher continuation is closed without another data/LR/duration retry.

This paired negative isolates a useful distinction. Complete teacher traces
are not sufficient: both the uniform and student-frontier subsets overwrite
useful behavior. Prior student self-distillation is still the only weight
update that produced the promoted MATH specialist. The final data-quality
intervention therefore admits only verifier-correct, non-exhausted student
drafts whose explicit terminal answer is present in the autonomous draft
itself. The V4 bank contains 691 such prompt identities; 1,309 additional
positive identities require a separate finalization and 176 have no clean
terminal trajectory, so neither category is silently converted into a
reasoning target.
