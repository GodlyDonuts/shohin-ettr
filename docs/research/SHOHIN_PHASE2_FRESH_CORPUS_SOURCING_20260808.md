# Shohin Phase 2 Fresh Corpus Sourcing

Status: reusable corpus conversions may finish, but scratch-corpus admission
is no longer the project critical path as of 2026-08-08. This document does
not admit any quarantined corpus or authorize a scratch run.

## Objective

Shohin needs a mostly fresh, broad corpus with unusually high useful-token
density. The historical 62.426B-token math/code stream is continuity data, not
an adequate frontier corpus. Download success, token count, and upstream
quality labels are insufficient admission evidence.

`pipeline/audit_phase2_admission_bundle.py` now requires every corpus in a
training contract to bind its physical manifest to evidence for provenance,
license, privacy, exact and near deduplication, cross-source residualization,
evaluation decontamination, document/domain holdouts, and retained-sample
review. At least 70% of sampling weight must come from sources marked fresh.
Production admission additionally requires a matched equal-token utility gate.

## Primary Candidates

### Broad educational English

[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) is the
primary broad source. Its official card reports 1.3T tokens, ODC-By licensing,
and controlled ablations showing better aggregate downstream results than
unfiltered FineWeb. Hugging Face also publishes a prepared
[100B-token subset](https://huggingface.co/datasets/HuggingFaceFW/fineweb_edu_100BT).
The existing Shohin score-4+ selection is only 8.764B tokens; it should be
recovered and audited first, then expanded without changing the held-out
policy after results are seen.

### Technical and scientific prose

The existing 4.350B-token peS2o balanced residual remains the leading
scientific source because it retains document provenance and explicit OA
licenses. It still requires exact/near cross-source residualization,
document/domain holdouts, privacy closure, and matched utility before weight.
Essential-Web and the bounded FinePDF core are challengers, not automatic
additions.

### Modern code, math, and STEM

NVIDIA's current
[Nemotron pretraining collection](https://huggingface.co/collections/nvidia/nemotron-pre-training-datasets)
is a high-value candidate. Its official
[v2 code card](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v2)
describes approximately 340B GitHub-code tokens, 427.9B Common-Crawl code
tokens, 2.5T refreshed English-web tokens, and specialized STEM data. It is a
gated source governed by NVIDIA's Data Access Agreement for Model Training.
Several synthetic subsets also cite Qwen, DeepSeek, Phi, and other teacher
licenses that may impose downstream conditions. Therefore:

1. no automated acceptance of the NVIDIA agreement;
2. no optimizer weight until exact subset licenses and redistribution terms
   are recorded;
3. prefer source-code and independently executable/verified subsets over
   unverifiable synthetic explanations;
4. execute, compile, or otherwise validate code where practical;
5. residualize against Shohin's existing Python/code and evaluation suites.

### Multilingual coverage

[FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) is a
reproducible ODC-By multilingual corpus covering more than 1,000 languages,
but its official card directs English users to original FineWeb. Use a bounded
multilingual allocation only after tokenizer and target-language evaluation
coverage exist; do not dilute the small model with arbitrary language breadth.

## Provisional Sampling Envelope

These are allocation bounds for corpus construction, not fixed final weights:

| Domain | Canary range | Rationale |
|---|---:|---|
| Fresh educational/general web | 45--60% | broad language, knowledge, and instruction substrate |
| Scientific/technical prose | 12--20% | compact factual and explanatory density |
| Code | 12--20% | executable algorithmic structure and practical utility |
| Verified math/STEM | 8--15% | symbolic competence without returning to a math-only model |
| High-confidence multilingual | 0--5% | bounded transfer and user coverage |
| Historical continuity stream | 10--25% | prevent distribution discontinuity; never count as fresh |

Architecture-native draft/revision and verifier-checked trajectories belong in
reasoning mid-training and posttraining. They are not bulk filler and are
reported separately from unique base-pretraining tokens.

## Staged Admission

1. **250M-token transport corpus:** all physical checks, no capability claim;
   verifies 1/2/4/8/16-H100 scaling, exact resume, and cursor continuation.
2. **5B-token capability corpus:** canary-level admission for every source,
   at least 70% fresh sampling weight, frozen validation sources, and no
   unreviewed synthetic trace stream.
3. **50B-token milestone:** only sources with positive or noninferior
   equal-token transfer and retention evidence expand.
4. **Production corpus:** production-level admission, mostly fresh unique
   data, exact licenses and redistribution terms, and a physical token budget
   consistent with the selected 389M or 919M trunk.

## Current Physical Gate

The frozen five-source exact cross-source audit covers 13,620,050,554 tokens
in peS2o, Essential-Web, bounded FinePDF, Formal Logic, and FineWeb-Edu and
found zero exact cross-source duplicate documents. Those physical corpora use
Shohin's historical 32K tokenizer, while both scratch presets bind a 49,152
vocabulary. A hash-bound 500-document comparison selected the mature SmolLM2
49K tokenizer: it uses 3.744% fewer tokens overall and 7.19% fewer on code,
improves every sampled domain, and has zero round-trip mismatch. The exact
report SHA-256 is
`777d0fd6a28153be73bf55370e95a784582edc37a6a7173021d4d2bc6e15ea2e`.

The obsolete 32K near-audit `768241` and dependency-held residual/holdout jobs
`768243--768251` were canceled before scientific output. Near deduplication
must operate on final token identities, so it will be relaunched only after
the selected corpora are converted to the final 49K tokenizer. The converter
verifies every source token span, removes only the bound source EOS, and
requires the decoded source text length and SHA-256 to match the document
ledger plus exact source encode/decode inversion. It then encodes that exact
text twice with the target tokenizer and requires identical target IDs. Target
decoder inversion is deliberately not required: standard tokenizer
normalization can make arbitrary raw Unicode controls non-invertible even
though replaying the original text deterministically yields the same training
tokens. The converter never reconstructs or edits a document from target
tokens, drops no rows, preserves document identities and metadata, and emits
new hash-bound v3 ledgers without text.

Three canaries established that boundary without publishing a corpus:
`768261` found SmolLM2's non-inversion of one embedded `U+0013`; `768267`
exposed an incorrect all-control-deletion classifier; and `768271` showed that
even a subset-deletion decoder rule was the wrong abstraction. Immutable r4
runtime `phase2_retokenize_a481bc4_r4` has SHA256SUMS SHA-256
`3fa5141b20404c7632369e9f1f52a715c8874f02d0e2026e0b3a83deadcea81d`.
Essential-Web canary `768275` is live against the exact-residual corpus;
FinePDF, peS2O, and FineWeb jobs `768276--768278` are dependency-held behind
it. The canary subsequently completed in `8m23s`, preserving all `108,115`
documents and independently verifying the complete output. It emitted
`295,999,956` target tokens from `308,147,626` source tokens, a `3.942%`
reduction, with manifest/report payload SHA-256 values `f2ce81ed...c93e` and
`9a30a5fd...78c7`. Its pass released `768276--768278` concurrently on three
48-core Stokes nodes. The final-token near-audit specification preserves the
frozen source priority: peS2O, Essential-Web, FinePDF, then FineWeb. It is
dependency-gated on all conversions and uses the existing exact-confirmed
five-token-shingle algorithm and thresholds without alteration. Immutable
runtime SHA256SUMS SHA-256 is `22d958d8...c947`, specification SHA-256 is
`8fd3a734...0344`. Dependency-held job `768282` was canceled before allocation
after the mission redirected to pretrained-host architecture transfer. Its
runtime and specification remain reproducible but do not release
automatically. Formal Logic remains zero weight and is absent from that
specification because its source has no domain provenance and cannot satisfy
the domain-holdout gate.

FinePDF conversion `768276` completed in `3m42s`, retaining all `31,702`
documents and independently verifying `97,196,851` target tokens versus
`100,004,847` source tokens, a `2.808%` reduction. Report and manifest file
SHA-256 values are `f6bae4d8...f497` and `9ac0c611...48d3`. This remains a
quarantined conversion result; it does not satisfy the later admission gates.

Essential-Web job `768237` separately exercises the complete historical-32K
holdout creator and independent verifier as a transport-mechanics canary; its
partitions are not the final 49K training contract. None of these jobs marks a
source admitted. Final-tokenizer near deduplication, privacy, license,
retained-review, and canary utility evidence remain required by the physical
training contract.

The 5B canary is not evidence that 5B tokens are enough to train a competitive
model. It is the smallest economical gate for optimization, data-mix, and
learning-curve decisions before hundreds of billions of token presentations.
