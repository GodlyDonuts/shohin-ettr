# Shohin Phase 2 Fresh Corpus Sourcing

Status: sourcing and admission plan, 2026-08-08. This document does not admit
any quarantined corpus or authorize a large scratch run.

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

The 5B canary is not evidence that 5B tokens are enough to train a competitive
model. It is the smallest economical gate for optimization, data-mix, and
learning-curve decisions before hundreds of billions of token presentations.
