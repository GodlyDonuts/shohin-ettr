# Phase 2 Quality-First Data Selection Standard

Status: active admission standard, 2026-07-28.

## Objective

Shohin is not being optimized only for public benchmark scores. The corpus must
support:

- clear and coherent language generation;
- factual and conceptual knowledge;
- mathematical and algorithmic problem solving;
- useful programming behavior;
- long-context explanation and composition;
- calibrated uncertainty and correction;
- the ETTR architecture's native state, intervention, and query mechanics.

No dataset is admitted because it is large, new, popular, or associated with a
strong benchmark result. Every source is a candidate until a hash-bound
admission receipt proves its exact selected payload passed the gates below.

## Two Streams

1. **General pretraining** supplies language, knowledge, code, mathematics,
   science, reference material, and procedural representations.
2. **ETTR-native training** supplies exact architecture episodes, state
   transitions, interventions, invariant views, and causal controls.

ETTR is sampled as a separate update stream. Converting its structured records
to ordinary prose or concatenating them into web-text shards would destroy the
architecture-specific supervision and make the exposure budget unauditable.

## Current Diagnosis

The structurally scanned historical inventory is 62,426,256,278 tokens, but it
is dominated by math and Python. FineMath-4+ is also nested within FineMath-3+
upstream, so nominal source totals are not unique-token totals until
cross-corpus deduplication is complete. This inventory is useful but is not a
balanced foundation for a model intended for broad use.

The 25B DCLM and 5B OpenMath candidates are quarantined. DCLM samples include
good prose mixed with forums, dated news, and SEO-like text. OpenMath samples
include useful solutions mixed with at least one unreliable derivation.
Structural entropy and zero byte fallback prove encoding health, not truth or
educational value.

The immediate correction is to acquire and test broad educational prose,
high-quality long-form PDF/reference/science material, and license-resolved
educational code before adding more unverified math volume.

Initial 1,000-row read-only probes passed schema and sampled-contamination
intake for FineWeb-Edu and English FinePDFs-Edu. Both exposed document IDs,
source paths/URLs, language metadata, and quality fields; neither sample had an
exact or 13-gram overlap with the live evaluation prompt index. FinePDFs-Edu
also exposes extractor, truncation, per-page language, duplicate-cluster, and
page-boundary metadata that can drive its extraction audit.

The consolidated Common Pile/Comma training dataset failed Shohin's provenance
gate because its streamed rows expose only `text`. Its component datasets
remain valuable, but they must be acquired directly so source and license
metadata are not discarded. The convenience mixture is not an admissible
training source.

Follow-up component probes found that direct peS2o rows preserve IDs, source,
version, dates, and metadata. Stack-Edu Python rows preserve blob IDs,
repository names, paths, detected licenses, encodings, language, and quality
scores. Stack-Edu intentionally contains metadata rather than code content;
selected blobs must be retrieved from Software Heritage only after license
resolution, then parsed, scanned, deduplicated, and repository-split.

## Current Source Slate

### P0 candidates

| Candidate | Intended role | Required selection |
|---|---|---|
| FineWeb-Edu | broad explanatory web core | English, pinned revision, source/domain caps, score retained as metadata, quality-stratified ablation |
| FinePDFs-Edu | textbooks, manuals, reports, long-form explanations | English, language-switch filter, formula/layout audit, document and publisher caps |
| Common Pile components | licensed books, science, reference, Stack Exchange, Wikipedia, government material | component-specific licenses and provenance, boilerplate removal, component caps |
| Stack-Edu | educational code and software knowledge | retrieve by SWHID, resolve per-file license, remove generated/vendor/minified code and secrets |
| Nemotron-CC-Math 4plus | high-quality math candidate | legal review first, upstream-only access, residual dedup, math correctness sample |
| Existing FineMath/OpenWebMath | retained math substrate | cross-source residual only; never double-count nested FineMath subsets |
| First-party verified procedural data | exact state, algorithm, correction, and execution examples | generator/version receipt, solver or execution verification, family holdouts |

### Held or rejected by default

| Candidate | Status | Reason |
|---|---|---|
| raw FineWeb | held | scale is high but semantic density is too variable for a 125M model's core budget |
| raw FinePDFs | held | lightly filtered extraction pool; use the educational subset |
| DCLM wholesale | held | manual samples show mixed utility; retain only a scored residual |
| OpenMath wholesale | held | synthetic derivations require final-answer and trace-consistency verification |
| FinePhrase | rejected from core | synthetic paraphrases can contain hallucinations and truncation |
| consolidated Common Pile/Comma text mix | rejected | streamed rows omit per-document provenance; use the component datasets directly |
| unlicensed/unknown-license code | rejected | unusable for a model intended for distribution and downstream use |
| benchmark-derived training examples | rejected | benchmark memorization is not general capability |

## Candidate Mix

This is an equal-token ablation proposal, not a production mixture:

| Slice | Percent |
|---|---:|
| FineWeb-Edu selected | 30 |
| FinePDFs-Edu English selected | 15 |
| licensed reference/science/books | 12 |
| Stack-Edu license-resolved | 15 |
| verified high-quality math | 12 |
| Essential-Web selected | 8 |
| DCLM high-quality cross-source residual | 5 |
| first-party verified procedural | 3 |

The proposal deliberately reduces math concentration relative to the historical
inventory. Source weights are frozen only after matched-token ablations. The
late-pretraining or cooldown mixture may upweight verified procedural, code,
and high-confidence explanatory data, but it must remain a separately measured
stage.

## Mandatory Admission Gates

### 1. Provenance and legal use

- Pin dataset repository and file revisions.
- Record upstream URL, configuration, split, source identifiers, and retrieval
  time.
- Preserve per-document provenance.
- Resolve per-file licenses for code and per-document licenses where required.
- Review both data-use terms and possible trained-model redistribution terms.
- Never mirror gated NVIDIA rows into Shohin's Hugging Face repository.

### 2. Structural integrity

- Reject malformed, empty, decode-corrupt, or pathological documents.
- Scan token entropy, repeated-token concentration, byte fallback, language,
  document length, and extraction artifacts.
- Audit PDF formula, table, header/footer, and language-switch behavior.
- Parse code and remove minified, generated, vendored, binary, and duplicated
  repository content.

### 3. Semantic quality

- Use deterministic, stratified samples by source, score bucket, domain,
  language, length, and time period.
- Review at least 1,000 documents per candidate before bulk selection, with a
  minimum 100-document human adjudication slice.
- Score clarity, correctness, completeness, educational value, originality,
  and real-user usefulness separately.
- Do not treat an upstream classifier score as ground truth. Measure each score
  bucket with Shohin's tokenizer and downstream ablations.

### 4. Verifiability

- Math: verify final answers when possible, reject contradictory steps and
  incomplete/self-correcting traces, and cap unverifiable synthetic solutions.
- Code: parse or compile, execute tests in a sandbox when tests exist, scan for
  secrets and unsafe payloads, and preserve license metadata.
- Factual/reference: spot-check claims and dates against stable references;
  reject content farms and unsupported pseudo-expertise.
- Procedural: require exact simulator, solver, or execution receipts.

### 5. Deduplication and contamination

- Exact and near-deduplicate within every source.
- Deduplicate across the fully assembled corpus, with retention priority given
  to verified, clearer, better licensed originals.
- Treat FineMath-4+ as nested in FineMath-3+ and retain only one copy.
- Decontaminate against public evaluations and private held-outs both before
  tokenization and after cross-source assembly.
- Hold out complete domains, repositories, generators, templates, and ETTR
  semantic families where the evaluation requires generalization.

### 6. Privacy and safety

- Scan for PII, credentials, private keys, malware, exploit payloads, and
  accidental private data.
- Exclude disallowed or unsafe records before publication or training.
- Keep removal identifiers so future opt-outs can be applied reproducibly.

### 7. Measured utility

Run equal-token, equal-update ablations from the same initialization. A source
must improve the aggregate utility battery without materially harming clean
held-out general language modeling or ETTR causal controls.

The battery includes:

- held-out NLL by source and domain, including prose not represented by public
  benchmarks;
- factual completion and calibration;
- long-form coherence, summarization, and document continuation;
- mathematical answer verification;
- code compilation and unit-test execution;
- procedural state tracking and correction;
- private human-written interaction prompts;
- ETTR treatment versus reset, deranged-binding, query-only, and dense-state
  controls.

Public benchmark scores are reported, but no source is selected solely because
it improves a public benchmark.

## Publication Contract

The Hugging Face dataset should contain Shohin-owned or redistribution-cleared
selected data, manifests, provenance, removal identifiers, quality reports,
and checksums. For sources that cannot be mirrored, publish only the immutable
source manifest and deterministic selection recipe; training must stream from
the original upstream repository.

Every admitted payload requires a no-replace receipt binding:

- source revision and selection code commit;
- raw and selected manifest hashes;
- filter counts by reason;
- dedup and contamination report hashes;
- semantic-review report hash;
- license decision;
- tokenizer and tokenized-shard hashes;
- matched-token ablation results;
- final approved token count and mixture ceiling.

## Fresh-Source Challenger Lane

Before each major pretraining tranche, refresh the registry against newly
released first-party dataset cards and papers. A new source never replaces the
core immediately. It enters a challenger lane:

1. pin the upstream revision and legal terms;
2. run a read-only 1,000-row provenance/contamination probe;
3. run a deterministic stratified 10,000-row structural and semantic profile;
4. human-adjudicate at least 100 retained and rejected documents;
5. build a small selected residual after cross-corpus deduplication;
6. compare it with the incumbent source at equal tokens and updates;
7. promote only on aggregate utility, with regressions and uncertainty shown.

This lets Shohin use genuinely better new data without turning dataset recency
or marketing claims into an unmeasured training decision.

## Decision Rule

Quality per token is the objective. A smaller selected payload beats a larger
uncurated one. If a source fails correctness, legal, privacy, contamination,
or utility gates, it is rejected or reduced to a residual slice regardless of
its popularity or advertised scale.
