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

The newly released Stack v3 is a stronger code challenger than a blind
Stack-Edu expansion because it carries inline file content, repository commit
identity, cross-language near-deduplication, PII redaction, and file-level
license metadata from an August 2025 GitHub snapshot. It is still not a
wholesale source: its scale includes a large low-star, toy-project, and
no-license tail. Shohin will retain only explicitly permissive files, apply
repository-quality and execution gates, and compare the resulting residual
against Stack-Edu and the historical Python corpus.

A pinned 1,000-file Stack v3 intake rejected the random feed as training data:
970 files had `no_license`, only 30 were classified permissive, 57 were exact
duplicates, and the manually reviewed permissive tail still contained
zero-star toy or placeholder repositories and trivial files. The sample had
no exact or bounded 13-gram evaluation overlap. Stack v3 is therefore a useful
raw reservoir, not a corpus. Its only reopenable path is a repository-level
residual with explicit permissive licenses, maturity or trusted-project
evidence, commit/content deduplication, parsing or compilation, tests and
documentation preference, secret scanning, and repository-level holdouts.

Deterministic 10,000-row profiles sharpened the source policies:

- FineWeb-Edu had zero sampled exact duplicates or evaluation overlaps, but
  8,655/10,000 rows were only `int_score=3`. Manual review found that bucket
  mixed useful explanations with content farms, promotional pages, weak
  medical/financial advice, and dated low-value articles. Score 4+ is the
  core candidate; score 3 is residual-ablation only. The complete
  100-document model-preliminary pass retained only 16 core and 31 residual
  records and rejected 53. Its text-free receipt remains explicitly
  ineligible for training pending human review.
- English FinePDFs-Edu had zero sampled exact duplicates or evaluation
  overlaps and a useful long-form tail, but review exposed answer-key spam,
  newsletters, catalogs, low-confidence pages, and extraction artifacts
  alongside excellent lectures, manuals, and essays. Selection must use page
  language confidence, repetition, length, extraction, publisher, and
  document-type gates rather than the dataset label alone. The complete
  model-preliminary pass retained 27 core and 51 residual records and rejected
  22; the corresponding receipt is not a human admission.
- Direct peS2o had zero sampled exact duplicates or evaluation overlaps and
  consistently substantive open-access scientific papers with strong
  provenance. The complete preliminary pass retained 63 core and 28 residual
  records and rejected nine. It is the strongest of these three direct
  candidates, but remains capped because journal prose is narrow, variable in
  explanatory quality, and occasionally extraction-damaged. Its receipt also
  remains ineligible for training pending human review.
- Stack-Edu Python is metadata-only; 8,207/10,000 sampled records were
  `no_license`. Its nominal 125B-token scale therefore cannot be treated as
  usable scale. Content quality is unmeasured until strict allowlisted blobs
  are retrieved and scanned.
- FinePhrase corrected 1,000-row probes covered FAQ, math, table, and tutorial
  generated outputs. Sampled evaluation overlap was zero, but manual review
  found invented quantities and arithmetic, source-unrelated tutorials,
  malformed or fabricated tables, unsupported medical advice, and outputs as
  short as a few characters. Polished structure is not semantic quality.
  FinePhrase is rejected wholesale. FAQ/tutorial records may be reconsidered
  only as a paired source-faithful residual; math requires solver verification
  and tables require cell-level source support. All configurations remain at
  zero mixture weight until equal-token utility ablations pass.

Direct, revision-pinned Dolma 3 component probes further show why ingredients
must be adjudicated separately:

- The `cc_hq_science` shard had no sampled duplicate or evaluation overlap,
  but a complete 100-document preliminary semantic pass retained only 23 as
  core and 30 as residual while rejecting 47. Rejections included content
  farms, promotion, opinion, obsolete or refuted articles, pseudoscience,
  extraction corruption, and non-science pages. The web-science classifier is
  therefore not a training admission.
- The `olmocr_science_pdfs` shard contained 306 rows, including 34 exact
  duplicates. Its 100-document preliminary pass retained 26 as core and 57 as
  residual while rejecting 17, including ten removed tombstones and seven
  weak or questionable publications. This is a promising filtered
  scholarly-PDF residual, not a wholesale core.
- The `stack_edu` FIM Python shard had no exact duplicates and one bounded
  13-gram evaluation overlap among 1,000 rows. In the 100-file review packet,
  92 records had no detected license; only eight were permissive. The
  preliminary pass retained one core and seven residual records and rejected
  the other 92. Random FIM code cannot enter a distributable model merely
  because an upstream educational score is high.

These three semantic passes are explicitly `model_preliminary`, contain no
document text in Git, and are ineligible for training admission. Their value
is policy falsification and filter design. A complete human packet plus all
structural, legal, privacy, decontamination, tokenization, and equal-token
utility gates remains mandatory.

## Current Source Slate

### P0 candidates

| Candidate | Intended role | Required selection |
|---|---|---|
| peS2o | open scholarly science and technical prose | pinned revision, explicit OA-license allowlist, extraction/repetition filters, source caps, human semantic review |
| FineWeb-Edu | broad explanatory web core | English, pinned revision, source/domain caps, score retained as metadata, quality-stratified ablation |
| FinePDFs-Edu | textbooks, manuals, reports, long-form explanations | English, language-switch filter, formula/layout audit, document and publisher caps |
| Common Pile components | licensed books, science, reference, Stack Exchange, Wikipedia, government material | component-specific licenses and provenance, boilerplate removal, component caps |
| PleIAs Common Corpus components | open books, culture, government, science, and reference challenger | document-level license/provenance, OCR/date/source caps, component-specific residual |
| Stack-Edu | educational code and software knowledge | retrieve by SWHID, resolve per-file license, remove generated/vendor/minified code and secrets |
| Stack v3 | current repository-context code challenger | permissive files only, repository and commit provenance, parse/compile, secret rescan, cross-version residual |
| Dolma 3 Dolmino components | proven high-quality web, PDF, STEM, and code challengers | inspect ingredients separately; never inherit the synthetic-heavy convenience mix wholesale |
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
| FinePhrase | rejected wholesale; verifier-backed residual only | sampled generations contain hallucination, source drift, malformed structures, near-empty outputs, and generator-style concentration |
| consolidated Common Pile/Comma text mix | rejected | streamed rows omit per-document provenance; use the component datasets directly |
| unlicensed/unknown-license code | rejected | unusable for a model intended for distribution and downstream use |
| benchmark-derived training examples | rejected | benchmark memorization is not general capability |

## Candidate Mix

This is an equal-token ablation proposal, not a production mixture:

| Slice | Percent |
|---|---:|
| FineWeb-Edu score 4+ selected | 30 |
| FinePDFs-Edu English selected | 15 |
| licensed reference/science/books | 12 |
| license-resolved educational code (Stack-Edu/Stack v3 winner) | 15 |
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
- For transformed or synthetic records, review the source and generated output
  together. Generated prose that cannot be shown faithful to its source is
  rejected even when fluent.

### 4. Verifiability

- Math: verify final answers when possible, reject contradictory steps and
  incomplete/self-correcting traces, and cap unverifiable synthetic solutions.
- Code: parse or compile, execute tests in a sandbox when tests exist, scan for
  secrets and unsafe payloads, and preserve license metadata.
- Factual/reference: spot-check claims and dates against stable references;
  reject content farms and unsupported pseudo-expertise.
- Procedural: require exact simulator, solver, or execution receipts.
- Synthetic transformations: require a source-output receipt, successful
  generation termination, nontrivial bounded length, and task-specific
  verification. A format classifier is not a correctness verifier.

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

The admission order is fixed:

1. verify the source-specific v3 candidate and human semantic decision;
2. materialize the cross-source exact residual;
3. audit and materialize the exact-confirmed near residual;
4. freeze an immutable train/document-validation/domain-validation partition;
5. run structural, contamination, privacy, and license audits on every
   partition;
6. train each source arm from the same checkpoint for the same target tokens,
   updates, batch geometry, optimizer schedule, and random seeds;
7. score every arm on the same cross-source document and whole-domain holdout
   matrix;
8. admit only an arm whose aggregate gain clears the preregistered regression
   limits.

Validation records are never used by an optimizer, source selector, semantic
reviewer, or mixture tuner. The document holdout measures representative
within-source generalization. The whole-domain holdout assigns all records
from a deterministically selected domain to validation and measures transfer
to unseen publishers/sites rather than local document memorization.

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

`pipeline/tokenize_shards.py` now emits the candidate-side
`shohin-tokenized-shards-v3` receipt. It binds the exact upstream revision,
selection-code hash, tokenizer hash and vocabulary identity, decontamination
index and live evaluation-file hashes, every filter setting and rejection
count, and every compressed shard's path, byte count, token count, and
SHA-256. A compressed text-free document ledger additionally binds every
retained source-row identity and document SHA-256 to its exact shard/token
range and exact uint16 token-span SHA-256. This is required for post-packing
provenance, removal, and cross-source exact deduplication. The manifest has
its own canonical payload hash.
`pipeline.verify_tokenized_shards` independently reopens every compressed
shard, verifies its digest, decompresses it to recover the exact uint16 token
count, recomputes every document token-span hash, rejects unbound files, and
revalidates all physical source, tokenizer, decontamination, and evaluation
inputs when admission requires them. Legacy v2 diagnostics remain readable,
but a new Phase 2 admission requires v3. Passing this verifier proves payload
identity, not source quality or production admission; the other gates above
remain mandatory.

`pipeline/build_general_source_review_packet.py` consumes only a fully
verified v3 corpus. It deterministically samples exact retained documents
across license and token-length strata with a publisher/domain cap, reopens
the hash-bound raw files, and refuses any source-row identity or document-hash
drift. The private packet contains bounded text for adjudication and is
mode-0600; its public-safe receipt contains no document text and binds the
packet SHA-256, ledger, manifest, strata, and verifier result. A model-authored
label remains preliminary; production semantic admission still requires the
declared human authority.

`pipeline/audit_cross_source_exact_dedup.py` consumes two or more independently
verified, internally exact-deduplicated v3 corpora in an explicit retention
priority order. A disk-backed index identifies every repeated document
SHA-256, retains the first declared source, and publishes a compressed,
text-free removal ledger plus a hash-bound accounting report. It requires one
tokenizer identity across all inputs and can revalidate every physical source,
tokenizer, decontamination, and evaluation input. Publication reserves the
destination and writes the report last through no-replace links, so an
interrupted or raced run cannot masquerade as a complete receipt. This audit
does not measure near-duplicates.

`pipeline/materialize_cross_source_exact_residual.py` applies that exact
removal ledger to one named corpus. It first revalidates the complete source
corpus and every external input, binds the canonical dedup report and
compressed removal ledger, reopens each retained document's hash-bound token
span, and repacks only the retained spans into a new no-replace v3 corpus. The
new document ledger preserves source identities while recording new shard
offsets; the output manifest binds the parent manifest, declared retention
priority, removal receipt, and residualizer source. The complete output is
independently reverified before its staging directory is atomically
published. A dedup report alone therefore cannot authorize training: only the
materialized and verified residual may proceed to later gates.

`pipeline/audit_cross_source_near_dedup.py` supplies the subsequent
near-duplicate gate. It reads only verified v3 token spans under one tokenizer
identity. Deterministic bottom hashes of five-token shingles localize possible
matches, but cannot remove a document. Every candidate is reopened from its
hash-verified shard and compared using exact unique-shingle Jaccard and
containment; the declared thresholds and source-priority order are bound in
the report. Exact document-hash duplication causes the near gate to fail,
rather than disguising a skipped exact-residual step. The output is a
text-free, no-replace removal ledger plus canonical accounting report.
Per-corpus selection-code identities are supported so an original corpus and
an already materialized residual can be verified in one chain.

`pipeline/materialize_cross_source_near_residual.py` applies only that
exact-confirmed near-removal ledger. It requires the source corpus's own
selection code, binds the near-audit algorithm and receipts, re-packs retained
token spans into a new v3 corpus, and verifies every output and external input
before publication. The output of this second residualization, not the
near-dedup report or pre-residual candidate, is eligible for later semantic
and utility gates.

`pipeline/materialize_v3_holdout_split.py` is the next mandatory gate. It
classifies each parent-ledger row with namespace- and seed-bound SHA-256
thresholds, assigning a whole domain before considering a document holdout so
one domain cannot leak between train and domain validation. Missing domains
are never treated as one giant domain; they proceed to the independent
document-hash decision. It reopens every exact parent token span and writes
fresh train, representative document-validation, and whole-domain-validation
v3 corpora under a single atomic no-replace root. Each child independently
verifies its shards, ledger, tokenizer, source files, and evaluation inputs.
The root receipt binds the parent, policy, child manifests, document counts,
and token counts.

`pipeline/verify_v3_holdout_split.py` independently merges the three child
ledgers in original source-row order, recomputes every assignment, compares
all non-location provenance fields with the parent, and re-verifies every
physical artifact. A creation receipt alone is not sufficient. The default
production policy reserves approximately one percent by document and one
percent by whole domain; source-specific deviations require a frozen written
justification before splitting.

`train/eval_corpus_nll.py` provides the first matched utility metric. It
requires the exact checkpoint SHA-256 and a fully verified v3 holdout,
constructs the model from the checkpoint's complete configuration, and
measures pure next-token cross-entropy with training z-loss excluded.
Midpoint-stratified, non-overlapping windows span the entire token stream up
to the declared fixed token budget. The no-replace report binds the checkpoint,
corpus manifest, selection code, sampling geometry, evaluated tokens, NLL,
perplexity, and runtime. Self-source NLL alone cannot promote a candidate:
every ablation arm must be scored on all frozen source holdouts.

`train/build_training_data_contract.py` and
`train/verify_training_data_contract.py` control the optimizer boundary.
Every general-training corpus in a contract must be an absolute, final,
non-partial `train` child from the holdout gate, with an exact manifest,
selection-code hash, positive weight, and common tokenizer identity. Contract
construction deep-verifies all physical inputs before no-replace publication.
`train/train.py --data-contract ... --data-contract-sha256 ...` derives its
shard directories and normalized weights from that signed object; conflicting
CLI paths or weights fail. Checkpoints record the contract and source-manifest
identities. A resume cannot silently drop or change a recorded contract; an
intentional curriculum-stage change requires the explicit
`--allow-data-contract-transition` flag.

NLL reports retain one mean NLL per deterministic window.
`train/assess_paired_corpus_nll.py` uses those same-window pairs to report the
candidate-minus-baseline mean, standard error, normal 95% interval, tail
quantiles, and window win fraction. A candidate has a statistically resolved
strict gain only when the upper 95% bound is below zero. Final source
promotion requires this paired evidence over the complete cross-holdout
matrix, not merely its own source holdout.

Legacy historical shards without v3 document ledgers cannot be represented as
cross-source residuals by assertion. They must be rebuilt from pinned source
records into v3 form or excluded from a claim-bearing Phase 2 mixture;
structural scans of legacy token files are not a substitute for document
provenance.

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
