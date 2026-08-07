# DIVERGE-QTE1: Qwen Transaction Entailment

Status: closed after one sealed confirmation. No training, prompt repair, or
QTE1 variant is authorized.

## Hypothesis and boundary

PQI1 and GTI1 show that unconditional role classification and transaction
emission memorize renderer layouts at 125M/135M. QTE1 changes both capability
floor and semantic computation:

> A pretrained 0.8B language model can ground a complete candidate transaction
> by judging whether a candidate-specific semantic claim is entailed by the
> source instruction. Whole-candidate entailment, not unconditional role
> emission, supplies the QUERY transaction.

The pinned parent is
`Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`, loaded from the
existing hash-verified Newton artifact. It receives the canonical anonymous
query plus one of two complete claims: candidate A is the requested source and
candidate B the distractor, or vice versa. It scores complete `YES` and `NO`
conditional likelihoods and selects the candidate with larger entailment log
odds. No target label, answer, state value, executor output, teacher, tool,
search, update, or fallback enters inference.

QTE1 is a capability-floor component result, not a sub-200M Shohin result and
not an open-domain reasoning claim. If qualified, its logits may later become
distillation targets for a smaller model-owned semantic owner.

## Fixed development evidence

Read-only job `744554` was implemented and committed before its score was
known. On the opened 768-query development board it reaches `768/768`, every
mode `256/256`, and every renderer `128/128`. Context scrub falls to
`384/768`; mapped mention swap remains `768/768`. The result SHA-256 is
`cbd3dbdd26bc894d5f8ff337ecc2b4e6996d09b390a300abfe245e35981bc269`.

## Sealed confirmation gate

The independent 256-episode PQI1 confirmation board remains sealed at SHA-256
`27f198680cc7bcd7e0203949fe4dee1658fc057fe75302b7e2d43d74321201b8`.
One direct run of the unchanged scorer is admitted. Promotion requires all of:

- QUERY at least `765/768`;
- every mode at least `254/256`;
- every renderer at least `127/128`;
- context scrub at most `430/768` and at least 250 below normal;
- mapped mention swap at least `765/768`;
- exact pinned model/data hashes and only legal complete candidate claims.

A pass qualifies one zero-training composition with protected TOL3 WORLD,
protected NVE1 EVIDENCE, exact factorized execution, and QTE1 QUERY. That
composite must pass the existing end-to-end semantic floors before natural
PL1 can run. A confirmation miss closes QTE1 without prompt, wording, label,
normalization, model, renderer, or threshold variants.

## Sealed result and attribution

Job `744558` scored QUERY `768/768`, every mode `256/256`, every renderer
`128/128`, and mapped mention swap `768/768`. Context scrub scored `512/768`,
above the frozen `430/768` maximum, so QTE1 formally fails and dependency-held
composition `744559` canceled without allocation. Result SHA-256 is
`ee55580e3aa0b140990ac983543b0eacd8657987e128b18d434f6e5749570b6e`.

The one read-only accounting attribution explains the failed control without
changing the decision. Every scrubbed source is exactly `alpha then beta`.
The sealed board contains 512 transaction-0 targets and 256 transaction-1
targets, and QTE1 predicts transaction 0 on all 768 scrubbed rows. Its
`512/768` is therefore exactly the majority-class baseline, not retained
source semantics. Attribution SHA-256 is
`7475c231956d0d5c5107c77344c06996eac7738170c3116bdad826c544e269d8`.

QTE1 remains useful evidence that candidate-wise entailment at the 0.8B
capability floor can read all held semantic renderers. It is not promoted as a
causally qualified transaction owner because the frozen conjunctive gate
failed. No QTE1 retry is allowed; the next mechanism is the independently
frozen outcome-grounded CGL1 lane.
