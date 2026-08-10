# WTV1: Whole-Trajectory Verifier Canary

**Status:** prospectively frozen before verifier scores  
**Date:** 2026-08-10  
**Host:** pinned Qwen3.6-35B-A3B  
**Scope:** already-open source-disjoint development diagnostic; holdout sealed

## Thesis

DSET, GSET, and ISET disagree on only 125 of 1,908 presentations. Every
disagreement contains at least one exact complete trajectory, while the 14
rows missed by all three are unanimous. Their coherent whole-trajectory oracle
is therefore 1,894/1,908 (99.266%). Majority voting is worse than ISET
(1,830 versus 1,838), so frequency is not a useful commitment rule.

WTV1 asks whether the same frozen Qwen host can semantically score each distinct
complete trajectory given the exact source and original model-owned draft. It
selects one complete lineage; it never averages fields, edits a candidate,
reads the gold answer, calls a verifier/tool, or repairs output at inference.
Verdicts use counterbalanced A/B next-token log odds to cancel fixed label
preference. The trained DSET checkpoint is used only to retain the qualified
same-family host and its pinned NF4 loading path; no weights are updated.

## Frozen canary

Only the 125 disagreement groups are scored. Exact duplicate trajectories are
deduplicated before scoring. The remaining 1,783 unanimous rows are carried
through unchanged: 1,769 correct and 14 wrong. Candidate construction, host,
checkpoint, prompt, context length, counterbalancing, and selection by maximum
score are immutable before output.

The canary passes only if all conditions hold:

- at least 105/125 disagreement groups select an exact trajectory;
- combined exactness is at least 1,874/1,908;
- choice exactness is at least 220/256;
- no verifier prompt is truncated.

A pass authorizes a causality control with source/draft associations shuffled
and then one trained compact verifier gate on source-disjoint candidate data.
A miss closes this zero-shot verifier without prompt, score, checkpoint, or
threshold variants. No sealed holdout is opened by this canary.

## Claim boundary

A pass is evidence that complete candidate lineages contain useful
complementary reasoning and that a model-owned semantic commit can recover it.
It is not yet a standalone architecture claim because it uses three trained
candidate producers and an extra model pass. Compute, latency, and matched
single-path controls remain required before promotion.
