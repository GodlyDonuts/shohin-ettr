# PCF1 completion audit — 2026-08-11

This audit compares the active publication objective with the authoritative
local repository and Newton scheduler/artifact state after the sole submitted
PCF1 graph. It does not authorize or perform a replay.

## Frozen completion requirements

| Requirement | Authoritative evidence | Verdict |
|---|---|---|
| Exactly one prospective PCF1 graph | Slurm root `750976`, terminal job `751004`, and immutable dispatch receipts | **Satisfied**: one graph was submitted once |
| Source-disjoint 1,289-row confirmation | Frozen contract requires the development confirmation to open and score once | **Not achieved**: no `prepared/`, `data/`, candidates, or score artifacts exist |
| Matched trained-revision, unchanged, and self-refinement arms | Frozen contract requires all three generated arms before atomic scoring | **Not achieved**: all downstream jobs `750977--751004` were cancelled without allocation |
| Learned whole-trajectory commit and conservative retention measurement | Frozen contract requires commit scoring, per-domain deltas, and both 95% retention checks | **Not achieved**: no model job or commit artifact exists |
| Complete scientific custody and terminal PASS/FAIL | Frozen contract requires `1289/1289` arm coverage, exact order, zero malformed/truncation, complete hashes/accounting, and `final_comparison.json` | **Not achieved**: `final_comparison.json` is absent and the formal result is `null` |
| Sealed confirmation assessor and protected data | Exact prepare-script order, CPU-only one-second allocation, absent output roots, and terminal receipt | **Satisfied**: no assessor, holdout, public, or product scoring path was reached |
| No prohibited retry, requeue, or successor | Slurm `Restarts=0`; 28 downstream jobs cancelled; queue empty; receipt flags all false | **Satisfied** |
| Stop and preserve terminal evidence | Frozen nonwritable run tree plus exact local receipt mirror | **Satisfied** |

## Terminal state

Root job `750976` failed on CPU node `evc21` after one second with exit
`2:0` and the exact message:

```text
pcf1: SLURM_TMPDIR is required for offline caches
```

The job had no GPU allocation and stopped before environment capture, source
preparation, assessor creation, model loading, or scoring. Every exact
downstream job ID `750977--751004` is `CANCELLED`, has elapsed time zero, no
assigned node, and zero restarts. The user queue is empty.

The remote run tree contains only the two dispatch receipts, prepare stdout
and stderr, and the terminal infrastructure receipt. It has no scientific or
custody output directory and no writable member. The terminal receipt SHA-256
is
`366ebd73e13d1f944b1a233bf86c87440a23295ecdc4caa4b045462a8d3dbef0`;
the byte-identical local mirror is
`SHOHIN_PCF1_TERMINAL_INFRASTRUCTURE_RECEIPT_20260811.json`.

Storage and sandbox admission remain qualified. Current Lustre use is
`840,987,128 KiB / 780,218` inodes against hard limits
`1,059,061,760 KiB / 1,010,000`, leaving `218,074,632 KiB / 229,782`
of headroom.

## Overall verdict

The publication objective is **not scientifically complete**. Exactly-one
execution and terminal preservation are proven, but the source-disjoint
matched-arm confirmation and its falsifiable PASS/FAIL gate were never
evaluated. The frozen contract independently states that an infrastructure
failure is terminal and does not authorize a shard replay, retry, requeue, or
successor. Therefore no remaining in-scope action can produce the missing
scientific result without violating the objective's own no-prohibited-retry
requirement. The prior Qwen3.5-9B `383/538` versus unchanged `316/538` claim
remains the strongest qualified result, but it remains prospectively
unconfirmed by PCF1.
