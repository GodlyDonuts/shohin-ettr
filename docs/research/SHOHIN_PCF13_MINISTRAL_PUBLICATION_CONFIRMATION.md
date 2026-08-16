# PCF13: Ministral Publication Confirmation Successor

Status: prospectively frozen on 2026-08-11 before PCF13 admission or compute.

PCF12 is closed with formal scientific result `null`. Preparation, the freshly
qualified mechanics gate, and B1 completed with zero restarts. Draft tasks 4
and 5 then stopped after two seconds on `evc37` because its allocation-local
scratch did not meet the frozen 128-GiB capacity floor. The other 14 tasks were
cancelled; no complete draft shard was published. Revision, calibration,
confirmation, assessor semantic read, and scoring never started. Preserve
`SHOHIN_PCF12_TERMINAL_INFRASTRUCTURE_RECEIPT_20260811.json`.

PCF13 is the separately named infrastructure successor under the user's
standing authorization. Its sole repair adds `evc37` to the exact scheduler,
job-header, compute-host, accounting, and custody exclusion set. Tests must
prove those copies agree before packaging. No partial PCF12 artifact may be
reused.

The pinned host/model/tokenizer behavior, sources, split, prompts, arms, seeds,
training geometry, candidate policy/timeouts, sandbox, custody, thresholds,
and sole 1,289-row terminal gate remain unchanged. PCF13 starts from a fresh
root and replays the complete graph. Disable requeue, retries, and automatic
successors. Any infrastructure failure, formal `PASS`, or formal `FAIL` ends
PCF13.
