# PCF12: Ministral Publication Confirmation Successor

Status: prospectively frozen on 2026-08-11 before PCF12 admission or compute.

PCF11 is closed with formal scientific result `null`. Preparation, mechanics,
B1, all 16 draft shards, ordered draft merge, repaired materialization, and
revision training completed with zero restarts. The two matched calibration
arrays then began. Before any calibration shard could publish candidates, two
unchanged-arm tasks colocated on `evc40` failed their non-scientific sandbox
resource-limit admission probe: the trusted one-second CPU limit did not emit
SIGXCPU within its 2.5-second outer wall allowance under contention. The other
six tasks were cancelled. Confirmation generation, the sole assessor semantic
read, and scoring never started. Preserve
`SHOHIN_PCF11_TERMINAL_INFRASTRUCTURE_RECEIPT_20260811.json`.

PCF12 is the separately named infrastructure successor under the user's
standing authorization. Its sole repair changes the CPU-limit admission probe
from a 0.5-second requested timeout (one-second CPU soft limit, 2.5-second outer
wall) to a two-second requested timeout (two-second CPU soft limit, four-second
outer wall). The probe must still terminate through reserved trusted exit 79
and classify `candidate_resource_limit`; outer timeout remains infrastructure.
Candidate assessment timeouts and every sandbox isolation/policy boundary are
unchanged. A fresh Newton qualification must pass before submission.

The pinned host/model/tokenizer behavior, sources, split, prompts, arms, seeds,
training geometry, generated-candidate policy, scientific sandbox behavior,
custody, thresholds, and sole 1,289-row terminal gate remain unchanged. PCF12
starts from a fresh root, replays the complete graph, and reuses no PCF11
checkpoint, draft, or partial output. Disable requeue, retries, and automatic
successors. Any infrastructure failure, formal `PASS`, or formal `FAIL` ends
PCF12.
