# PCF10: Ministral Publication Confirmation Successor

Status: prospectively frozen on 2026-08-11 before PCF10 admission or compute.

PCF9 is closed with formal scientific result `null`. Preparation completed.
The mechanics allocation on `evc27` had one visible H100 and completed its
ephemeral update, then the separate mechanics process rejected the valid
compute-host receipt before generation. The receipt correctly listed the
seven-node scheduler exclusion set including `evc33`; one duplicated Python
validator constant still required the former six-node list. B1 and all
downstream stages never started, the assessor was not semantically read, and
no score was emitted. Preserve
`SHOHIN_PCF9_TERMINAL_INFRASTRUCTURE_RECEIPT_20260811.json`.

PCF10 is the separately named infrastructure successor under the user's
standing authorization. Its sole repair synchronizes the already-frozen
seven-node exclusion set in the mechanics receipt validator and the final
Slurm-accounting custodian. The dispatcher, job headers, common runtime, and
compute-host receipt already use those exact seven nodes. Tests must exercise
the exact receipt and accounting boundary before packaging.

The pinned host/model/tokenizer behavior, sources, split, prompts, arms,
seeds, training geometry, candidate policy, sandbox, custody, thresholds, and
sole 1,289-row terminal gate remain unchanged. PCF10 starts from a fresh root,
replays the complete graph, and reuses no prior checkpoint or partial output.
Disable requeue, retries, and automatic successors. Any infrastructure
failure, formal `PASS`, or formal `FAIL` ends PCF10.
