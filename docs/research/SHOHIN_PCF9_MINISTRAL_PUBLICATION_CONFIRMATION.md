# PCF9: Ministral Publication Confirmation Successor

Status: closed terminal-null on 2026-08-11.

Terminal update: preparation completed. Mechanics had one visible H100 and
completed its ephemeral update, then rejected its valid seven-node
compute-host receipt because a duplicated Python validator still required the
former six-node exclusion list. This occurred before mechanics generation,
B1, candidates, assessor read, or scoring. Preserve
`SHOHIN_PCF9_TERMINAL_INFRASTRUCTURE_RECEIPT_20260811.json`.

PCF8 is closed with formal scientific result `null`. Preparation, no-score
mechanics, and the frozen B1 training completed. Draft task `751823_3`
received one Slurm H100 GRES on `evc33`, but its first `nvidia-smi` invocation
reported `No devices were found` and exited 6 before entering the generator.
No draft file or partial was published. The other 15 tasks and every downstream
stage were canceled once success was impossible. Confirmation assessors were
never semantically read and no score was emitted. Preserve
`SHOHIN_PCF8_TERMINAL_INFRASTRUCTURE_RECEIPT_20260811.json`.

PCF9 is the separately named infrastructure successor under the user's
standing authorization. Its sole change is to add proven-broken node `evc33`
to the existing scheduler exclusion set. The pinned host family, model,
tokenizer behavior, source bytes, split, prompts, arms, seeds, training
geometry, candidate policy, sandbox, thresholds, assessor custody, and final
gate are unchanged. PCF9 starts from a fresh run root and executes the entire
graph; it does not reuse the PCF8 B1 checkpoint or any partial output.

The successful H100 allocations in PCF8 established device visibility on the
remaining nodes used by the draft array, while the PCF8 mechanics and scratch
cleanup passed exactly. Replay the full live preflight, require the queue empty
and at least 128 GiB/150,000 inodes of durable headroom, disable requeue and
retry, and submit exactly one PCF9 graph. Any infrastructure failure, formal
`PASS`, or formal `FAIL` ends PCF9. No automatic successor is authorized.
