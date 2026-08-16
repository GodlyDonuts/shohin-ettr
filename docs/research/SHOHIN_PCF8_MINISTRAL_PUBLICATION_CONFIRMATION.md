# PCF8: Ministral Publication Confirmation Successor

Status: closed terminal-null on 2026-08-11.

Terminal update: preparation, no-score mechanics, and B1 training completed.
Draft task `751823_3` then failed on `evc33` before generator entry because
`nvidia-smi` reported no devices despite one allocated H100 GRES. No draft
file or partial was published. All downstream jobs were canceled, the
confirmation board was not semantically read, and no score or scientific
PASS/FAIL was emitted. Preserve
`SHOHIN_PCF8_TERMINAL_INFRASTRUCTURE_RECEIPT_20260811.json`.

Admission update: the required no-model H100 scratch qualification completed
as job `751818` on `evc23` in four seconds. It reported 183,262,756,864 bytes
and 357,942,594 inodes available, verified the H100 identity and the existing
write/rename/fsync boundary, and completely deleted a nested read-only mock
model tree. The exact scratch path is absent. Model, data, and assessor access
were all false. Preserve `SHOHIN_PCF8_SCRATCH_QUALIFICATION_20260811.json`;
its remote receipt SHA-256 is `2cb17ef5...cb17`.

PCF1 through PCF7 remain closed with formal scientific result `null`. PCF7
passed preparation and its complete no-score mechanics payload: the pinned
model loaded, one ephemeral optimizer update completed, all 24 mechanics rows
passed, and the immutable mechanics report says `capability_scored=false`.
The allocation then failed only during its EXIT cleanup because the staged
model retained read-only source modes and `shutil.rmtree` could not unlink it.
B1 and all capability stages never started. The exact node-local tree was
subsequently validated and permanently deleted by cleanup job `751816`; all
durable evidence was frozen. Preserve
`SHOHIN_PCF7_TERMINAL_INFRASTRUCTURE_RECEIPT_20260811.json`.

PCF8 is the separately named, prospectively declared infrastructure successor
under the user's standing authorization. It inherits every scientific and
security clause of PCF1 through PCF7. The host, model and tokenizer semantics,
source bytes, split, prompts, arms, seeds, training geometry, candidate policy,
thresholds, assessor custody, and terminal gate remain unchanged. In
particular, the observed Transformers tokenizer-regex warning is preserved as
a disclosed risk; PCF8 does not change tokenizer flags or bytes.

Its sole repair is bounded scratch deletion. After validating the exact
private direct-child scratch root, cleanup walks only its same-owner,
same-device, nonsymlink directories without following links and restores owner
write/search permission before deletion. No durable model/source tree is ever
modified. Before a graph, run one no-model H100 scratch canary that creates a
nested read-only mock model tree and proves complete cleanup plus the existing
128-GiB/150,000-inode and write/rename/fsync gates.

If that canary passes, preserve its receipt, replay the full live preflight,
and submit one PCF8 graph. Disable requeue, retries, and automatic successors.
Any infrastructure failure, formal `PASS`, or formal `FAIL` ends PCF8. Do not
change or open an alternate host, tokenizer, data split, prompt, arm, seed,
threshold, or protected board.
