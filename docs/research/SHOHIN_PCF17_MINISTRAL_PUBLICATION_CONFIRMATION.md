# PCF17: Ministral Publication Confirmation Successor

Status: closed terminal-null on 2026-08-12. The sole graph completed every
scientific generation stage, then precompute custody failed before score
authorization or assessor open. Formal result is `null`; see
`SHOHIN_PCF17_TERMINAL_INFRASTRUCTURE_RECEIPT_20260812.json`. This terminal
status does not alter the frozen contract below.

PCF16 closed during infrastructure qualification and never submitted a
publication graph. Its unmocked CPU allocation passed all 41 sandbox probes.
Two simultaneous H100 workers on `evc28` then loaded the exact dense host and
PCF15 revision adapter and each completed 27 worst-case
generation-plus-sandbox cycles. Both stopped in lockstep on cycle 28 because
the separate outer launcher wall watchdog expired. Jobs `752844` and `752845`
were `FAILED 1:0`, with zero restarts; scratch teardown completed, the queue is
empty, protected access is zero, and no formal scientific result exists.
Preserve `SHOHIN_PCF16_TERMINAL_QUALIFICATION_RECEIPT_20260812.json`
(SHA-256 `0f11c1a3...cec3`) and the frozen qualification root. Do not reuse its
receipt as admission evidence.

PCF17 changes one infrastructure constant only. The candidate's exact CPU
soft/hard limits remain `3:4` seconds, address-space limit remains 1 GiB,
file-size limit remains 1 MiB, and candidate wall parameter remains 3 seconds.
The parent-side launcher watchdog grace increases from 2 to 30 seconds. This
grace covers host scheduling plus `posix_spawn -> prlimit -> Bubblewrap`
namespace setup/teardown; it does not grant candidate CPU time, alter trusted
termination codes, or turn a timeout into a wrong answer. Any expiry remains a
terminal `PCF1SandboxError` infrastructure event.

The explicit `os.posix_spawn` FD projection, empty environment and signal
mask, exact reset signals, pinned `/usr/bin/prlimit`, Bubblewrap namespace,
minimal runtime/ELF closure, sealed candidate/assessor transport, candidate
policy, and bootstrap remain unchanged. Bootstrap SHA-256 remains
`726af7a5...a60`; generated-candidate policy remains `f27124db...f1e`. The
prospective sandbox source SHA-256 is `7e97c28d...f1c4`, and derived config
SHA-256 is `70a37245...1802`.

Before a PCF17 graph may exist, fresh Newton evidence must prove:

1. one CPU allocation passes all 41 unmocked sandbox probes using the exact
   new source/config and zero protected access;
2. two simultaneous H100 allocations on one eligible two-GPU dense node each
   load the exact pinned host plus frozen PCF15 revision adapter and complete
   1,000 greedy-generation-plus-sandbox cycles on exact calibration MBPP
   identity `e23c78d9...a8dc`;
3. both workers have zero infrastructure failures, zero restarts, exact
   model/adapter/sandbox hashes, and no confirmation access;
4. exact postchecks prove queue empty, scratch absent, and durable headroom of
   at least 128 GiB plus 150,000 inodes.

Any qualification divergence closes PCF17 before science. A passing
qualification authorizes exactly one complete graph at a fresh root, with no
reuse of any PCF14/PCF15 artifact or PCF16 qualification state.

The pinned dense host/tokenizer, source banks, identity order, prompts, arms,
seeds, 256-update training geometry, generation settings, candidate CPU
semantics, candidate policy, custody, sealed board, thresholds, and sole
1,289-row gate remain byte-for-byte unchanged:

- unchanged reaches at least `387/1289` with every domain nonzero;
- revision exceeds unchanged by at least `65` and self-refinement by at least
  `39`, with no per-domain loss;
- learned commit exceeds revision by at least `13`, retains at least 95% of
  revision-correct and unchanged-correct identities, and loses no domain;
- all arms have exact ordered coverage, zero truncation/malformed evidence,
  complete custody, one assessor open, and zero holdout/public/product access.

Requeue, retries, partial continuation, automatic successors, and protected
split access are disabled. Infrastructure failure, formal `PASS`, or formal
`FAIL` ends PCF17; a formal result authorizes no successor.
