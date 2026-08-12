# PCF16: Ministral Publication Confirmation Successor

Status: prospectively frozen on 2026-08-12 before admission or scientific
compute.

PCF15 is terminal-null. Preparation, mechanics, B1, all 16 source-only draft
shards, exact draft merge, materialization, and the frozen 256-update revision
training completed with zero restarts. All eight calibration allocations then
ran beyond PCF14's earlier failure point. Revision-calibration shard 3 stopped
after `768/1456` rows on `evc28` because the isolated Python bootstrap did not
emit trusted READY. The seven peer allocations and every descendant were
cancelled. Commit training and confirmation generation never started; the
prepared confirmation assessor board has zero semantic reads; no score
authorization, score, normalized report, compute custody, final comparison,
or formal scientific result exists. Preserve
`SHOHIN_PCF15_TERMINAL_INFRASTRUCTURE_RECEIPT_20260812.json` (SHA-256
`a252dfc7...ff8e`) and the frozen remote PCF15 run. Do not replay or continue
PCF15.

PCF15 removed Python `preexec_fn`, but its `subprocess` call retained
`pass_fds`, which forces Python away from its safe `posix_spawn` fast path and
through a fork/exec implementation in the resident multithreaded 9B
PyTorch/CUDA process. PCF16 removes that last parent-process fork boundary.
It invokes exact `/usr/bin/prlimit` through `os.posix_spawn` with five explicit
`POSIX_SPAWN_DUP2` actions: assessor stdin, bounded stdout/stderr, sealed raw
candidate, and Bubblewrap namespace-info FD. Child candidate/info FDs are
chosen deterministically in `[32,63]`; every other Python-opened descriptor is
close-on-exec. The spawned process receives an empty environment, and the
existing wall-time monitor waits and fail-closes exactly as before.

The exact resource chain remains `prlimit -> bubblewrap -> pinned minimal
Python`: CPU `3:4` seconds, address space `1 GiB:1 GiB`, file size `1 MiB:1
MiB`, private PID/proc/dev/tmp, no network, read-only root/runtime/candidate,
sealed assessor transport closed before trusted READY, bootstrap exit
attestation, and infrastructure/scientific termination separation. Bootstrap
SHA-256 remains `726af7a5...a60`; generated-candidate policy remains
`f27124db...f1e`; the three-second scientific candidate limit is unchanged.
The prospective sandbox source SHA-256 is `448730d6...cf2d`, and its derived
config SHA-256 is `d8f2060a...3733`.

Before a PCF16 graph may exist, all of these infrastructure-only gates must
pass on fresh Newton allocations and be frozen with exact source/runtime/model
hashes and zero protected access:

1. one CPU allocation runs the complete unmocked sandbox qualification with
   all 41 probes true using the `os.posix_spawn` production path;
2. two simultaneous H100 allocations are co-located on one eligible two-GPU
   dense node, each loads the exact pinned Ministral host plus the frozen PCF15
   revision adapter and repeatedly performs greedy generation followed by the
   exact late-failure calibration MBPP assessment;
3. each H100 allocation completes at least 1,000 generation-plus-sandbox
   iterations with zero infrastructure failures, zero restarts, exact
   environment/model/adapter/sandbox hashes, and no confirmation access;
4. an independent postcheck verifies the queue empty, private scratch absent,
   and at least 128 GiB plus 150,000 inodes of durable headroom.

Any qualification divergence closes PCF16 before science. A passing
qualification authorizes exactly one fresh complete graph at a new immutable
root. PCF16 may not reuse any PCF14/PCF15 source view, B1 adapter, draft,
revision adapter, calibration output, or partial artifact in that graph.

The pinned dense host and tokenizer, source banks, split and identity order,
prompts, unchanged/revision/self-refinement/commit arms, seeds, 256-update
training geometry, generation settings, generated-candidate policy, sandbox
visibility boundary, three-second candidate limit, custody rules, sealed
board, and sole 1,289-row conjunctive gate remain byte-for-byte unchanged.
Requeue, retries, partial continuation, automatic successors, and protected
split access are disabled.

The sole falsifiable gate remains:

- unchanged reaches at least `387/1289` with every domain nonzero;
- trained revision exceeds unchanged by at least `65` correct and
  self-refinement by at least `39`, with no per-domain loss against either;
- learned whole-trajectory commit exceeds revision by at least `13`, retains
  at least 95% of both revision-correct and unchanged-correct identity sets,
  and loses no domain against revision;
- all four arms cover the exact 1,289 identities in frozen order, with zero
  truncation, zero malformed selections, exact runtime/model/data/checkpoint/
  scheduler custody, one consumed assessor authorization, and zero
  holdout/public/product access.

Any infrastructure failure, formal `PASS`, or formal `FAIL` ends PCF16. A
formal result must be preserved exactly and authorizes no successor.
