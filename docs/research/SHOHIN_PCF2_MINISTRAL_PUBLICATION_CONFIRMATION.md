# PCF2: Ministral Publication Confirmation Successor

Status: prospectively frozen on 2026-08-11 before any PCF2 remote mutation,
model load, source materialization, or scientific compute.

## Authorization and relation to PCF1

PCF1 remains permanently closed with formal result `null`. Its only started
allocation was CPU job `750976`, which failed after one second because Newton
does not set `SLURM_TMPDIR`. It performed no model load, H100 work, source
materialization, assessor read, candidate generation, or scientific score.
Jobs `750977--751004` never started. The immutable failure receipt is
`SHOHIN_PCF1_TERMINAL_INFRASTRUCTURE_RECEIPT_20260811.json`.

After that terminal record was preserved and pushed, the user explicitly
authorized a separately named successor in pursuit of Shohin. This document
is that authorization boundary. PCF2 is not a replay or relabeling of PCF1:
it is a new, independently frozen execution protocol whose scientific
contract is byte-for-byte identical to PCF1 and whose only permitted change
is the repair of allocation-local scratch orchestration described below.

## Immutable scientific contract

PCF2 inherits without alteration every scientific clause of
`SHOHIN_PCF1_MINISTRAL_PUBLICATION_CONFIRMATION.md`, including:

- host `mistralai/Ministral-3-8B-Reasoning-2512` at revision
  `81eaece1948f3875421d9a45bc55487d10e2d894` and the explicit multimodal
  loader;
- the exact B1, pair, math, logic/science, and MBPP source hashes;
- source split `2026080811`, draft seed `2026080818`, revision seeds
  `2026080815/2026080814`, evaluation seed `2026080816`, and commit seeds
  `2026080820/2026080822`;
- the same model-owned draft, trained same-family revision, unchanged and
  self-refinement controls, and learned whole-trajectory commit;
- the same updates, batches, accumulation, ranks, layers, learning rates,
  contexts, generation lengths, four-arm semantics, and 1,289-row board;
- the same sealed-data firewall and exactly one authorized CPU assessor open;
- no access to the holdout, protected product, or public data; and
- no reopening of NDR1, KCR1, VTE1, the natural-language microcode bridge,
  the Q35 edit cascade, or small-OLMoE variants.

The sole falsifiable gate is unchanged:

1. unchanged `>=387/1289` and every domain nonzero;
2. revision `>= unchanged+65` and `>= self-refinement+39`, with nonnegative
   per-domain deltas against both controls;
3. learned commit `>= revision+13`, retains at least 95% of both the
   revision-correct and unchanged-correct identities, and has nonnegative
   per-domain deltas against revision; and
4. exact `1289/1289` custody and order, zero assessment truncation, zero
   malformed selections, complete model/data/runtime/environment/sandbox/
   checkpoint/compute hashes, zero retries, and zero holdout/public/product
   access.

`PASS` remains the conjunction of every clause. Any scientifically false
clause is the final PCF2 `FAIL`. Either `PASS` or `FAIL` stops the phase.

## Sole infrastructure repair

Newton advertises node-local `/tmp` but does not populate `SLURM_TMPDIR`.
Every PCF2 allocation therefore creates exactly one deterministic private
directory `/tmp/pcf1-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID|scalar}`. Before
use, the runtime must prove:

- `/tmp` resolves exactly, is root-owned mode `1777`, and is not a symlink;
- the target did not exist and is created owner-only mode `0700` on the same
  filesystem;
- at least 128 GiB and 150,000 inodes are available;
- an atomic write, fsync, rename, and delete succeeds; and
- the exact owned target is deleted and its parent fsynced on success,
  failure, signal, and evidence-preservation exits.

Ambient `SLURM_TMPDIR` is rejected. Cleanup validates the exact job/task path,
owner, type, device, and resolved parent before recursive removal; no variable,
glob, home root, project root, or Lustre path may be a cleanup target.

Before PCF2 submission, one CPU allocation and one eligible H100 allocation
must run `pcf2_scratch_canary.sbatch`. Each is infrastructure-only: it may
test scratch capacity/lifecycle and H100 identity, but must not receive or
open a model, source bank, assessor, candidate, or scientific output. Both
immutable receipts and independent post-job absence checks are required.

## Execution and stop rule

After local tests, immutable runtime packaging, storage re-verification,
sandbox re-verification, and both scratch canaries pass, submit exactly one
fresh dependency-bound PCF2 graph. Disable Slurm requeue and automatic
successors. Preserve any infrastructure failure as terminal evidence; do not
convert it into a wrong answer. If the graph reaches scoring, open the sealed
confirmation assessor exactly once, write the formal gate, mirror its hashes,
and stop. No automatic retry, alternate host, protected split, or successor is
authorized after PCF2.
