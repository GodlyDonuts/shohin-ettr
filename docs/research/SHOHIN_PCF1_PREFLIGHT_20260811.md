# PCF1 preflight and infrastructure qualification — 2026-08-11

This receipt preserves the repository, host, scheduler, model, and baseline
storage state observed before any PCF1 source materialization, model load, or
job submission, followed by the authorized storage remediation and Newton
CPU-only sandbox qualification. The frozen scientific contract is
`SHOHIN_PCF1_MINISTRAL_PUBLICATION_CONFIRMATION.md`. No PCF1 scientific,
model, or H100 job has run, and no confirmation, holdout, public, or product
assessor has been opened.

## Repository and remotes

- private architecture worktree pre-mutation `HEAD`:
  `c3fb091d2d28e658715311c10668a3c0ef04fc98`;
- `origin/main` at the same commit, divergence `+0/-0`;
- private fetch/push remote: `https://github.com/GodlyDonuts/shohin-ettr.git`;
- public fetch remote: `https://github.com/GodlyDonuts/shohin.git`;
- public push URL: `DISABLED_PUBLIC_REPO_DO_NOT_PUSH`;
- publication work branch:
  `codex/pcf1-ministral-publication-confirmation`;
- public worktree preflight `HEAD`:
  `1065ef98c2b29ea6adcb37b47f57abf07a3a6f78`.

No closed-lane artifact or branch was modified or used as a retry.

## Qualified release anchor

The immutable positive-control release remains present in Newton's campaign
artifact namespace as `product_reasoning/idr_aqc_release_8f0bd8d_r1` (about
80 MiB).
Its complete `SHA256SUMS` verification passed read-only. The independently
checked anchors are:

- Qwen3.5-9B revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`;
- release manifest SHA-256
  `554e841f71edd3a19063411348340e337532db2db05dd5e1e2adc25a3d347e7b`;
- release `SHA256SUMS` SHA-256
  `0dad031312dec0859e35bb7e9daea8aef688ef350b9053f587fba5acdc9c58c5`;
- product report SHA-256
  `3e86751bb234ee29465885206da5316890060ad8b0b88ea752c4fb012bbf7187`;
- draft adapter SHA-256
  `854a7cc44fbc2b54418f4e5bd09b7efeed0da44fc9ce217b0bb6b1997b722971`;
- trained revision SHA-256
  `df3c264d426941fef8ba9c10a90fe9fab304ec2864738209a4d79f9f81e0c473`;
- learned commit SHA-256
  `434d1ec0a8e05d49ee8cc6eaaba1ad36657f507d7416a7417931096c23d2aabc`.

The bound product report is exactly unchanged `316/538`, trained revision
`374/538`, and learned coherent commit `383/538`; PCF1 does not reopen or
rescore it.

## Newton scheduler and account

The read-only audit at 2026-08-11 00:51 EDT found:

- no queued or running jobs for `sa305415`;
- `normal` has 29 two-H100-PCIe nodes: 12 idle, 13 mixed, and 4
  drained;
- `evc26`, `evc31`, `evc32`, and `evc38` are drained for full Slurm spool;
- `evc29` remains excluded for full local temporary storage and `evc46` for
  its prior unusable-CUDA observation;
- every PCF1 job is restricted to `normal` and excludes
  `evc26,evc29,evc31,evc32,evc38,evc46`;
- August accounting was approximately 695 GPU-hours against the documented
  2,000-hour cap, leaving about 1,305 nominal GPU-hours.

## Dense host

The sole candidate is the cached
`mistralai/Ministral-3-8B-Reasoning-2512` snapshot at revision
`81eaece1948f3875421d9a45bc55487d10e2d894`. Its read-only configuration
declares `model_type=mistral3`, architecture
`Mistral3ForConditionalGeneration`, nested 34-layer/4,096-width
`Ministral3Config` text, and Pixtral vision configuration. The pinned campaign
environment reports Transformers `5.15.0.dev0`; offline configuration and
auto-class lookup recognize `AutoModelForMultimodalLM` but not
`AutoModelForCausalLM` for the outer config. PCF1 therefore freezes the
explicit `multimodal` loader. Full weight load, scoped-LoRA trainability,
serialization, generation, and swapped-order mechanics remain unqualified
until the single 24-presentation no-score H100 admission job passes.

The read-only audit located the immutable download root at
`/lustre/fs1/home/sa305415/shohin/artifacts/external/ministral-3-8b-reasoning-2512-81eaece`.
Completed CPU download job `747023` created that root and a pre-existing
58-entry `SHA256SUMS`; every directory is mode `0555`, all files are
nonwritable, and there are no symlinks.
The complete manifest verified read-only and covers 35,706,515,534 bytes. Its
fixed anchors are:

- `SHA256SUMS` SHA-256
  `46cc9203a18a414e08a53109662c3802b57c046896185ca9ab31875e8167cf1f`;
- `SOURCE_REVISION` SHA-256
  `3576c1bfaa0652940d12817ad3267ffe65645dc558ceb9a153ffb72f7211a982`,
  whose content is the exact 40-hex revision above;
- `config.json` SHA-256
  `5aae04beb9f2a9949eb1df870cf47ba292012a066bdcdcb115a9ac43425f8086`.

Preparation must verify these existing anchors and every manifest entry. It
must not replace them with a newly self-declared revision or manifest.

The pinned environment is
`/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2`.
Its immutable `runtime.json` SHA-256 is
`277b97fbd6b18760c9789cf3f3372bdb6b40ca87bf84a1df4b41ee3194c4e9dd`
and reports Python `3.13.13`, Torch `2.6.0+cu124`, and Transformers
`5.15.0.dev0`. Its package freeze SHA-256 is
`1d4dfd4a1dc11af9788b0bab072d262278db1814d3fca49465d4df5931b3b87a`;
the resolved Python 3.13 binary SHA-256 is
`051a031d827eab9778e982571db754662809164c8a3ec01e9beea1e1088123e0`.
The no-score mechanics allocation must reproduce and bind an external
environment receipt before any scientific output.

The initial audit also found `/usr/bin/bwrap` 0.4.0, SHA-256
`eb767688b8224d8d3dbe1f8cb30ac3dff9ae8b02ff0452eaec9f94874d4e0011`.
A read-only namespace probe showed that `--unshare-all` is permitted: a
Python 3.13 child saw only its anonymous candidate, read-only runtime,
private `/proc` and `/dev`, and temporary filesystem; the parent process was
absent and an external network connection returned `ENETUNREACH`. This is
the initial preflight observation; the complete independent qualification is
recorded below. The frozen mechanics and assessment path retains its exact
fail-closed sandbox and probe custody.

## Newton sandbox qualification

The final Newton CPU-only qualification independently reports `status=pass`
and `sandbox_isolation_passed=true`. It binds:

- sandbox source SHA-256
  `7b1eb83fb5546fd3c782cccef9a3254b90657b36cc90c023184136a6ed196523`;
- immutable qualification receipt
  `SHOHIN_PCF1_SANDBOX_QUALIFICATION_20260811.json`, file SHA-256
  `f1423aaed0d4b764f81f48a0289d4122b755955f9d961db50b45f485130df070`;
- sandbox config SHA-256
  `4e3aaf268e3d16ba900b467c543ac074c9c738f5dee05d0d8b22f0366ae99a33`;
- candidate-policy SHA-256
  `f27124db3d134a1e3dbde06958ab03220cd5e9585abcc356baa6a49d9edd1f1e`;
- Python-runtime-descriptor SHA-256
  `025190cde6346cdbebfc04a06650f4813e2e8ead5350eec55c0b460caabb362f`;
- probe-set SHA-256
  `4d7bd7f009c802aff22a4a7550212e0b481b78c5fd10a86e1f202f6a90d160b4`.

All `40/40` adversarial results are true. They cover filesystem and
environment escape, parent `/proc`, network, subprocess/fork, symlink and
path traversal, entropy and hash determinism, resource and timeout abuse,
candidate/read-only input mutation, safe-import namespace reachability,
direct PID 1, trusted completion, status/exit-code forgery, and correct
separation of infrastructure, candidate, setup, and test failures. The
qualified implementation uses a raw candidate-only sealed memfd, direct PID 1
bubblewrap with no network and a private `/proc`, `/dev`, and `/tmp`, a
scrubbed environment, a read-only minimal Python/ELF closure, and the exact
pinned libc `memfd_create` ABI. Infrastructure failure never becomes a wrong
answer.

Qualification was intentionally fail-closed while exact Newton host facts
were learned. Earlier admissions stopped on, in order, a one-nibble libc hash
pin error, the pinned Python build's absent `os.memfd_create`/seal constants,
a UTF-8 runtime-descriptor mismatch, direct-PID-1 CPU-limit exit `137`, and
root-writability/safe-import probe failures. Each was infrastructure evidence,
emitted no qualification receipt, and ran no model, score-bearing scientific,
or H100 job. Their transcripts were preserved; none is a scientific attempt
or a wrong answer. The final PASS was separately reviewed against the exact
source and receipt bytes.

## Durable-storage qualification

The baseline Newton audit reported user Lustre usage of
`1,700,584,688 KiB` against a
`1,059,061,760 KiB` hard limit: `641,522,928 KiB` (about 611.8 GiB) over.
It also reported `2,336,246` files against a `1,010,000` hard limit:
1,326,246 inodes over.

Read-only follow-up initially found no safe bypass:

- Stokes shares the same `/lustre/fs1` user accounting;
- the writable Skattel group directory is charged to that same user quota;
- `/lustre/fs2` has capacity but no allocated or accessible user path;
- observed rstore space is ACL-inaccessible;
- login-node `/tmp` is ephemeral and non-shared;
- no configured durable VPS target exists;
- the local Mac has about 54 GiB free, enough for compact evidence mirroring
  but not a safe cluster execution namespace.

The user then authorized an exact provenance audit followed by age-ordered,
bounded removal of old Shohin cache, staging, failed/closed-lane output, raw
artifact, and duplicate-runtime targets, while preserving the private
repository, the complete qualified 9B release, the pinned Ministral model and
runtime, all current PCF1 material, credentials/configuration, and unrelated
work. Every destructive batch resolved literal absolute paths, checked the
preserve boundary, recorded age/size/inodes, quarantined within the same
filesystem, deleted only the exact quarantine, and remeasured quota. The full
manifest and recovery accounting are in
`SHOHIN_PCF1_STORAGE_RECLAMATION_20260811.md`.

Three settled post-cleanup observations were identical:
`838,918,136 KiB / 768,478` in use. This is below both hard limits and leaves
`220,143,624 KiB / 241,522` of headroom, exceeding the frozen
128-GiB/150,000-inode PCF1 admission target by
`85,925,896 KiB / 91,522`. An independent read-only postcheck reproduced the
quota and empty scheduler state, found all 20 protected anchors, found no
quarantine remnant, and reproduced the 402-record deletion-transcript SHA-256
`a3a58a160206a0f676cf2faad18234ef7dbb448871651bb4a097cdd1c68b907a`.
The storage gate therefore passes and cleanup stopped without selecting
another target.

The ledger's cache/duplicate removals are reproducible from their named
sources where recorded. Its user-authorized old closed-lane raw artifacts and
other unique historical removals were permanently deleted after quarantine;
they are not locally recoverable, and only their compact git conclusions and
hashes remain. No protected PCF1, pinned-host, qualified-release, credential,
or unrelated-workspace anchor was removed.

## Live deployment admission chronology

The first packaged-runtime live preflight stopped before creating the run
root. Its hash-only frozen-input verifier exposed only `pipeline/` on
`PYTHONPATH`, while the hardened input module now imports the packaged
code-sandbox module from `train/`. Python raised `ModuleNotFoundError` before
any model load, H100 allocation, assessor open, data materialization, or
scientific job submission. This is an infrastructure admission failure, not a
PCF1 result or retry. The dispatcher now exposes both immutable packaged
roots for that verifier and directly regression-tests the exact path. The same
narrow correction also raises the executable preflight minimum from the old
96-GiB/100,000-inode floor to the frozen 128-GiB/150,000-inode admission
target. A fresh runtime, checkout, and complete preflight are required before
the sole graph may be submitted.

## Sole scientific gate and stop

The only score-bearing decision remains the frozen 1,289-row conjunctive
PCF1 gate:

1. unchanged continuation is at least `387/1289` and solves at least one
   math, one logic/science, and one executable-code identity;
2. trained revision is at least 65 identities above unchanged and at least 39
   above self-refinement, with nonnegative per-domain deltas against both;
3. coherent commit is at least 13 identities above revision, retains at least
   95% of revision-correct and at least 95% of unchanged-correct identities,
   and has nonnegative per-domain deltas against revision;
4. every arm covers the exact same `1289/1289` identities in order, candidate
   assessment truncation and malformed selections are zero, every frozen
   hash/runtime/checkpoint/result/accounting item verifies, and
   holdout/public/product access counters remain zero.

Exactly one atomic assessor open returns formal PASS only if every conjunct
is true; any false conjunct returns formal FAIL. Either result is terminal and
does not authorize a changed host, data, arm, prompt, threshold, seed,
schedule, selector, decoding setup, retry, protected split, or successor.
Infrastructure failure is preserved as terminal infrastructure evidence, is
never scored as a wrong answer, and authorizes no replay or retry.

## Authorization state

- local contract/runtime/test implementation: authorized;
- durable storage: independently qualified with substantial byte and inode
  headroom;
- Newton code sandbox: independently qualified, receipt bound, `40/40` true;
- remote data preparation: authorized only as the frozen PCF1 graph;
- no-score mechanics: not submitted;
- scientific dependency graph: not submitted;
- scientific/model/H100 work: none executed;
- confirmation labels/product/public/holdout: not opened by PCF1;
- retry, alternate host, and successor: not authorized.
