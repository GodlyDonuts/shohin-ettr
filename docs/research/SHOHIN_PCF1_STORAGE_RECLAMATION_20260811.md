# PCF1 storage reclamation receipt — 2026-08-11

Status: authorized by the user, bounded to exact demonstrably reproducible or
failed/partial targets, and in progress before any PCF1 remote data write or
job submission.

## Immutable preserve boundary

The cleanup must not remove or mutate:

- any unique checkpoint, adapter, qualified release, immutable report,
  manifest, scheduler/accounting receipt, source bank, protected split,
  dataset required by the phase handoff, or PCF1 model/runtime anchor;
- either Shohin repository or an unrelated project/user tree;
- the pinned Ministral snapshot at
  `/lustre/fs1/home/sa305415/shohin/artifacts/external/ministral-3-8b-reasoning-2512-81eaece`;
- the pinned `product-reasoning-b3a3603-r2` environment or the qualified Qwen
  release/runtime roots;
- FineWeb-Edu, peS2o, or another physical source holding unless a verified
  external mirror is first recorded.

## Baseline

Read-only Lustre quota at `2026-08-11T02:54:30-04:00`:

- used: `1,700,584,688 KiB` and `2,336,246` files;
- hard limit: `1,059,061,760 KiB` and `1,010,000` files;
- overage: `641,522,928 KiB` and `1,326,246` files;
- queued/running user jobs: zero.

## Candidate audit

The first candidate is the exact cache root
`/lustre/fs1/home/sa305415/.cache/bazel`. It contains two Bazel-generated
output bases whose own `README` files bind them to
`/lustre/fs1/home/sa305415/tensorflow-codex-124833` and
`/lustre/fs1/home/sa305415/tensorflow-codex-123997`, plus Bazel repository and
install caches. The two workspaces are outside the deletion target and remain
preserved. No Bazel process or Slurm job is active. Bazel explicitly identifies
both output bases as generated cache state; the target is therefore
reproducible rather than scientific evidence.

The exact output-base names are `68cc65ad2ac50b8863efd4819edb2b3e`
and `aa84f33fa455b1fd9f29f6010c4dfe14`. Their `README` SHA-256 values are
`fc4cf90d47cb21f0db96274e51c23bb397d23999892c9f691969dd3f9c6e221e`
and `1b48fffbeb4f5e89ad270d8727970dea677f5b088be0bfa5d2c10c2cfbe65321`.
The completed one-pass metadata walk counted `482,133` entries: `427,891`
regular files, `47,116` directories, and `7,126` symbolic links. It measured
`88,693,501,496` apparent bytes and `89,756,495,872` allocated bytes. The
inventory read did not mutate the cache.

The independently measured lowest-risk cache batch is:

- `/lustre/fs1/home/sa305415/.cache/pip`: `12,528,881,664` allocated bytes,
  `3,811` inodes;
- `/lustre/fs1/home/sa305415/shohin/.cache/huggingface`:
  `9,344,045,056` bytes, `50` inodes;
- `/lustre/fs1/home/sa305415/.cache/huggingface`: `5,111,406,592` bytes,
  `897` inodes;
- `/lustre/fs1/home/sa305415/.cache/mace`: `854,020,096` bytes, `6` inodes;
- `/lustre/fs1/home/sa305415/shohin/miniforge3/pkgs`: `457,461,760`
  bytes, `14,916` inodes. This is the Conda package-download cache only; the
  installed `miniforge3` runtime remains preserved.

These five targets total `28,295,815,168` allocated bytes (`26.35 GiB`) and
`19,680` inodes. They are download/build caches, not scientific artifacts;
their content is reproducible from package/Hugging Face sources. The pinned
PCF1 model is stored outside both Hugging Face cache targets.

## Deletion ledger

Every bounded action records the exact target, justification, recoverability,
pre/post quota, and recovered bytes/inodes here before PCF1 submission.

The first authorized cache tier was removed on `2026-08-11` after resolving
each target to its exact absolute path, confirming it was a non-symlink owned
by `sa305415`, confirming its unique quarantine name did not exist, and
confirming the user had no queued or running Slurm job. Each directory was
atomically renamed within its parent, deleted at that exact quarantine path
with a one-filesystem boundary, and then confirmed absent. Same-filesystem
quarantine was recoverable only until the following delete; after deletion,
recovery is by re-downloading the cache content.

| UTC | Exact removed target | Audited bytes / inodes | Immediate post-action Lustre usage |
|---|---|---:|---:|
| `07:12:02Z` | `/lustre/fs1/home/sa305415/.cache/pip` | `12,528,881,664 / 3,811` | `1,690,218,436 KiB / 2,332,435` |
| `07:12:18Z` | `/lustre/fs1/home/sa305415/shohin/.cache/huggingface` | `9,344,045,056 / 50` | `1,688,349,388 KiB / 2,332,385` |
| `07:12:28Z` | `/lustre/fs1/home/sa305415/.cache/huggingface` | `5,111,406,592 / 897` | `1,679,171,504 KiB / 2,331,489` |
| `07:12:39Z` | `/lustre/fs1/home/sa305415/.cache/mace` | `854,020,096 / 6` | `1,674,232,824 KiB / 2,331,483` |
| `07:12:50Z` | `/lustre/fs1/home/sa305415/shohin/miniforge3/pkgs` | `457,461,760 / 14,916` | `1,673,253,248 KiB / 2,325,999` |

Lustre quota accounting can reconcile asynchronously and the Conda cache
contained installed-environment hardlinks, so the per-row audit estimates are
not claimed as independently additive recovered quota. The settled read at
`2026-08-11T07:13:45Z` was `1,673,197,724 KiB / 2,325,999` files: a measured
net reduction of `27,386,964 KiB` and `10,247` quota-counted files from the
baseline. All five exact original paths and quarantine paths were absent. The
installed Miniforge environment, pinned PCF1 model/runtime, and scientific
artifacts were untouched.

The exact Bazel cache root was then revalidated as a non-symlink directory
owned by `sa305415`, with no active Bazel command and no user Slurm job. The
`123997` source workspace remained present and was not mutated. The `124833`
workspace referenced by the other generated README was already absent before
cleanup, making that output base an orphaned cache rather than a source copy.
At `2026-08-11T07:27:06Z`, the cache was atomically renamed to
`/lustre/fs1/home/sa305415/.cache/.pcf1-delete-bazel-20260811`; that exact
quarantine was deleted by `07:29:33Z`. Its `482,133` audited entries and
`89,756,495,872` allocated bytes are recoverable only by Bazel rebuild. At
`07:29:56Z`, both original and quarantine paths were absent, the preserved
workspace was present, and Lustre reported `1,585,544,896 KiB / 1,843,866`
files. The measured net reduction from the original baseline is therefore
`115,039,792 KiB` and `492,380` files. Remaining overage is
`526,483,136 KiB` and `833,866` files, so storage is still not qualified.

Three byte-proven duplicate shard holdings were removed next. Before mutation,
all 15 DCLM files, the one FineWeb-Edu file, and the 11 named peS2o shard files
were re-compared byte-for-byte with their completed canonical holdings. The
first two partial directories contained no other member. The peS2o partial
directory also contained `documents.jsonl.zst`, which was explicitly retained;
its SHA-256 before and after cleanup is
`c00f83f4ea48112ffd26cd196a85ddff5e89619aeda61832a0bf8554a1ccc308`.

| UTC | Exact removed target | Audited bytes / inodes | Recovery source |
|---|---|---:|---|
| `07:54:14Z` | `artifacts/shards/dclm_baseline_25b.partial.pre_live_eval_gate` | `2,093,383,680 / 16` | identical 15 files in completed `dclm_baseline_25b` |
| `07:54:26Z` | `artifacts/shards/fineweb_edu_25b.partial.pre_live_eval_gate` | `140,308,480 / 2` | identical file in completed `fineweb_edu_25b` |
| `07:55:34Z` | exactly `shard_00000.u16.zst` through `shard_00010.u16.zst` under `pes2o_selected_core_10b.partial` | `1,268,203,520 / 11` | identical files in completed `pes2o_selected_core_10b_r2` |

The settled `07:56:18Z` quota was `1,582,125,076 KiB / 1,843,837`
files, an exact tier reduction of `3,419,820 KiB / 29` files. The total
measured reduction from baseline is `118,459,612 KiB / 492,409` files;
remaining overage is `523,063,316 KiB / 833,837` files. These shard deletions
are fully recoverable from the named completed immutable copies.

The subsequent runtime audit did **not** authorize broad deletion. The 168
`shohin/runtimes` roots are experiment outputs, not a generic software cache;
the largest individual roots are below 1 GiB and carry thousands of files that
may include unique checkpoints or reports. Even the explicitly named
`mtr1_2bf7023_r2.failed_copy_1786249064` failed an exact comparison with its
nominal successor: `SHA256SUMS` differs and the failed copy alone contains
`train/jobs/dispatch_transfer_fits.sbatch`. It was preserved. Likewise, the
rejected scratchpad runtime snapshots contain historical source/report trees;
their failure suffix and later successor are not by themselves proof that
every immutable file is duplicated. They remain preserved pending exact
content-inclusion evidence. This rules out the hundreds-of-gigabytes deletion
that would otherwise be required and makes temporary quota expansion the
conservative path.

## Durable-space and quota route

The separate `/lustre/fs2` mount has ample physical capacity, but the campaign
identity has neither an allocated directory nor an ACL there. Both its user
and `arcc_pi_skattel` group quota reads are zero/default. The visible
`/lustre/fs2/rstore/rstore_skattel_catalysis` directory is owned by a distinct
`rstore_skattel_catalysis` group; `sa305415` is not a member and has no read,
write, or traverse permission. No directory was created and no ACL was
changed.

The current official ARCC file-system policy states the `1 TB / 1,000,000
files` user limits, warns that group-space files remain charged to the owning
user quota, and directs precise temporary capacity or file-limit requests to
the [Resource Expansion Request](https://ucf.qualtrics.com/jfe/form/SV_8k3Swgjmzf6hPBc).
The policy requires the exact added capacity/file count, justification, and
duration. That is the supported escalation route if the bounded provenance
audit cannot create safe PCF1 headroom.

The bounded request prepared for that form is: increase the `sa305415`
`/lustre/fs1` user limit by exactly `1 TiB` and `1,500,000` files (to `2 TiB`
and `2,500,000` files) through `2026-10-10`. After cleanup, usage remains
`1,582,125,076 KiB / 1,843,837` files, while the frozen dispatcher requires at
least `96 GiB / 100,000` free before its sole run. The additional margin is for
one source-disjoint Ministral publication confirmation, immutable
checkpoint/candidate/accounting evidence, and safe transfer before the
temporary expansion expires. The request explicitly promises no experiment
proliferation or retry and explains that further deletion would remove unique
scientific custody rather than cache waste. Submission is pending only the
form's required human/CAPTCHA authorization.

## Age-ordered destructive cleanup authorized later on 2026-08-11

The user subsequently superseded the quota-expansion wait and authorized
permanent deletion of old Shohin raw artifacts from closed lanes, including
unique raw outputs whose compact conclusions and hashes remain in git. The
preserve set remains exact: the private repository and its pushed
documentation/result JSONs; `SHOHIN_PHASE_HANDOFF_20260811.md`; the complete
qualified 9B temporal-revision release and all of its verification/runtime
dependencies; the pinned Ministral model and PCF1 environment; every current
PCF1 code, data, runtime, and custody artifact; credentials/configuration; and
unrelated workspaces. No root directory is a deletion target.

### Destructive batch A manifest — recorded before mutation

Batch A is the oldest substantial closed-lane material by resolved directory
mtime. Every target below was resolved with `realpath -e`, verified to be an
owned nonsymlink directory strictly below
`/lustre/fs1/home/sa305415/shohin/artifacts`, and measured with one-filesystem
`du` before deletion. There were no queued or running jobs. The batch totals
`136,004,235,264` allocated bytes and `1,073` inodes. It is permanently
nonrecoverable locally after quarantine deletion; only the compact git
conclusions/hashes remain.

| Resolved absolute target | Directory mtime (EDT) | Allocated bytes | Inodes |
|---|---:|---:|---:|
| `/lustre/fs1/home/sa305415/shohin/artifacts/rg` | `2026-07-03 23:20:02` | 4,046,848 | 4 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/rg_big` | `2026-07-03 23:51:07` | 265,232,384 | 4 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/shards/code_python` | `2026-07-04 10:06:35` | 11,819,839,488 | 86 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/shards/openwebmath` | `2026-07-04 11:33:47` | 15,131,181,056 | 73 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/shards/finemath4` | `2026-07-04 20:48:15` | 6,467,702,784 | 35 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/shards/finemath3` | `2026-07-07 12:08:21` | 23,341,879,296 | 127 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/profiles` | `2026-07-10 04:21:48` | 354,971,648 | 5 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/cudagraph_canary` | `2026-07-10 11:00:17` | 28,672 | 7 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/rg_v2` | `2026-07-10 14:02:40` | 299,266,048 | 6 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/shards/openmath_pt` | `2026-07-10 16:59:51` | 3,569,676,288 | 52 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/shards/fineweb_edu_5b` | `2026-07-12 07:54:27` | 6,450,515,968 | 48 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/problems` | `2026-07-12 08:35:28` | 4,354,048 | 2 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/shards/dclm_baseline_5b` | `2026-07-12 11:59:48` | 6,993,186,816 | 52 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/monitors` | `2026-07-12 14:30:52` | 1,728,512 | 5 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/monitor_history` | `2026-07-12 14:33:36` | 16,384 | 4 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/shards/fineweb_edu_25b` | `2026-07-12 16:36:27` | 6,450,475,008 | 48 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/vrwm` | `2026-07-12 20:19:20` | 1,998,848 | 8 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/rejected` | `2026-07-13 00:11:02` | 456,581,120 | 9 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/shards/dclm_baseline_25b` | `2026-07-13 09:47:13` | 35,001,827,328 | 252 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/shards/fineweb_edu_25b_r2.partial` | `2026-07-13 17:42:19` | 14,023,565,312 | 101 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/research` | `2026-07-14 21:12:47` | 12,288 | 3 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/wgrq_stage_a` | `2026-07-15 04:17:44` | 302,596,096 | 6 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/evidence` | `2026-07-15 15:19:27` | 62,894,080 | 2 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/eval_history_mirror` | `2026-07-15 15:24:22` | 1,851,392 | 3 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/factorial_v4` | `2026-07-17 15:26:57` | 3,265,114,112 | 61 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/carry_motor` | `2026-07-18 04:05:27` | 186,535,936 | 43 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/tokenizer` | `2026-07-19 17:51:04` | 2,314,240 | 2 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/diagnostics` | `2026-07-24 20:54:15` | 71,471,104 | 9 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/rg_v3` | `2026-07-24 20:54:23` | 713,076,736 | 8 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/rg_v4` | `2026-07-24 20:54:23` | 760,295,424 | 8 |

The protected roots under `artifacts/product_reasoning` and
`artifacts/external`, the repository `.git`, all documentation, the pinned
environment, and all current PCF1 paths are outside this manifest and must be
revalidated present after deletion.

Batch A executed exactly as manifest-bound. Each target was revalidated with
the same byte/inode measurement, owner, nonsymlink, and resolved-path checks,
then atomically renamed to a literal same-parent
`.pcf1-age-delete-A-*` quarantine and permanently deleted with a
one-filesystem boundary. All originals and quarantines are absent. The pinned
Ministral tree, qualified Qwen 9B release, `idr_aqc_release` artifacts and
runtimes, pinned PCF1 environment, repository `.git`, and documentation were
revalidated present. Settled quota at `2026-08-11T08:28:31Z` was
`1,449,308,440 KiB / 1,842,764` files: exact Batch A recovery of
`132,816,636 KiB / 1,073` files. Remaining hard-limit overage is
`390,246,680 KiB / 832,764` files. The deleted bytes are not locally
recoverable.

### Destructive batch B1 manifest — recorded before mutation

Batch B1 resumes strict age order with four missed synchronization/staging
trees, four closed ETTR artifact trees, and the first eight direct scratchpad
children. All are literal, owned, nonsymlink paths whose resolved paths remain
under the named Shohin subdirectory; no parent root is a target. The legacy
`artifacts/runtime` tree was enumerated and contains only July ETTR claim
bundles, trusted smoke sources, and a cached wheel—no Qwen release, Ministral,
PCF1, or `idr_aqc` member. Batch B1 totals `326,680,899,584` allocated bytes
and `9,850` inodes. It is permanently nonrecoverable locally after deletion.

| Resolved absolute target | Directory mtime (EDT) | Allocated bytes | Inodes |
|---|---:|---:|---:|
| `/lustre/fs1/home/sa305415/shohin/staging_sync` | `2026-07-05 11:42:55` | 24,576 | 3 |
| `/lustre/fs1/home/sa305415/shohin/__sync_multi.tmp` | `2026-07-09 11:38:19` | 106,496 | 4 |
| `/lustre/fs1/home/sa305415/shohin/.codex-retention-sync` | `2026-07-14 09:51:49` | 565,248 | 19 |
| `/lustre/fs1/home/sa305415/shohin/.sync_v5` | `2026-07-25 19:47:57` | 94,208 | 6 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/runtime` | `2026-07-26 16:14:16` | 22,518,054,912 | 46 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/ettr-supervisor-smoke` | `2026-07-26 16:18:30` | 3,907,584 | 118 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/shard_scans` | `2026-07-28 18:04:01` | 188,416 | 7 |
| `/lustre/fs1/home/sa305415/shohin/artifacts/source_probes` | `2026-07-28 18:21:34` | 126,976 | 19 |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_8a68e4a_r1` | `2026-07-28 19:07:06` | 209,268,736 | 2,728 |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/deploy_fc7df73` | `2026-07-28 23:53:21` | 53,248 | 5 |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/pes2o_uniform30_v1` | `2026-07-29 00:07:14` | 17,437,204,480 | 31 |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/fineweb_edu_score4_code_e87be57` | `2026-07-29 01:00:18` | 118,784 | 13 |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/ettr_release_source_d84a5a0` | `2026-07-29 01:20:00` | 42,373,120 | 3,353 |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/ettr_release_source_81e3184_clean` | `2026-07-29 01:21:40` | 38,051,840 | 1,680 |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/ettr_release_source_d84a5a0_clean` | `2026-07-29 01:21:40` | 35,356,672 | 1,677 |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/fineweb_edu_sample100bt_87f0914` | `2026-07-29 02:38:24` | 286,395,404,288 | 141 |

The final 286.4-GB tree is the old FineWeb-Edu physical sample holding; its
compact selection conclusions/hashes remain in git. The old peS2o physical
holding is likewise a closed source-stage input. Neither is a PCF1 source
bank. The qualified release, its exact `idr_aqc_8f0bd8d_r2` runtime, pinned
Ministral/environment, repository, docs, credentials, and PCF1 paths remain
explicit post-deletion checks.

The first B1 transaction attempt failed closed before mutation because its
preserve assertion named a nonexistent stale `/shohin-ettr-architecture/docs`
path. The actual protected repository and documentation anchors are
`/shohin/.git`, `/shohin/SHOHIN.md`, `/shohin/AGENT_RUNBOOK.md`, and
`/shohin/docs`; after correcting only that check, the identical 16-target
manifest executed. Every target was revalidated as wholly owned by
`sa305415`, same-device, nonsymlink-rooted, exact in realpath and
byte/inode measurement, with zero jobs and zero protected-anchor overlap.
Each was renamed to a unique literal same-parent
`.pcf1-age-delete-B1-*` quarantine and permanently deleted one-filesystem.
All originals/quarantines are absent and all 14 release/model/runtime/data/git/
documentation anchors remain present.

Settled quota at `2026-08-11T08:43:14Z` was
`1,130,284,128 KiB / 1,832,914` files, observed B1 recovery of
`319,024,312 KiB / 9,850` files. The four-KiB difference from the manifest's
byte quotient is parent-directory allocation. Remaining hard-limit overage is
`71,222,368 KiB / 822,914` files; reaching 128-GiB byte and 150,000-file
headroom still requires `205,440,096 KiB / 972,914` files of recovery. B1 is
permanently nonrecoverable locally.

### Destructive batch B2a manifest — recorded before mutation

Batch B2a is a bounded inode-focused selection of 79 explicit direct-child
runtime trees from closed ETTR, capability-floor, CSDC/divergence,
small-OLMoE/ECR, and prohibited NDR1 lanes. No parent root is a target. A
fresh read-only traversal found zero user jobs. Every literal path below
equals its resolved realpath; every root and every traversed entry is owned
by sa305415; every root is a nonsymlink directory; and every tree remains on
one device.

The per-row allocated-byte/path-inode sum is 65,623,306,240 bytes and
368,163 entries. There is no inode identity overlap between listed targets,
but 4,550 regular-file inodes have additional hardlinks outside this batch.
Treating those files and their blocks as retained gives a conservative
projected quota recovery of 64,776,028,160 bytes (63,257,840 KiB) and
363,613 inodes. Mutation remains unauthorized until this exact set is
independently verified.

| Resolved absolute target | Root mtime (UTC) | Allocated bytes | Path inodes |
|---|---:|---:|---:|
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_4f10b65_r1 | 2026-07-31 09:17:12 | 849,043,456 | 7,100 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_algebraic_joint_runtime_cf875bc_r1 | 2026-08-01 14:07:56 | 848,044,032 | 7,132 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_algebraic_joint_runtime_cf875bc_r2 | 2026-08-01 14:07:56 | 848,044,032 | 7,132 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_algebraic_joint_runtime_cf875bc_r3 | 2026-08-01 14:07:56 | 848,044,032 | 7,132 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_parallel_runtime_b45c046_r4 | 2026-08-01 20:40:47 | 848,211,968 | 7,151 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_parallel_runtime_78ed30d_r1 | 2026-08-01 20:46:48 | 848,211,968 | 7,151 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_opcode_projection_runtime_1a0cd1c_r4 | 2026-08-02 07:38:09 | 838,041,600 | 3,836 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_failure_tree_runtime_f66d3a4_r1 | 2026-08-02 07:53:47 | 838,090,752 | 3,841 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_mask_audit_runtime_9c47151_r1 | 2026-08-02 07:53:47 | 838,135,808 | 3,844 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_public_audit_runtime_f48337b_r1 | 2026-08-02 07:53:47 | 838,107,136 | 3,841 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_cover_verified_runtime_e5dcb41_r1 | 2026-08-02 08:56:48 | 838,148,096 | 3,844 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_structured_program_runtime_cbf58b9_r1 | 2026-08-02 09:04:59 | 838,144,000 | 3,844 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_declaration_binding_runtime_1bd41aa_r1 | 2026-08-02 09:35:33 | 838,221,824 | 3,848 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_guarded_edit_runtime_17b4e60_r1 | 2026-08-02 09:48:23 | 838,246,400 | 3,850 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_operation_recurrent_runtime_0e42ba1_r1 | 2026-08-02 10:07:12 | 838,332,416 | 3,854 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_public_operation_audit_runtime_f70bca1_r1 | 2026-08-02 10:21:08 | 838,365,184 | 3,858 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_operation_state_runtime_29b272f_r1 | 2026-08-02 10:38:12 | 838,451,200 | 3,863 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_operation_audits_runtime_70e5dcb_r1 | 2026-08-02 10:49:47 | 838,479,872 | 3,867 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_operation_factor_audit_runtime_3cdc96d_r1 | 2026-08-02 11:25:50 | 838,479,872 | 3,867 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_operation_factor_marginal_runtime_e5cb34c_r1 | 2026-08-02 11:55:09 | 838,684,672 | 3,879 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_sparse_effect_runtime_e5cb34c_r1 | 2026-08-02 11:57:33 | 838,684,672 | 3,879 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_operation_effect_set_runtime_fb8b372_r1 | 2026-08-02 12:38:09 | 838,795,264 | 3,881 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_operation_effect_set_runtime_3c5b954_r1 | 2026-08-02 12:44:37 | 838,803,456 | 3,881 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_operation_effect_set_runtime_c630a55_r1 | 2026-08-02 12:49:57 | 838,803,456 | 3,881 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_effect_diagnostic_work_r1 | 2026-08-02 12:50:41 | 840,577,024 | 3,933 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_operation_effect_set_runtime_c630a55_r2 | 2026-08-02 12:50:41 | 838,803,456 | 3,881 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_effect_diagnostic_runtime_16455f3_r3 | 2026-08-02 13:29:51 | 839,020,544 | 3,896 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_effect_comparative_runtime_3aefc3e_r1 | 2026-08-02 13:45:15 | 839,057,408 | 3,898 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_effect_role_runtime_670e3cc_r1 | 2026-08-02 14:24:01 | 839,102,464 | 3,900 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_effect_role_runtime_e1c8245_r2 | 2026-08-02 14:26:31 | 839,090,176 | 3,902 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_effect_role_runtime_33785eb_r3 | 2026-08-02 14:42:43 | 839,106,560 | 3,905 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_effect_cardinality_runtime_0ca7408_r1 | 2026-08-02 15:03:23 | 839,069,696 | 3,892 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_effect_cardinality_runtime_0ca7408_r2 | 2026-08-02 15:04:25 | 839,069,696 | 3,892 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_effect_balance_runtime_22d0943_r2 | 2026-08-02 15:17:28 | 839,102,464 | 3,895 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_effect_balance_runtime_49464a2_r1 | 2026-08-02 15:40:51 | 839,114,752 | 3,895 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_write_link_rail_runtime_27f8b44_r1 | 2026-08-02 16:06:48 | 839,139,328 | 3,895 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_write_link_dependency_runtime_e74f361_r1 | 2026-08-02 16:45:44 | 839,184,384 | 3,895 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_operation_family_gate_runtime_d42a054_r1 | 2026-08-02 17:09:26 | 839,204,864 | 3,895 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_operation_family_island_runtime_2456dd7_r2 | 2026-08-02 18:17:21 | 839,262,208 | 3,898 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_deferred_ledger_runtime_3f9817e_r1 | 2026-08-02 19:06:11 | 839,335,936 | 3,904 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_state_bound_family_runtime_898591c_r1 | 2026-08-02 19:59:29 | 839,364,608 | 3,905 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_operation_family_campaign_runtime_43084cd_r1 | 2026-08-02 20:38:37 | 839,450,624 | 3,914 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/capability_floor_runtime_ebc8483_r2 | 2026-08-02 22:51:52 | 839,696,384 | 3,934 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/capability_floor_runtime_651d510_r1 | 2026-08-02 23:24:59 | 839,745,536 | 3,937 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/capability_floor_runtime_9bc1769_r1 | 2026-08-03 00:09:05 | 839,786,496 | 3,940 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/capability_floor_runtime_6fd7c94_r1 | 2026-08-03 00:39:06 | 839,823,360 | 3,944 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/capability_floor_runtime_dd6efea_r1 | 2026-08-03 01:13:30 | 839,868,416 | 3,949 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/math_verify_0_9_0_r1 | 2026-08-03 04:29:22 | 87,158,784 | 3,850 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/csdc_lexical_runtime_8c62b5c | 2026-08-05 11:11:06 | 842,223,616 | 4,218 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/diverge_v0_matched_gate_e7ef5af_r1 | 2026-08-05 23:17:27 | 842,743,808 | 4,251 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/diverge_nve1_b06bae1_r1 | 2026-08-06 17:18:11 | 1,038,667,776 | 4,508 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/diverge_iem1_8fd48c5_r1 | 2026-08-06 19:27:47 | 848,220,160 | 4,517 |
| /lustre/fs1/home/sa305415/shohin/runtime/diverge_rrg1_f23d032 | 2026-08-07 01:13:47 | 848,871,424 | 4,586 |
| /lustre/fs1/home/sa305415/shohin/runtime/diverge_sti1_8880157 | 2026-08-07 01:44:07 | 848,916,480 | 4,592 |
| /lustre/fs1/home/sa305415/shohin/runtime/diverge_npl1_semantic_c39611b | 2026-08-07 02:09:17 | 848,977,920 | 4,599 |
| /lustre/fs1/home/sa305415/shohin/runtime/diverge_sti1_continuation_aebc910 | 2026-08-07 02:32:52 | 848,986,112 | 4,600 |
| /lustre/fs1/home/sa305415/shohin/runtime/diverge_pqi1_defc87f | 2026-08-07 03:54:35 | 849,072,128 | 4,611 |
| /lustre/fs1/home/sa305415/shohin/runtime/diverge_pqi1_defc87f_2cacc988 | 2026-08-07 03:55:51 | 849,068,032 | 4,611 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/diverge_snl1_6876723_r1 | 2026-08-07 18:25:24 | 851,673,088 | 4,827 |
| /lustre/fs1/home/sa305415/shohin/scratchpad/diverge_opb1_1d5b1b7_r1 | 2026-08-07 19:19:58 | 851,812,352 | 4,838 |
| /lustre/fs1/home/sa305415/shohin/runtime/idr4_6dddeaf_r1 | 2026-08-08 15:34:17 | 854,863,872 | 4,996 |
| /lustre/fs1/home/sa305415/shohin/runtimes/ecr1_22c3679_r1 | 2026-08-09 07:19:12 | 856,371,200 | 5,122 |
| /lustre/fs1/home/sa305415/shohin/runtimes/ecr1_a5606ca_r2 | 2026-08-09 07:22:49 | 856,395,776 | 5,127 |
| /lustre/fs1/home/sa305415/shohin/runtimes/ecr1_fddc09e_r3 | 2026-08-09 07:25:09 | 856,403,968 | 5,130 |
| /lustre/fs1/home/sa305415/shohin/runtimes/ecr1_eb79755_r4 | 2026-08-09 07:27:42 | 856,403,968 | 5,130 |
| /lustre/fs1/home/sa305415/shohin/runtimes/ecr1_ae6234c_r5 | 2026-08-09 07:30:41 | 856,424,448 | 5,134 |
| /lustre/fs1/home/sa305415/shohin/runtimes/ecr1_19e96d1_r6 | 2026-08-09 07:37:05 | 856,485,888 | 5,145 |
| /lustre/fs1/home/sa305415/shohin/runtimes/ecr1_b7461a6_r7 | 2026-08-09 07:43:17 | 856,481,792 | 5,145 |
| /lustre/fs1/home/sa305415/shohin/runtimes/ecr1_d7cbf57_r8 | 2026-08-09 07:52:40 | 856,510,464 | 5,149 |
| /lustre/fs1/home/sa305415/shohin/runtimes/ecr1_f1e7538_r9 | 2026-08-09 07:56:54 | 856,514,560 | 5,149 |
| /lustre/fs1/home/sa305415/shohin/runtimes/ecr1_d033b1f_r10 | 2026-08-09 08:09:43 | 856,514,560 | 5,149 |
| /lustre/fs1/home/sa305415/shohin/runtimes/ser1_6ad9739_r11 | 2026-08-09 08:19:45 | 856,563,712 | 5,156 |
| /lustre/fs1/home/sa305415/shohin/runtimes/rme1_f492a38_r12 | 2026-08-09 08:50:37 | 856,559,616 | 5,149 |
| /lustre/fs1/home/sa305415/shohin/runtimes/ctsr1_95ea1c5_r13 | 2026-08-09 09:22:44 | 856,625,152 | 5,156 |
| /lustre/fs1/home/sa305415/shohin/runtime/wgp1_confirm_1b60f40_r1 | 2026-08-10 15:38:44 | 859,279,360 | 5,526 |
| /lustre/fs1/home/sa305415/shohin/runtime/dtc1_gate_f9b5fda_r1 | 2026-08-10 19:10:38 | 859,742,208 | 5,586 |
| /lustre/fs1/home/sa305415/shohin/runtime/ectr0_0cf5a29_r1 | 2026-08-10 20:56:27 | 859,967,488 | 5,617 |
| /lustre/fs1/home/sa305415/shohin/runtimes/ndr1_e481dc8_r2 | 2026-08-10 22:03:32 | 883,568,640 | 11,251 |
| /lustre/fs1/home/sa305415/shohin/runtimes/ndr1_87c5c9d_r1 | 2026-08-10 22:33:34 | 229,646,336 | 7,583 |

The selected Shohin ETTR/capability-floor/CSDC/divergence runtimes are closed
copies; ECR/SER/RME/CTSR are non-surviving small-OLMoE progression copies;
IDR4 is the closed 4B branch; and the two NDR1 targets belong to the
explicitly prohibited retry lane. The target set has zero path overlap with
the protected Ministral model, external Qwen 9B source model, qualified
idr_aqc release/inputs/interactions, idr_aqc runtime revisions r1/r2, pinned
PCF1 environment, product data, either repository .git anchor,
documentation/result JSONs/handoff, credentials/configuration, or any current
PCF1 path. Qwen-9B scratch preflight/B1/SAG1 directories were deliberately
excluded.

#### Batch B2a execution receipt

The first B2a launch failed closed before mutation because the protected-path
presence check still asserted
`/lustre/fs1/home/sa305415/shohin/SHOHIN.md`, which does not exist. A fresh
read-only check confirmed that this working tree instead retains
`/shohin/.git`, `/shohin/docs`, and `/shohin/AGENT_RUNBOOK.md`, while the
stale repository retains its `.git`, `SHOHIN.md`, and `AGENT_RUNBOOK.md`.
After explicit authorization to remove only that impossible assertion, the
identical recorded 79-row target set was used; no path was added, omitted, or
changed.

Immediately before mutation there were zero user jobs. All 79 targets were
again confirmed to match their recorded resolved realpath, UTC root mtime,
allocated bytes, and path-inode count; each root was a nonsymlink directory,
every traversed entry was owned by `sa305415`, every tree remained on one
device, and the set had zero path overlap with all 20 valid protected
anchors. Processing followed the recorded ascending chronology. Each target
was atomically renamed to its unique literal same-parent
`.pcf1-age-delete-B2a-*` quarantine and only that quarantine was permanently
deleted with a one-filesystem boundary. All 79 originals and quarantines are
absent. All 20 protected anchors remain present, and the post-transaction
scheduler checks remained empty.

The immediate quota observation at completion
(`2026-08-11T09:07:22Z`) was `1,067,243,156 KiB / 1,469,301` files. Lustre
accounting then settled at `1,067,026,292 KiB / 1,469,301` files, unchanged
across observations at `09:07:31Z`, `09:07:52Z`, and `09:08:12Z`. Relative
to the settled pre-B2a baseline of `1,130,284,128 KiB / 1,832,914` files,
actual recovery was `63,257,836 KiB / 363,613` files. The four-KiB byte
difference from the conservative projection is parent-directory allocation;
the inode recovery exactly matches the hardlink-adjusted projection. B2a is
permanently nonrecoverable locally.

Remaining hard-limit overage is `7,964,532 KiB / 459,301` files. Reaching
128-GiB byte and 150,000-file headroom still requires
`142,182,260 KiB / 609,301` files of recovery. No B2b selection or mutation
has begun.

### Destructive batch B2b manifest — recorded before mutation

Status: **read-only manifest only; mutation is not authorized.** B2b is one
globally age-ordered selection of literal direct-child directories beneath
`/shohin/scratchpad`, `/shohin/runtime`, and `/shohin/runtimes`. No
parent root, top-level loose file, external model, repository, or unrelated
workspace is a target.

A first bounded metadata pass over 509 conservatively eligible closed runtime
trees found that those trees could recover at most
`123,577,503,744 bytes / 847,415` quota inodes after hardlink adjustment:
the inode requirement passed, but the byte recovery was only
`120,681,156 KiB`, below 160 GiB. No mutation occurred. The final candidate
order therefore adds 12 older, explicit historical ETTR raw/staging children
from the same three authorized roots, as allowed by the superseding
age-ordered cleanup authority. They were merged by root mtime and literal path
with the closed runtime candidates. No external-model child was needed.

The final replay at `2026-08-11T09:23:48Z` found 521 eligible directories and
stopped at the first globally ordered prefix satisfying both conservative
targets: the 402 literal rows below, from
`2026-07-29T06:58:15Z` through `2026-08-09T17:18:31Z`. The
bounded one-filesystem walk visited `774,947` entries
against a 1,500,000-entry fail-closed ceiling. There were zero user jobs both
before and after. Every listed path is its resolved realpath and an immediate
child of one of the three named roots; every root is a nonsymlink directory;
every traversed entry is owned by `sa305415`; and every tree remains on its
root device.

The per-row path sum is `251,842,965,504` allocated bytes and
`774,947` path entries. The selection contains
`13,538` complete multi-link inode identities
(`3,326,873,600` uniquely allocated bytes); all
`13,538` span more than one selected target.
Their blocks and quota inode are credited only once. There are zero selected
identities with links outside the batch: outside identities, selected link
entries, missing links, and retained outside-linked bytes are all zero. After
cross-target deduplication, conservative projected quota recovery is
`233,581,510,656 bytes = 228,106,944 KiB / 700,823`
quota inodes. This exceeds 160 GiB / 700,000 by
`61,782,818,816 bytes / 823`
inodes, and exceeds the remaining PCF1 headroom requirement by
`85,924,684 KiB / 91,522`
inodes.

The protected in-root Qwen-9B preflight/B1/SAG1 and both
`idr_aqc_8f0bd8d` runtime revisions were exact exclusions:

- `/lustre/fs1/home/sa305415/shohin/runtimes/idr_aqc_8f0bd8d_r1`
- `/lustre/fs1/home/sa305415/shohin/runtimes/idr_aqc_8f0bd8d_r2`
- `/lustre/fs1/home/sa305415/shohin/scratchpad/qwen9b_b1_dddb36f_r1`
- `/lustre/fs1/home/sa305415/shohin/scratchpad/qwen9b_preflight_1467e45_r1`
- `/lustre/fs1/home/sa305415/shohin/scratchpad/sag1_9b_594c7ca_r1`

A fresh post-replay check found all 20 protected anchors present: pinned
Ministral, external and product Qwen-9B models, qualified release/inputs/
interactions, both qualified runtime revisions, pinned environment, product
data/sources, the three Qwen-9B scratch anchors, both repository/documentation
sets, and the current runbook. All other protected/current PCF1/model/release/
repository/data/custody paths are structurally outside the three roots, and
the candidate filter also excluded every direct-child name containing PCF1,
Qwen, Ministral, `idr_aqc`, SAG1, product, custody, assessor, holdout,
credential, configuration, private-review, interaction, result/report/
manifest, or product-data provenance. The historical FinePDF, Essential-Web,
Nemotron, scale-pilot, and ETTR source-materializer rows below belong only to
the closed synthetic ETTR program; none is a handoff-required, qualified-9B,
or current PCF1 source bank.

If this exact projection were later authorized and recovered as measured,
settled usage would be approximately `838,919,348 KiB / 768,478` files,
giving `220,142,412 KiB / 241,522` of hard-limit headroom. This paragraph is
a projection, not authorization. No path has been renamed, quarantined, or
deleted.

| Resolved literal target | Root mtime (UTC) | Allocated bytes | Path inodes | Ownership / root / device contract | Phase provenance |
|---|---:|---:|---:|---|---|
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_batch_b31dd27` | `2026-07-29T06:58:15Z` | 13,946,880 | 842 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_fineweb_r2_49a5584` | `2026-07-29T07:03:19Z` | 13,946,880 | 842 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_scale_758d94d` | `2026-07-29T07:34:51Z` | 190,226,432 | 2,686 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/scale_pilot_719591_758d94d` | `2026-07-29T07:47:41Z` | 500,465,664 | 4 | `sa305415`; nonsymlink directory; same-device | closed historical ETTR raw/staging tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/scale_pilot_719591_b32a4_758d94d` | `2026-07-29T07:53:02Z` | 500,465,664 | 4 | `sa305415`; nonsymlink directory; same-device | closed historical ETTR raw/staging tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_pes2o_r2_2b0d30c` | `2026-07-29T08:04:15Z` | 13,959,168 | 844 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_883875c0` | `2026-07-29T09:13:54Z` | 19,369,984 | 1,847 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_97f19321` | `2026-07-29T09:18:59Z` | 15,638,528 | 930 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_524f3e50` | `2026-07-29T09:32:21Z` | 15,663,104 | 934 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_finepdfs_c8acb46a` | `2026-07-29T09:45:03Z` | 830,955,520 | 3,410 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/finepdfs_edu_eng_uniform16_9cfabe2` | `2026-07-29T10:11:11Z` | 47,624,335,360 | 17 | `sa305415`; nonsymlink directory; same-device | closed historical ETTR raw/staging tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_essential_runtime_e8d59eb` | `2026-07-29T10:57:18Z` | 147,456 | 10 | `sa305415`; nonsymlink directory; same-device | closed historical Shohin runtime tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/essential_web_2024_38_uniform256_ce4eccc` | `2026-07-29T11:08:23Z` | 65,818,554,368 | 257 | `sa305415`; nonsymlink directory; same-device | closed historical ETTR raw/staging tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_specialized_runtime_4d4d06d` | `2026-07-29T11:30:46Z` | 65,536 | 7 | `sa305415`; nonsymlink directory; same-device | closed historical Shohin runtime tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/nemotron_specialized_v1_1_13fa979_profile_sources` | `2026-07-29T11:31:15Z` | 281,444,352 | 5 | `sa305415`; nonsymlink directory; same-device | closed historical ETTR raw/staging tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_formal_logic_runtime_976cdc6` | `2026-07-29T11:40:57Z` | 77,824 | 6 | `sa305415`; nonsymlink directory; same-device | closed historical Shohin runtime tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_transfer_74c4fe3` | `2026-07-29T15:14:12Z` | 1,466,368 | 4 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_fast_085f7a8` | `2026-07-29T19:25:30Z` | 28,672 | 4 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_assembly_d170f25` | `2026-07-29T19:34:05Z` | 45,056 | 5 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_assembly_8ae4eca` | `2026-07-29T19:35:33Z` | 45,056 | 5 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_publish_b1c617d.partial` | `2026-07-29T21:34:56Z` | 823,877,632 | 2,960 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_publish_b1c617d_lite` | `2026-07-29T21:35:39Z` | 14,553,088 | 893 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/ettr_release_source_35333c3_full_clean` | `2026-07-29T22:32:18Z` | 1,028,042,752 | 3,507 | `sa305415`; nonsymlink directory; same-device | closed historical ETTR raw/staging tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/ettr_release_source_81e3184_full_clean` | `2026-07-29T22:32:18Z` | 1,027,215,360 | 3,500 | `sa305415`; nonsymlink directory; same-device | closed historical ETTR raw/staging tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/ettr_release_source_d9d77af_full_clean` | `2026-07-29T22:32:18Z` | 1,027,940,352 | 3,504 | `sa305415`; nonsymlink directory; same-device | closed historical ETTR raw/staging tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_68bf17f` | `2026-07-29T22:59:48Z` | 135,168 | 10 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_68bf17f_r1` | `2026-07-29T22:59:48Z` | 15,667,200 | 934 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_1afc3b4_full_r1` | `2026-07-29T23:19:01Z` | 31,326,208 | 1,735 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/ettr_packet_audit_source_fbf12df_full_r1` | `2026-07-30T01:32:03Z` | 1,028,972,544 | 3,530 | `sa305415`; nonsymlink directory; same-device | closed historical ETTR raw/staging tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/ettr_materializer_source_e5f3705_packet_v2_r1` | `2026-07-30T01:54:06Z` | 1,030,692,864 | 3,551 | `sa305415`; nonsymlink directory; same-device | closed historical ETTR raw/staging tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/ettr_materializer_adapter_e5f3705_packet_v2_r1` | `2026-07-30T01:54:37Z` | 24,576 | 3 | `sa305415`; nonsymlink directory; same-device | closed historical ETTR raw/staging tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/ettr_materializer_assembly_e5f3705_packet_v2_r1` | `2026-07-30T02:07:27Z` | 24,576 | 3 | `sa305415`; nonsymlink directory; same-device | closed historical ETTR raw/staging tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_e5f3705_full_r1` | `2026-07-30T02:55:39Z` | 200,826,880 | 2,296 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_confirm_fast_3106202_r1` | `2026-07-30T06:49:31Z` | 69,632 | 14 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_confirm_assembly_b614b30_r1` | `2026-07-30T06:53:21Z` | 94,208 | 16 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_ladder_c9ede03_r1` | `2026-07-30T08:39:27Z` | 40,960 | 9 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_eager_e660153_r1` | `2026-07-30T11:55:33Z` | 49,152 | 9 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_diag_077f57b_r1` | `2026-07-30T12:03:34Z` | 28,672 | 6 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_07e7f63_full_r1.partial-fork-failed-20260730T0806` | `2026-07-30T12:07:02Z` | 199,299,072 | 2,276 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_07e7f63_full_r2.partial-no-pytest-20260730T0808` | `2026-07-30T12:09:30Z` | 199,544,832 | 2,277 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_07e7f63_full_r3` | `2026-07-30T12:09:59Z` | 201,297,920 | 2,324 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_25d84f0_full_r1` | `2026-07-30T12:14:34Z` | 201,297,920 | 2,324 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_02e8c77_full_r1` | `2026-07-30T12:34:09Z` | 831,901,696 | 3,492 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_132d983_full_r1` | `2026-07-30T12:34:09Z` | 831,909,888 | 3,492 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_deea68c_full_r1` | `2026-07-30T12:59:52Z` | 831,913,984 | 3,492 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_f644406_full_r1` | `2026-07-30T13:18:01Z` | 831,913,984 | 3,492 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_4232338_full_r1` | `2026-07-30T16:00:44Z` | 199,593,984 | 2,278 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_99eda96_full_r1` | `2026-07-30T16:08:00Z` | 199,598,080 | 2,278 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_2926987_full_r1` | `2026-07-30T16:24:30Z` | 199,622,656 | 2,280 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_43005ad_full_r1` | `2026-07-30T16:29:09Z` | 199,622,656 | 2,280 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_aeb4aff_full_r1` | `2026-07-30T16:31:46Z` | 199,626,752 | 2,280 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_b7df632_full_r1` | `2026-07-30T16:45:15Z` | 832,045,056 | 3,498 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_b7df632_full_r2` | `2026-07-30T16:46:07Z` | 832,040,960 | 3,498 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_70f8bb0_full_r1` | `2026-07-30T17:10:50Z` | 832,053,248 | 3,498 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_309cb02_full_r1` | `2026-07-30T17:35:58Z` | 833,810,432 | 3,544 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_0b3c8bb_full_r1` | `2026-07-30T17:45:35Z` | 833,810,432 | 3,544 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_66eafea_full_r1` | `2026-07-30T18:06:05Z` | 833,814,528 | 3,544 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_7412a0f_full_r1` | `2026-07-30T20:25:37Z` | 832,061,440 | 3,498 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_7d7695b_full_r1` | `2026-07-30T20:33:41Z` | 832,065,536 | 3,498 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_3da6f44_full_r1` | `2026-07-30T21:27:20Z` | 832,094,208 | 3,500 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_821351b_full_r1` | `2026-07-30T22:14:15Z` | 832,131,072 | 3,502 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_0cc7e05_full_r1` | `2026-07-30T22:17:39Z` | 832,131,072 | 3,502 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_c53fa8d_full_r1` | `2026-07-30T22:23:40Z` | 832,135,168 | 3,502 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_b2fc0ba_full_r1` | `2026-07-30T22:31:26Z` | 832,135,168 | 3,502 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_68c8780_full_r1` | `2026-07-30T23:00:40Z` | 832,180,224 | 3,504 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_634dc76_full_r1` | `2026-07-31T00:02:35Z` | 831,795,200 | 3,503 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_634dc76_full_r2` | `2026-07-31T00:06:11Z` | 832,196,608 | 3,504 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_88336a5_full_r1` | `2026-07-31T00:14:50Z` | 831,803,392 | 3,503 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_88336a5_full_r2` | `2026-07-31T00:17:24Z` | 832,200,704 | 3,504 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_dev_anchor_20260730_r1` | `2026-07-31T00:35:44Z` | 836,669,440 | 3,636 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_92f8f0a_full_r1` | `2026-07-31T00:37:02Z` | 831,803,392 | 3,503 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_92f8f0a_full_r2` | `2026-07-31T00:40:22Z` | 832,200,704 | 3,504 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_b315875_full_r1` | `2026-07-31T00:54:14Z` | 832,208,896 | 3,504 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_549c7bd_full_r1` | `2026-07-31T01:04:42Z` | 832,200,704 | 3,504 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_47c4fb2_full_r1` | `2026-07-31T02:15:04Z` | 832,241,664 | 3,506 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_096f703_full_r1` | `2026-07-31T05:08:50Z` | 832,282,624 | 3,512 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_29b393d_full_r1` | `2026-07-31T05:21:59Z` | 832,294,912 | 3,512 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_b981eb4_full_r1` | `2026-07-31T06:19:35Z` | 832,303,104 | 3,512 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_22ed5e1_r1` | `2026-07-31T08:55:05Z` | 31,383,552 | 1,971 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_9ea7981_r1` | `2026-07-31T08:58:28Z` | 31,395,840 | 1,972 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_89fc0ad_r1` | `2026-07-31T09:44:43Z` | 832,286,720 | 3,532 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_89fc0ad_r2` | `2026-07-31T09:46:24Z` | 832,552,960 | 3,532 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_fd78cbd_r1` | `2026-07-31T09:57:34Z` | 4,096 | 1 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_fd78cbd_r2` | `2026-07-31T09:58:12Z` | 832,565,248 | 3,532 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_479472d_r2` | `2026-07-31T10:27:47Z` | 847,343,616 | 7,065 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_a13caf3_r1` | `2026-07-31T12:07:42Z` | 832,905,216 | 3,535 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_47008dc_r1` | `2026-07-31T12:18:01Z` | 832,614,400 | 3,535 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_9e1b107_r1` | `2026-07-31T12:45:04Z` | 832,622,592 | 3,535 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_dbe27ef_r1` | `2026-07-31T12:57:52Z` | 832,622,592 | 3,535 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_e1f8e3f_r1` | `2026-07-31T12:59:38Z` | 832,626,688 | 3,535 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_e44976b_r1` | `2026-07-31T13:46:23Z` | 832,622,592 | 3,535 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_743ece1_r1` | `2026-07-31T14:16:07Z` | 832,626,688 | 3,535 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_42c3456_r1` | `2026-07-31T14:16:15Z` | 832,626,688 | 3,535 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_de79470_r1` | `2026-07-31T14:36:41Z` | 832,630,784 | 3,535 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_685f685_r1` | `2026-07-31T14:37:52Z` | 832,630,784 | 3,535 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_a630392_r1` | `2026-07-31T15:00:58Z` | 832,634,880 | 3,535 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_0a6e4e8_r1` | `2026-07-31T15:28:09Z` | 832,655,360 | 3,535 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_e818e8d_rejected_wrong_source_commit` | `2026-07-31T15:40:07Z` | 832,655,360 | 3,535 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_e818e8d_r2` | `2026-07-31T15:40:37Z` | 832,655,360 | 3,535 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_0bba846_r1` | `2026-07-31T15:56:17Z` | 832,667,648 | 3,535 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_f529fdb_r2` | `2026-07-31T16:42:43Z` | 832,675,840 | 3,535 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_ac55889_r1` | `2026-07-31T16:55:01Z` | 832,688,128 | 3,536 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_3ba93f2_r1` | `2026-07-31T17:09:13Z` | 832,688,128 | 3,536 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_fb8725b_r1` | `2026-07-31T17:24:19Z` | 832,688,128 | 3,536 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_fb8725b_r2` | `2026-07-31T17:27:46Z` | 832,688,128 | 3,537 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_78c8e3d_r1` | `2026-07-31T17:33:33Z` | 832,716,800 | 3,540 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_dfe4d28_r1` | `2026-07-31T17:33:33Z` | 832,724,992 | 3,541 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_f8282c4_r1` | `2026-07-31T17:33:33Z` | 832,733,184 | 3,541 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_90f698a_r1` | `2026-07-31T18:46:20Z` | 832,757,760 | 3,541 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_4f5fc69_r1` | `2026-07-31T19:07:46Z` | 832,749,568 | 3,541 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_e341136_r1` | `2026-07-31T19:14:41Z` | 832,753,664 | 3,542 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_25383c2_r1` | `2026-07-31T19:57:44Z` | 832,757,760 | 3,542 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_c7b672a_r1` | `2026-07-31T20:12:55Z` | 832,765,952 | 3,542 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_9947374_r1` | `2026-07-31T20:21:13Z` | 832,761,856 | 3,542 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_smollm2_runtime_7881d8e_r1` | `2026-07-31T21:43:22Z` | 539,504,640 | 1,128 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_smollm2_runtime_7881d8e_r2` | `2026-07-31T21:49:21Z` | 832,815,104 | 3,547 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_smollm2_runtime_59734ed_r2` | `2026-07-31T22:23:04Z` | 832,827,392 | 3,547 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_smollm2_runtime_8071de7_r1` | `2026-08-01T06:10:16Z` | 832,827,392 | 3,547 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_smollm2_runtime_6255ce0_r1` | `2026-08-01T06:28:48Z` | 832,827,392 | 3,547 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_smollm2_runtime_e21a312_r1` | `2026-08-01T06:38:12Z` | 832,827,392 | 3,547 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_smollm2_runtime_bbaf9f3_r1` | `2026-08-01T06:52:33Z` | 832,839,680 | 3,547 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_8a68e4a_r2` | `2026-08-01T08:13:19Z` | 209,514,496 | 2,728 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_5dd10f0_r3` | `2026-08-01T08:15:27Z` | 209,522,688 | 2,729 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_5dd10f0_r4` | `2026-08-01T08:16:26Z` | 209,522,688 | 2,729 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_dfef0c2_r5` | `2026-08-01T08:20:51Z` | 209,518,592 | 2,729 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_45b5246_r6` | `2026-08-01T08:32:44Z` | 209,530,880 | 2,730 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_5a6e935_r1` | `2026-08-01T09:03:25Z` | 209,518,592 | 2,730 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_446411d_r1` | `2026-08-01T09:11:15Z` | 209,526,784 | 2,730 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_7f347ad_r1` | `2026-08-01T09:21:06Z` | 209,539,072 | 2,730 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_57e2713_r1` | `2026-08-01T09:30:37Z` | 209,530,880 | 2,730 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_45d0412_r1` | `2026-08-01T10:08:18Z` | 209,559,552 | 2,732 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_3f9add1_r1` | `2026-08-01T10:29:03Z` | 832,970,752 | 3,559 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_3f9add1_r3` | `2026-08-01T10:29:03Z` | 832,937,984 | 3,554 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_f174a58_r1` | `2026-08-01T10:29:03Z` | 832,937,984 | 3,554 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_3cda9ca_r1` | `2026-08-01T11:25:18Z` | 832,954,368 | 3,554 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_typed_query_runtime_88d7215_r2` | `2026-08-01T11:53:11Z` | 833,024,000 | 3,559 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_typed_query_runtime_8b84b10_r1` | `2026-08-01T12:15:26Z` | 833,024,000 | 3,559 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_algebraic_query_runtime_727fd04_r1` | `2026-08-01T12:35:15Z` | 832,647,168 | 3,560 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_algebraic_query_runtime_727fd04_r2` | `2026-08-01T12:36:29Z` | 833,056,768 | 3,561 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_algebraic_query_runtime_68599da_r3` | `2026-08-01T12:47:57Z` | 833,056,768 | 3,561 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_algebraic_query_runtime_562942f_r4` | `2026-08-01T12:52:11Z` | 833,060,864 | 3,561 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_algebraic_joint_runtime_7a83623_r1` | `2026-08-01T13:15:01Z` | 833,097,728 | 3,563 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_algebraic_joint_runtime_7a83623_r2` | `2026-08-01T13:15:55Z` | 833,089,536 | 3,563 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_algebraic_joint_runtime_7a83623_r3` | `2026-08-01T13:17:37Z` | 833,093,632 | 3,563 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_algebraic_joint_runtime_f27b1b8_r1` | `2026-08-01T13:20:35Z` | 833,093,632 | 3,563 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_algebraic_joint_runtime_3365d01_r1` | `2026-08-01T13:24:31Z` | 833,093,632 | 3,563 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_state_semantic_runtime_115c69b_r1` | `2026-08-01T13:35:40Z` | 833,118,208 | 3,565 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_state_semantic_runtime_938610c_r1` | `2026-08-01T13:45:39Z` | 833,118,208 | 3,565 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_algebraic_joint_runtime_cf875bc_r4` | `2026-08-01T14:07:56Z` | 327,950,336 | 1,881 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_algebraic_joint_runtime_cf875bc_r5` | `2026-08-01T14:07:56Z` | 833,134,592 | 3,566 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_state_owner_runtime_7e50a6d_r1` | `2026-08-01T14:20:10Z` | 833,146,880 | 3,567 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_state_owner_runtime_20d6394_r1` | `2026-08-01T14:26:47Z` | 537,690,112 | 2,022 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_state_owner_runtime_20d6394_r2` | `2026-08-01T14:26:47Z` | 835,633,152 | 3,637 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_state_owner_runtime_89b41c6_r1` | `2026-08-01T14:26:47Z` | 835,637,248 | 3,637 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_state_owner_runtime_a569c32_r1` | `2026-08-01T14:26:47Z` | 637,181,952 | 2,158 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_state_owner_runtime_a569c32_r2` | `2026-08-01T14:26:47Z` | 835,641,344 | 3,637 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_state_basis_runtime_3e0aecc_r1` | `2026-08-01T18:34:33Z` | 835,661,824 | 3,637 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_state_basis_runtime_bb65958_r2` | `2026-08-01T19:45:15Z` | 835,686,400 | 3,639 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_state_quotient_runtime_2d8c91d_r1` | `2026-08-01T20:00:53Z` | 835,702,784 | 3,640 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_parallel_runtime_b45c046_r3.partial.725237` | `2026-08-01T20:36:17Z` | 833,253,376 | 3,576 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_parallel_runtime_62082b4_r1` | `2026-08-01T21:14:41Z` | 835,788,800 | 3,647 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_cross_runtime_293545f_r1` | `2026-08-01T21:40:31Z` | 835,760,128 | 3,646 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_grounded_runtime_58f504d_r1` | `2026-08-01T22:21:47Z` | 836,005,888 | 3,663 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_semantic_prefix_runtime_82394b8_r1` | `2026-08-01T23:52:04Z` | 836,034,560 | 3,663 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_ensemble_runtime_03472e8_r1` | `2026-08-02T00:53:20Z` | 833,556,480 | 3,598 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_deployed_state_runtime_fb5a3bd_r1` | `2026-08-02T01:04:51Z` | 833,576,960 | 3,598 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_terminal_state_runtime_08cfaa4_r1` | `2026-08-02T01:48:57Z` | 833,662,976 | 3,606 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_terminal_state_runtime_08cfaa4_r2` | `2026-08-02T01:50:02Z` | 833,863,680 | 3,630 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_causal_delta_runtime_0fe3d5c_r1` | `2026-08-02T02:14:34Z` | 834,920,448 | 3,739 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_sparse_residual_runtime_a1e9c77_r1_REJECTED_INCOMPLETE` | `2026-08-02T02:32:50Z` | 834,916,352 | 3,739 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_sparse_residual_runtime_a1e9c77_r2` | `2026-08-02T02:35:25Z` | 835,293,184 | 3,760 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_sparse_residual_runtime_a543fd7_r3_REJECTED_INEXACT_SOURCE` | `2026-08-02T02:40:53Z` | 835,293,184 | 3,760 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_sparse_residual_runtime_a543fd7_r4` | `2026-08-02T02:45:00Z` | 835,297,280 | 3,760 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_atomic_typed_edit_runtime_fb368f2_r1` | `2026-08-02T03:10:08Z` | 835,604,480 | 3,771 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_dual_rail_lexical_command_runtime_c4fa270_r1` | `2026-08-02T03:53:11Z` | 836,935,680 | 3,792 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_token_native_router_runtime_8235fc9_r1` | `2026-08-02T04:21:15Z` | 836,960,256 | 3,795 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_occurrence_binding_runtime_9841b7d_r1.rejected-incomplete-20260802T0045` | `2026-08-02T04:45:06Z` | 671,576,064 | 3,050 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_occurrence_binding_runtime_9841b7d_r2.partial.rejected-truncated` | `2026-08-02T04:47:09Z` | 592,015,360 | 1,349 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_occurrence_binding_runtime_9841b7d_r3` | `2026-08-02T04:49:35Z` | 837,029,888 | 3,800 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_occurrence_schedule_runtime_60ecda3_r2.rejected-wrong-source-marker-20260802T0105` | `2026-08-02T05:03:18Z` | 837,050,368 | 3,801 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_occurrence_schedule_runtime_60ecda3_r3` | `2026-08-02T05:04:46Z` | 837,046,272 | 3,801 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_syntax_graph_runtime_00eefb9_r1` | `2026-08-02T05:40:24Z` | 837,169,152 | 3,810 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_program_audit_632bde0_r1` | `2026-08-02T06:15:07Z` | 32,768 | 6 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_program_audit_632bde0_r2` | `2026-08-02T06:16:55Z` | 32,768 | 6 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_program_audit_e868d3c_r3` | `2026-08-02T06:27:48Z` | 36,864 | 6 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_sticky_macro_runtime_960333f_r1` | `2026-08-02T06:53:46Z` | 837,971,968 | 3,829 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_opcode_projection_runtime_1a0cd1c_r2` | `2026-08-02T07:31:07Z` | 18,472,960 | 1,271 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_opcode_projection_runtime_1a0cd1c_r3` | `2026-08-02T07:37:01Z` | 836,734,976 | 3,808 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_effect_diagnostic_runtime_16455f3_r1` | `2026-08-02T13:27:05Z` | 4,096 | 1 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_effect_diagnostic_runtime_16455f3_r2` | `2026-08-02T13:27:57Z` | 467,746,816 | 1,393 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_effect_family_id_runtime_1ed1a60_r1` | `2026-08-02T17:21:08Z` | 667,025,408 | 3,084 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_family_state_runtime_2523e7e_r1` | `2026-08-02T18:30:19Z` | 667,549,696 | 3,095 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_family_state_runtime_f6e65ec_r1` | `2026-08-02T19:15:33Z` | 667,590,656 | 3,098 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_operation_family_campaign_admission_43084cd_r1` | `2026-08-02T20:37:06Z` | 2,846,720 | 5 | `sa305415`; nonsymlink directory; same-device | closed historical synthetic ETTR tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/quarantine_capability_floor_runtime_7730e7f_r1_bad_source_receipt` | `2026-08-02T22:44:45Z` | 544,702,464 | 1,402 | `sa305415`; nonsymlink directory; same-device | closed capability-floor or rejected staging tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/quarantine_capability_floor_runtime_7730e7f_r2_truncated_archive` | `2026-08-02T22:46:01Z` | 297,447,424 | 1,122 | `sa305415`; nonsymlink directory; same-device | closed capability-floor or rejected staging tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/quarantine_capability_floor_runtime_ebc8483_r1_bad_source_and_truncated_archive` | `2026-08-02T22:48:05Z` | 544,698,368 | 1,402 | `sa305415`; nonsymlink directory; same-device | closed capability-floor or rejected staging tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_neural_reranker_runtime.JG6Yn0` | `2026-08-04T13:36:00Z` | 397,312 | 34 | `sa305415`; nonsymlink directory; same-device | closed historical Shohin runtime tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_process_verifier_runtime.KhKI8o` | `2026-08-04T14:31:43Z` | 348,160 | 21 | `sa305415`; nonsymlink directory; same-device | closed historical Shohin runtime tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/code_selector_runtime_5fc7127_r1` | `2026-08-05T00:23:19Z` | 286,720 | 20 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/code_eval_runtime_824e2fa_r1` | `2026-08-05T00:36:57Z` | 335,872 | 24 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/code_eval_runtime_7889aa7_r1` | `2026-08-05T00:52:02Z` | 344,064 | 25 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/code_eval_runtime_daa1067_r1` | `2026-08-05T00:54:04Z` | 344,064 | 25 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/function_graph_runtime_4c8d2d1_r1` | `2026-08-05T01:31:33Z` | 86,016 | 7 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/function_graph_runtime_849c6d4_r1` | `2026-08-05T01:31:33Z` | 36,864 | 4 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/function_curriculum_runtime_16e6cc6_r1` | `2026-08-05T01:50:11Z` | 20,480 | 4 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/function_curriculum_runtime_9ec54d8_r1` | `2026-08-05T02:03:12Z` | 20,480 | 4 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/function_curriculum_runtime_912b8fb_r1` | `2026-08-05T02:06:00Z` | 24,576 | 4 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/function_graph_v2_runtime_ea52bf8_r1` | `2026-08-05T02:21:08Z` | 126,976 | 10 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/code_candidate_merge_runtime_6009e48_r1` | `2026-08-05T03:19:44Z` | 368,640 | 28 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/code_eval_runtime_9f00eb9_r1` | `2026-08-05T03:42:57Z` | 184,320 | 11 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/visible_code_repair_runtime_3321004_r2` | `2026-08-05T04:15:38Z` | 20,480 | 4 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/visible_code_repair_runtime_35b10f4_r1` | `2026-08-05T04:15:38Z` | 20,480 | 4 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/visible_code_repair_aggregate_cef5169_r1` | `2026-08-05T04:39:19Z` | 20,480 | 4 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/verified_code_repair_runtime_d20823d_r1` | `2026-08-05T04:45:32Z` | 28,672 | 4 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/model_failure_repair_runtime_d238755_r1` | `2026-08-05T04:51:57Z` | 73,728 | 8 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/code_teacher_runtime_9878ba3_r1` | `2026-08-05T05:21:46Z` | 339,968 | 24 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/code_teacher_runtime_2fa929f_r1` | `2026-08-05T05:30:00Z` | 339,968 | 24 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/model_failure_preference_runtime_f4b430f_r1` | `2026-08-05T05:39:27Z` | 24,576 | 5 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/code_preference_runtime_f4b430f_r1` | `2026-08-05T05:40:15Z` | 90,112 | 8 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/pcsd_runtime_a15e121` | `2026-08-05T07:39:28Z` | 90,112 | 13 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/fcpt_runtime_313ffea` | `2026-08-05T07:55:52Z` | 73,728 | 9 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/fcpt_runtime_650cc21` | `2026-08-05T08:01:18Z` | 77,824 | 9 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/cgsgr_runtime_fb60f2d` | `2026-08-05T08:10:23Z` | 106,496 | 11 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/qvesr_runtime_b0c6f25` | `2026-08-05T08:23:34Z` | 229,376 | 23 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/qvesr_runtime_e3d4999` | `2026-08-05T08:25:11Z` | 282,624 | 26 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/ceer_runtime_0e40f3f` | `2026-08-05T08:35:47Z` | 282,624 | 26 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/pcdl_runtime_39ece41` | `2026-08-05T08:54:09Z` | 106,496 | 12 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/pspa_runtime_8ea9d4b` | `2026-08-05T09:11:38Z` | 57,344 | 7 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/pspa_runtime_071229a` | `2026-08-05T09:15:35Z` | 61,440 | 7 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/learned_pspa_dev` | `2026-08-05T09:27:32Z` | 409,600 | 31 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/learned_pspa_runtime_8fb5d61` | `2026-08-05T09:33:28Z` | 90,112 | 9 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/dwpc_runtime_e99b4ef` | `2026-08-05T09:56:26Z` | 90,112 | 8 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/csdc_runtime_6121e2f` | `2026-08-05T10:05:50Z` | 102,400 | 9 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/csdc_semantic_runtime_152d42f` | `2026-08-05T10:21:08Z` | 139,264 | 12 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/csdc_copy_runtime_6359453` | `2026-08-05T10:40:36Z` | 159,744 | 13 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/csdc_smollm2_bridge_96217cb_r1` | `2026-08-05T11:45:55Z` | 212,992 | 14 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/csdc_smollm2_diag_2850b94_r1` | `2026-08-05T12:09:57Z` | 217,088 | 13 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/csdc_span_quotient_f1b91e9_r1` | `2026-08-05T17:51:23Z` | 203,706,368 | 2,668 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/csdc_span_quotient_405fcbb_r2` | `2026-08-05T17:52:42Z` | 203,706,368 | 2,668 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/csdc_span_quotient_e28f6d3_r3` | `2026-08-05T17:57:37Z` | 203,726,848 | 2,671 | `sa305415`; nonsymlink directory; same-device | closed code/CSDC development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_v0_smollm2_bd34385_r1` | `2026-08-05T21:00:09Z` | 192,512 | 11 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_v0_smollm2_bd34385_r2` | `2026-08-05T21:47:25Z` | 208,896 | 12 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_v0_smollm2_5241a56_audit_r1` | `2026-08-05T22:17:57Z` | 208,896 | 11 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_v0_role_copy_536f29c_r1` | `2026-08-05T22:26:04Z` | 233,472 | 13 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_v0_role_copy_e1bdf8b_r2` | `2026-08-05T22:27:31Z` | 233,472 | 13 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_v0_matched_gate_0249b2c_r2` | `2026-08-05T23:23:01Z` | 245,760 | 13 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_v0_matched_gate_0249b2c_r3` | `2026-08-05T23:24:15Z` | 290,816 | 15 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_v0_matched_gate_8e846c3_r4` | `2026-08-05T23:28:43Z` | 299,008 | 15 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_v0_matched_gate_fe2e5a8_r5` | `2026-08-05T23:45:12Z` | 339,968 | 17 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_v0_matched_gate_0de0afe_r6` | `2026-08-05T23:46:43Z` | 339,968 | 17 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_v0_matched_gate_56a2aa5_r7` | `2026-08-05T23:48:41Z` | 319,488 | 16 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_v0_matched_gate_e56a37f_r8` | `2026-08-05T23:51:08Z` | 319,488 | 16 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_sc1_e59fe33_r1` | `2026-08-06T00:21:51Z` | 147,456 | 17 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_sc1_e59fe33_r2` | `2026-08-06T00:22:04Z` | 118,784 | 10 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_sc1_d0af1d1_audit_r1` | `2026-08-06T00:59:49Z` | 143,360 | 13 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_wra1_4f07bdf_r1` | `2026-08-06T01:20:36Z` | 2,441,216 | 37 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_hsc1_d7360cd_r1` | `2026-08-06T02:21:12Z` | 28,672 | 7 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_hsc1_d7360cd_r2` | `2026-08-06T02:22:12Z` | 397,312 | 23 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_hsc1_fce7efb_r3` | `2026-08-06T02:24:51Z` | 581,632 | 28 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/diverge_ulc1_hsc1_2e7b326` | `2026-08-06T03:54:22Z` | 844,697,600 | 4,297 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/diverge_ulc1_hsc1_b4dc261` | `2026-08-06T03:54:22Z` | 844,697,600 | 4,297 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/diverge_ulc1_hsc1_fbc8623` | `2026-08-06T03:54:22Z` | 844,697,600 | 4,297 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/diverge_ulc1_hsc1_bf049df` | `2026-08-06T04:07:04Z` | 845,361,152 | 4,304 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_vmt1_b4f5766_r1` | `2026-08-06T08:22:15Z` | 225,280 | 23 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_vcr1_3224d1e_r1` | `2026-08-06T09:20:45Z` | 184,320 | 22 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_vcr1_20c94f4_r1` | `2026-08-06T09:22:07Z` | 184,320 | 22 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_vcr1_c5fe1d4_r1` | `2026-08-06T09:33:33Z` | 126,976 | 10 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_vcr1_c5fe1d4_r2` | `2026-08-06T09:35:33Z` | 139,264 | 11 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_vcr1_c5fe1d4_r3` | `2026-08-06T09:39:26Z` | 122,880 | 11 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_vcr1_620a8c3_eval_r1` | `2026-08-06T09:42:26Z` | 176,128 | 12 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_vcr1_7f38109_score_r1` | `2026-08-06T10:01:56Z` | 24,576 | 4 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_vcr1_7f38109_finalizer_r1` | `2026-08-06T10:14:22Z` | 20,480 | 5 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_vcr1_2c64021_finalizer_r1` | `2026-08-06T10:16:24Z` | 20,480 | 5 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_crp1_43d81d0_r1` | `2026-08-06T10:53:55Z` | 19,415,040 | 25 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_rsm1_f19c6dc_r1` | `2026-08-06T12:01:27Z` | 278,528 | 26 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_ats1_a9790ee_r1` | `2026-08-06T12:43:23Z` | 5,226,496 | 25 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_ats1_ecc07eb_r2` | `2026-08-06T12:45:21Z` | 5,169,152 | 22 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_ats1_207fbc1_eval_r3` | `2026-08-06T12:50:52Z` | 81,920 | 11 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_fta1_8eeb136_r1` | `2026-08-06T13:01:04Z` | 6,991,872 | 24 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_tol1_89622d5_r1` | `2026-08-06T13:57:52Z` | 167,936 | 24 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_iem1_eval_fb23f6b_r1` | `2026-08-06T19:54:40Z` | 106,496 | 11 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_sot1_baa50a1_r1` | `2026-08-06T20:44:37Z` | 222,588,928 | 2,906 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_sot1_eval_1b755f4_r1` | `2026-08-06T20:52:43Z` | 8,192 | 2 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_npw1_2ba2837_r1` | `2026-08-06T21:33:22Z` | 4,096 | 1 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_srp1_ce8b1c4_r1` | `2026-08-06T22:02:36Z` | 206,528,512 | 2,924 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_srp1_906b6f1_r1` | `2026-08-06T22:16:29Z` | 206,528,512 | 2,924 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_ccr1_ec6d2f3_r1` | `2026-08-07T00:00:44Z` | 207,429,632 | 2,973 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_ccr1_c2ee33f_r1` | `2026-08-07T00:04:42Z` | 207,429,632 | 2,973 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_ccr1_b0f0a6e_diag_r1` | `2026-08-07T00:38:38Z` | 206,712,832 | 2,945 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_nls1_eff7d06_r1` | `2026-08-07T14:52:02Z` | 2,555,904 | 28 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_ncp1_f84cabc_r1` | `2026-08-07T15:33:09Z` | 2,584,576 | 28 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_jrb1_1ab21c9_r1` | `2026-08-07T16:23:52Z` | 2,666,496 | 29 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_cab1_6820c49_r1` | `2026-08-07T16:57:31Z` | 24,805,376 | 1,438 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_oqb1_2ead185_r1` | `2026-08-07T17:20:38Z` | 22,577,152 | 1,441 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_sve1_bf7656f_r3` | `2026-08-07T17:49:24Z` | 38,621,184 | 2,416 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_snl1_2451988_r1` | `2026-08-07T18:49:44Z` | 851,664,896 | 4,827 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_qst1_0973e7e_r2` | `2026-08-07T20:02:26Z` | 159,744 | 13 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_qpt1_402fa8e_r1` | `2026-08-07T21:51:42Z` | 2,551,808 | 18 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_qpt1_controls_d50e262_r1` | `2026-08-07T22:21:31Z` | 2,560,000 | 18 | `sa305415`; nonsymlink directory; same-device | closed divergence tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/cvg1_rollouts_cb48f18_r1` | `2026-08-08T01:29:07Z` | 209,580,032 | 3,221 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/cvg1_rollouts_bc1d185_r1` | `2026-08-08T01:31:35Z` | 209,580,032 | 3,221 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/cvg1_rollouts_dd4ef87_r1` | `2026-08-08T01:43:53Z` | 209,604,608 | 3,223 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/cvg1_critic_a960206_r1` | `2026-08-08T02:59:45Z` | 159,744 | 11 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/cvg1_apply_88498a1_r1` | `2026-08-08T03:24:08Z` | 167,936 | 12 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/pcj1_beb127f_r1` | `2026-08-08T06:51:45Z` | 200,704 | 14 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/vcr1_8152cc8_r1` | `2026-08-08T07:35:15Z` | 225,280 | 16 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/sdr1_d11b231_r1` | `2026-08-08T08:31:19Z` | 241,664 | 17 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/idr1_drafts_e7b5be3_r1` | `2026-08-08T09:10:43Z` | 167,936 | 16 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/idr1_drafts_e7b5be3_r2` | `2026-08-08T09:10:54Z` | 143,360 | 10 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/idr1_ad790b0_r1` | `2026-08-08T09:18:24Z` | 253,952 | 20 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/idr1_d3fbff5_r1` | `2026-08-08T09:21:10Z` | 262,144 | 22 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/idr1_bc2bb10_r1` | `2026-08-08T10:30:36Z` | 262,144 | 22 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/idr1_55b6476_r1` | `2026-08-08T10:32:09Z` | 274,432 | 22 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/idr1_0ecf5f0_r1` | `2026-08-08T11:42:11Z` | 282,624 | 23 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/aqc1_192a7f7_r1` | `2026-08-08T13:34:36Z` | 327,680 | 27 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/aqc1_61da147_r1` | `2026-08-08T13:35:23Z` | 327,680 | 27 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/aqc1_38bc1f3_r1` | `2026-08-08T13:52:50Z` | 327,680 | 27 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/idr4_4b81c22_r1` | `2026-08-08T15:30:16Z` | 75,325,440 | 481 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/idr4_6dddeaf_r2` | `2026-08-08T15:33:09Z` | 2,625,536 | 23 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/idr4_6dddeaf_r3` | `2026-08-08T15:33:47Z` | 266,240 | 23 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/idr4_8891621_r1` | `2026-08-08T15:36:06Z` | 282,624 | 26 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/idr4_687a91a_r1` | `2026-08-08T15:37:36Z` | 290,816 | 27 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/idr4_d681daa_r1` | `2026-08-08T16:34:08Z` | 290,816 | 27 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/idr4_eval_54454d7_r1` | `2026-08-08T17:01:07Z` | 290,816 | 27 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/idr08_28c1246_r1` | `2026-08-08T20:22:34Z` | 315,392 | 32 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/idr08_28c1246_r2_incomplete_20260808T171128` | `2026-08-08T20:22:34Z` | 315,392 | 32 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/phase2_near_3ab418c_r1` | `2026-08-08T20:40:55Z` | 208,896 | 12 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/phase2_near_residual_3ab418c_r1` | `2026-08-08T20:43:03Z` | 249,856 | 14 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/tokenizer_compare_fbf81cf_r1` | `2026-08-08T20:51:31Z` | 20,480 | 4 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/phase2_retokenize_0f167fa_r1` | `2026-08-08T21:03:48Z` | 122,880 | 9 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/idr08_28c1246_r2` | `2026-08-08T21:11:28Z` | 339,968 | 33 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/checkpoint_normalize_1a01705_r1` | `2026-08-08T21:17:16Z` | 16,384 | 3 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/phase2_retokenize_3d6e140_r2` | `2026-08-08T21:19:04Z` | 126,976 | 9 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/phase2_retokenize_4abf367_r3` | `2026-08-08T21:23:44Z` | 126,976 | 9 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/phase2_retokenize_a481bc4_r4` | `2026-08-08T21:32:33Z` | 122,880 | 9 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtime/phase2_near49k_f9aeeef_r1` | `2026-08-08T21:45:41Z` | 204,800 | 13 | `sa305415`; nonsymlink directory; same-device | closed dense revision/development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/ttr1_ddba463_r1` | `2026-08-08T22:31:38Z` | 155,648 | 14 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/ttr1_8659590_r2` | `2026-08-08T22:33:38Z` | 335,872 | 31 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/ttr1_b3d7cb9_r3` | `2026-08-08T22:38:14Z` | 356,352 | 33 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/ttr1_6c1db22_r4` | `2026-08-08T22:41:19Z` | 364,544 | 34 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/ttr1_d0b7730_r5` | `2026-08-08T22:44:01Z` | 376,832 | 36 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/ttr1_5f4a83b_r6` | `2026-08-08T23:19:50Z` | 376,832 | 36 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/ttr1_bf58208_r7` | `2026-08-08T23:29:21Z` | 855,810,048 | 5,052 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/ttr1_4ebdfe8_r8` | `2026-08-08T23:37:00Z` | 855,830,528 | 5,054 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/ttr1_d03eafc_r9` | `2026-08-08T23:40:33Z` | 855,838,720 | 5,054 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/ttr1_7b5415d_r10` | `2026-08-09T00:11:55Z` | 855,834,624 | 5,054 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/ttr1_1c96116_r11` | `2026-08-09T00:14:45Z` | 855,830,528 | 5,054 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/sctr1_7ac0a94_r1` | `2026-08-09T00:57:41Z` | 210,575,360 | 3,354 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/sctr1_bbe3a7a_r2` | `2026-08-09T01:00:19Z` | 210,587,648 | 3,356 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/sctr1_0a92c76_r3` | `2026-08-09T01:03:05Z` | 210,595,840 | 3,357 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/sctr1_75f99ae_r5` | `2026-08-09T01:09:26Z` | 210,624,512 | 3,361 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/sctr1_1260cce_r6` | `2026-08-09T01:19:49Z` | 210,554,880 | 3,351 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/sctr1_f6f7f55_r7` | `2026-08-09T02:17:48Z` | 210,554,880 | 3,351 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/esr1_9253961_r8` | `2026-08-09T02:50:36Z` | 210,595,840 | 3,358 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/q36moe_e39a53f_r9` | `2026-08-09T03:14:18Z` | 210,608,128 | 3,360 | `sa305415`; nonsymlink directory; same-device | closed Q35 edit-cascade tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/q36kernels_761a972_r1` | `2026-08-09T03:29:30Z` | 20,480 | 5 | `sa305415`; nonsymlink directory; same-device | closed Q35 edit-cascade tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/q36moe_9eaeeff_r10` | `2026-08-09T03:30:52Z` | 210,612,224 | 3,361 | `sa305415`; nonsymlink directory; same-device | closed Q35 edit-cascade tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/q36kernels_5d4b09e_r2` | `2026-08-09T03:34:50Z` | 210,612,224 | 3,361 | `sa305415`; nonsymlink directory; same-device | closed Q35 edit-cascade tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/q36kernels_36e56f6_r3` | `2026-08-09T03:38:39Z` | 210,612,224 | 3,361 | `sa305415`; nonsymlink directory; same-device | closed Q35 edit-cascade tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/q36kernels_0381541_r4` | `2026-08-09T03:40:10Z` | 210,612,224 | 3,361 | `sa305415`; nonsymlink directory; same-device | closed Q35 edit-cascade tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/q36kernels_65b81e3_r5` | `2026-08-09T03:44:30Z` | 210,612,224 | 3,361 | `sa305415`; nonsymlink directory; same-device | closed Q35 edit-cascade tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/q36rollouts_4d74a59_r11` | `2026-08-09T03:52:04Z` | 210,616,320 | 3,361 | `sa305415`; nonsymlink directory; same-device | closed Q35 edit-cascade tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/q36steady_05fdbbe_r12` | `2026-08-09T03:57:13Z` | 210,616,320 | 3,361 | `sa305415`; nonsymlink directory; same-device | closed Q35 edit-cascade tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/mtr1_2bf7023_r2.failed_copy_1786249064` | `2026-08-09T04:11:23Z` | 216,023,040 | 3,370 | `sa305415`; nonsymlink directory; same-device | closed small-OLMoE tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/mtr1_6809591_r1` | `2026-08-09T04:11:23Z` | 216,023,040 | 3,370 | `sa305415`; nonsymlink directory; same-device | closed small-OLMoE tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/mtr1_2bf7023_r2` | `2026-08-09T04:17:50Z` | 216,027,136 | 3,371 | `sa305415`; nonsymlink directory; same-device | closed small-OLMoE tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/mtr1_32bc0df_r3` | `2026-08-09T04:21:28Z` | 216,039,424 | 3,373 | `sa305415`; nonsymlink directory; same-device | closed small-OLMoE tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/mtr1_6ee8506_r4` | `2026-08-09T04:29:24Z` | 216,068,096 | 3,378 | `sa305415`; nonsymlink directory; same-device | closed small-OLMoE tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/mtr1_592a35e_r5` | `2026-08-09T04:36:08Z` | 216,084,480 | 3,380 | `sa305415`; nonsymlink directory; same-device | closed small-OLMoE tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/mtr1_14275aa_r6` | `2026-08-09T04:39:08Z` | 216,088,576 | 3,380 | `sa305415`; nonsymlink directory; same-device | closed small-OLMoE tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/mtr1_14275aa_r7` | `2026-08-09T05:07:13Z` | 216,088,576 | 3,380 | `sa305415`; nonsymlink directory; same-device | closed small-OLMoE tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/mtr1_277e1ca_r8` | `2026-08-09T05:12:44Z` | 216,088,576 | 3,380 | `sa305415`; nonsymlink directory; same-device | closed small-OLMoE tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/rcr1_1c44a27_r1` | `2026-08-09T05:53:16Z` | 216,100,864 | 3,381 | `sa305415`; nonsymlink directory; same-device | closed small-OLMoE tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/rcr1_67ac03f_r2` | `2026-08-09T05:55:58Z` | 216,117,248 | 3,384 | `sa305415`; nonsymlink directory; same-device | closed small-OLMoE tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/mtr2_ffa7e06_r1` | `2026-08-09T06:16:11Z` | 216,117,248 | 3,384 | `sa305415`; nonsymlink directory; same-device | closed small-OLMoE tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/mtr2_750800e_r2` | `2026-08-09T06:17:10Z` | 216,117,248 | 3,384 | `sa305415`; nonsymlink directory; same-device | closed small-OLMoE tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/moeattr_cb0ce73_r1` | `2026-08-09T06:30:47Z` | 216,133,632 | 3,386 | `sa305415`; nonsymlink directory; same-device | closed small-OLMoE tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/drem1_e532105_r1_INVALID_INCOMPLETE_ARCHIVE` | `2026-08-09T06:52:46Z` | 299,970,560 | 1,145 | `sa305415`; nonsymlink directory; same-device | closed small-OLMoE tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/drem1_e532105_r2` | `2026-08-09T06:54:10Z` | 216,182,784 | 3,389 | `sa305415`; nonsymlink directory; same-device | closed small-OLMoE tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/ridr1_7cc2292_r1` | `2026-08-09T12:20:55Z` | 380,928 | 39 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/ridr1_7cc2292_r2` | `2026-08-09T12:21:10Z` | 311,296 | 22 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/kr2_236c8e0_r1` | `2026-08-09T12:42:26Z` | 352,256 | 27 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/kr2_87a37c8_r1` | `2026-08-09T12:46:03Z` | 364,544 | 29 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/kr2_56579a2_r1` | `2026-08-09T12:46:51Z` | 364,544 | 29 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/kr2_78b4715_r1` | `2026-08-09T12:48:08Z` | 364,544 | 29 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/kr2_8197da6_r1` | `2026-08-09T12:58:14Z` | 364,544 | 29 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/kr2_abdf2ac_r1` | `2026-08-09T12:58:55Z` | 364,544 | 29 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/tcs1_e25b858_r1` | `2026-08-09T13:28:04Z` | 40,960 | 5 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/tcs1_5c06b92_r1` | `2026-08-09T13:29:42Z` | 40,960 | 5 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/tcs1_semantic_9d0f109_r1` | `2026-08-09T13:35:51Z` | 389,120 | 31 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/vfr1_514ce86_r1` | `2026-08-09T14:13:03Z` | 211,337,216 | 3,451 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/vfr1_699ddf6_r1` | `2026-08-09T14:14:43Z` | 211,337,216 | 3,451 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/vfr1_3dc9d29_r1` | `2026-08-09T14:19:32Z` | 211,353,600 | 3,453 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/vfr1_671eca0_r1` | `2026-08-09T14:29:09Z` | 211,423,232 | 3,462 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/vfr1_1da7300_r1` | `2026-08-09T14:35:04Z` | 211,435,520 | 3,465 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/cfr1_6dc5eff_r1` | `2026-08-09T15:03:04Z` | 211,456,000 | 3,467 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/cfr1_fe80947_r2` | `2026-08-09T15:06:02Z` | 211,472,384 | 3,470 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/cfr1_cda70e5_r1` | `2026-08-09T15:17:01Z` | 211,488,768 | 3,471 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/cfr1_79b5272_r1` | `2026-08-09T15:18:50Z` | 211,492,864 | 3,472 | `sa305415`; nonsymlink directory; same-device | closed historical development tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/ndr1_c3ebdb3_r1` | `2026-08-09T17:17:08Z` | 211,521,536 | 3,475 | `sa305415`; nonsymlink directory; same-device | closed prohibited transaction-family tree |
| `/lustre/fs1/home/sa305415/shohin/runtimes/ndr1_31606c8_r1` | `2026-08-09T17:18:31Z` | 211,525,632 | 3,476 | `sa305415`; nonsymlink directory; same-device | closed prohibited transaction-family tree |
#### Batch B2b execution receipt

B2b was executed once against exactly the 402 literal rows recorded above; no
path was added, omitted, substituted, globbed, or widened to a parent root.
Immediately before mutation there were zero user jobs and all 20 protected
anchors were present. A complete fresh traversal revalidated, for every row,
the exact resolved realpath, recorded UTC root mtime, current allocated bytes
and path-inode count, immediate-child boundary, nonsymlink directory root,
ownership of every entry by `sa305415`, one-device boundary, and zero overlap
with every protected anchor. The complete hardlink replay also reproduced
`13,538` complete multi-link identities, all cross-target and none outside
the batch, for a conservative quota projection of
`233,581,510,656 bytes = 228,106,944 KiB / 700,823` inodes. The
pre-mutation Lustre baseline was
`1,067,026,292 KiB / 1,469,301` inodes.

Processing followed the recorded ascending chronology. From
`2026-08-11T09:30:25.876740Z` through
`2026-08-11T09:34:55.263951Z`, each original was atomically renamed to the
unique literal same-parent `.pcf1-age-delete-B2b-*` quarantine shown in the
transcript and only that quarantine was deleted with a one-filesystem
boundary. These 402 deletions are permanently nonrecoverable locally. The
transaction completed at `2026-08-11T09:34:55.404191Z`. Every original and
quarantine was then confirmed absent; all 20 protected anchors remained
present; and the scheduler remained empty. Post-run reconciliation found
indices `001` through `402` exactly once, in the authorized manifest order,
with 402 unique quarantine paths. Per-row values sum to the unchanged
preflight totals of `251,842,965,504` allocated bytes and `774,947` path
inodes.

The exact per-target deletion transcript follows. Its SHA-256, computed over
these 402 `DELETED_PERMANENT` lines with one terminating newline, is
`a3a58a160206a0f676cf2faad18234ef7dbb448871651bb4a097cdd1c68b907a`.

```text
DELETED_PERMANENT|001|2026-08-11T09:30:25.876740Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_batch_b31dd27|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-001-shohin_ettr_batch_b31dd27-20260811|13946880|842
DELETED_PERMANENT|002|2026-08-11T09:30:26.081885Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_fineweb_r2_49a5584|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-002-shohin_ettr_fineweb_r2_49a5584-20260811|13946880|842
DELETED_PERMANENT|003|2026-08-11T09:30:26.829862Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_scale_758d94d|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-003-shohin_ettr_scale_758d94d-20260811|190226432|2686
DELETED_PERMANENT|004|2026-08-11T09:30:26.839861Z|/lustre/fs1/home/sa305415/shohin/scratchpad/scale_pilot_719591_758d94d|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-004-scale_pilot_719591_758d94d-20260811|500465664|4
DELETED_PERMANENT|005|2026-08-11T09:30:26.849337Z|/lustre/fs1/home/sa305415/shohin/scratchpad/scale_pilot_719591_b32a4_758d94d|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-005-scale_pilot_719591_b32a4_758d94d-20260811|500465664|4
DELETED_PERMANENT|006|2026-08-11T09:30:27.118673Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_pes2o_r2_2b0d30c|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-006-shohin_ettr_pes2o_r2_2b0d30c-20260811|13959168|844
DELETED_PERMANENT|007|2026-08-11T09:30:27.558219Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_883875c0|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-007-shohin_ettr_runtime_883875c0-20260811|19369984|1847
DELETED_PERMANENT|008|2026-08-11T09:30:27.781460Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_97f19321|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-008-shohin_ettr_runtime_97f19321-20260811|15638528|930
DELETED_PERMANENT|009|2026-08-11T09:30:28.024259Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_524f3e50|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-009-shohin_ettr_runtime_524f3e50-20260811|15663104|934
DELETED_PERMANENT|010|2026-08-11T09:30:29.306915Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_finepdfs_c8acb46a|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-010-shohin_ettr_finepdfs_c8acb46a-20260811|830955520|3410
DELETED_PERMANENT|011|2026-08-11T09:30:29.320120Z|/lustre/fs1/home/sa305415/shohin/scratchpad/finepdfs_edu_eng_uniform16_9cfabe2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-011-finepdfs_edu_eng_uniform16_9cfabe2-20260811|47624335360|17
DELETED_PERMANENT|012|2026-08-11T09:30:29.338444Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_essential_runtime_e8d59eb|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-012-shohin_essential_runtime_e8d59eb-20260811|147456|10
DELETED_PERMANENT|013|2026-08-11T09:30:29.404027Z|/lustre/fs1/home/sa305415/shohin/scratchpad/essential_web_2024_38_uniform256_ce4eccc|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-013-essential_web_2024_38_uniform256_ce4eccc-20260811|65818554368|257
DELETED_PERMANENT|014|2026-08-11T09:30:29.418953Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_specialized_runtime_4d4d06d|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-014-shohin_specialized_runtime_4d4d06d-20260811|65536|7
DELETED_PERMANENT|015|2026-08-11T09:30:29.428971Z|/lustre/fs1/home/sa305415/shohin/scratchpad/nemotron_specialized_v1_1_13fa979_profile_sources|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-015-nemotron_specialized_v1_1_13fa979_profile_sources-20260811|281444352|5
DELETED_PERMANENT|016|2026-08-11T09:30:29.443254Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_formal_logic_runtime_976cdc6|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-016-shohin_formal_logic_runtime_976cdc6-20260811|77824|6
DELETED_PERMANENT|017|2026-08-11T09:30:29.452672Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_transfer_74c4fe3|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-017-shohin_ettr_transfer_74c4fe3-20260811|1466368|4
DELETED_PERMANENT|018|2026-08-11T09:30:29.462391Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_fast_085f7a8|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-018-shohin_ettr_fast_085f7a8-20260811|28672|4
DELETED_PERMANENT|019|2026-08-11T09:30:29.472587Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_assembly_d170f25|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-019-shohin_ettr_assembly_d170f25-20260811|45056|5
DELETED_PERMANENT|020|2026-08-11T09:30:29.483255Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_assembly_8ae4eca|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-020-shohin_ettr_assembly_8ae4eca-20260811|45056|5
DELETED_PERMANENT|021|2026-08-11T09:30:30.575963Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_publish_b1c617d.partial|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-021-shohin_ettr_publish_b1c617d.partial-20260811|823877632|2960
DELETED_PERMANENT|022|2026-08-11T09:30:30.787066Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_publish_b1c617d_lite|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-022-shohin_ettr_publish_b1c617d_lite-20260811|14553088|893
DELETED_PERMANENT|023|2026-08-11T09:30:32.205388Z|/lustre/fs1/home/sa305415/shohin/scratchpad/ettr_release_source_35333c3_full_clean|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-023-ettr_release_source_35333c3_full_clean-20260811|1028042752|3507
DELETED_PERMANENT|024|2026-08-11T09:30:33.486454Z|/lustre/fs1/home/sa305415/shohin/scratchpad/ettr_release_source_81e3184_full_clean|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-024-ettr_release_source_81e3184_full_clean-20260811|1027215360|3500
DELETED_PERMANENT|025|2026-08-11T09:30:34.759173Z|/lustre/fs1/home/sa305415/shohin/scratchpad/ettr_release_source_d9d77af_full_clean|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-025-ettr_release_source_d9d77af_full_clean-20260811|1027940352|3504
DELETED_PERMANENT|026|2026-08-11T09:30:34.775335Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_68bf17f|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-026-shohin_ettr_runtime_68bf17f-20260811|135168|10
DELETED_PERMANENT|027|2026-08-11T09:30:35.102869Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_68bf17f_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-027-shohin_ettr_runtime_68bf17f_r1-20260811|15667200|934
DELETED_PERMANENT|028|2026-08-11T09:30:35.517484Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_1afc3b4_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-028-shohin_ettr_runtime_1afc3b4_full_r1-20260811|31326208|1735
DELETED_PERMANENT|029|2026-08-11T09:30:36.804606Z|/lustre/fs1/home/sa305415/shohin/scratchpad/ettr_packet_audit_source_fbf12df_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-029-ettr_packet_audit_source_fbf12df_full_r1-20260811|1028972544|3530
DELETED_PERMANENT|030|2026-08-11T09:30:38.228619Z|/lustre/fs1/home/sa305415/shohin/scratchpad/ettr_materializer_source_e5f3705_packet_v2_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-030-ettr_materializer_source_e5f3705_packet_v2_r1-20260811|1030692864|3551
DELETED_PERMANENT|031|2026-08-11T09:30:38.238648Z|/lustre/fs1/home/sa305415/shohin/scratchpad/ettr_materializer_adapter_e5f3705_packet_v2_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-031-ettr_materializer_adapter_e5f3705_packet_v2_r1-20260811|24576|3
DELETED_PERMANENT|032|2026-08-11T09:30:38.248039Z|/lustre/fs1/home/sa305415/shohin/scratchpad/ettr_materializer_assembly_e5f3705_packet_v2_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-032-ettr_materializer_assembly_e5f3705_packet_v2_r1-20260811|24576|3
DELETED_PERMANENT|033|2026-08-11T09:30:38.923056Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_e5f3705_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-033-shohin_ettr_runtime_e5f3705_full_r1-20260811|200826880|2296
DELETED_PERMANENT|034|2026-08-11T09:30:38.939895Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_confirm_fast_3106202_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-034-shohin_ettr_confirm_fast_3106202_r1-20260811|69632|14
DELETED_PERMANENT|035|2026-08-11T09:30:38.956559Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_confirm_assembly_b614b30_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-035-shohin_ettr_confirm_assembly_b614b30_r1-20260811|94208|16
DELETED_PERMANENT|036|2026-08-11T09:30:38.971721Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_ladder_c9ede03_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-036-shohin_ettr_ladder_c9ede03_r1-20260811|40960|9
DELETED_PERMANENT|037|2026-08-11T09:30:38.987210Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_eager_e660153_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-037-shohin_ettr_eager_e660153_r1-20260811|49152|9
DELETED_PERMANENT|038|2026-08-11T09:30:39.000741Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_diag_077f57b_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-038-shohin_ettr_diag_077f57b_r1-20260811|28672|6
DELETED_PERMANENT|039|2026-08-11T09:30:39.788888Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_07e7f63_full_r1.partial-fork-failed-20260730T0806|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-039-shohin_ettr_runtime_07e7f63_full_r1.partial-fork-failed-20260730T0806-20260811|199299072|2276
DELETED_PERMANENT|040|2026-08-11T09:30:40.514334Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_07e7f63_full_r2.partial-no-pytest-20260730T0808|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-040-shohin_ettr_runtime_07e7f63_full_r2.partial-no-pytest-20260730T0808-20260811|199544832|2277
DELETED_PERMANENT|041|2026-08-11T09:30:41.295716Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_07e7f63_full_r3|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-041-shohin_ettr_runtime_07e7f63_full_r3-20260811|201297920|2324
DELETED_PERMANENT|042|2026-08-11T09:30:41.989602Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_25d84f0_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-042-shohin_ettr_runtime_25d84f0_full_r1-20260811|201297920|2324
DELETED_PERMANENT|043|2026-08-11T09:30:43.342688Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_02e8c77_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-043-shohin_ettr_runtime_02e8c77_full_r1-20260811|831901696|3492
DELETED_PERMANENT|044|2026-08-11T09:30:44.626220Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_132d983_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-044-shohin_ettr_runtime_132d983_full_r1-20260811|831909888|3492
DELETED_PERMANENT|045|2026-08-11T09:30:45.868446Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_deea68c_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-045-shohin_ettr_runtime_deea68c_full_r1-20260811|831913984|3492
DELETED_PERMANENT|046|2026-08-11T09:30:47.152554Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_f644406_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-046-shohin_ettr_runtime_f644406_full_r1-20260811|831913984|3492
DELETED_PERMANENT|047|2026-08-11T09:30:47.847032Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_4232338_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-047-shohin_ettr_runtime_4232338_full_r1-20260811|199593984|2278
DELETED_PERMANENT|048|2026-08-11T09:30:48.577577Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_99eda96_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-048-shohin_ettr_runtime_99eda96_full_r1-20260811|199598080|2278
DELETED_PERMANENT|049|2026-08-11T09:30:49.300856Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_2926987_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-049-shohin_ettr_runtime_2926987_full_r1-20260811|199622656|2280
DELETED_PERMANENT|050|2026-08-11T09:30:49.982437Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_43005ad_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-050-shohin_ettr_runtime_43005ad_full_r1-20260811|199622656|2280
DELETED_PERMANENT|051|2026-08-11T09:30:50.680859Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_aeb4aff_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-051-shohin_ettr_runtime_aeb4aff_full_r1-20260811|199626752|2280
DELETED_PERMANENT|052|2026-08-11T09:30:51.931553Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_b7df632_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-052-shohin_ettr_runtime_b7df632_full_r1-20260811|832045056|3498
DELETED_PERMANENT|053|2026-08-11T09:30:53.172548Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_b7df632_full_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-053-shohin_ettr_runtime_b7df632_full_r2-20260811|832040960|3498
DELETED_PERMANENT|054|2026-08-11T09:30:54.390218Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_70f8bb0_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-054-shohin_ettr_runtime_70f8bb0_full_r1-20260811|832053248|3498
DELETED_PERMANENT|055|2026-08-11T09:30:55.709804Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_309cb02_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-055-shohin_ettr_runtime_309cb02_full_r1-20260811|833810432|3544
DELETED_PERMANENT|056|2026-08-11T09:30:56.994009Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_0b3c8bb_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-056-shohin_ettr_runtime_0b3c8bb_full_r1-20260811|833810432|3544
DELETED_PERMANENT|057|2026-08-11T09:30:58.338632Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_66eafea_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-057-shohin_ettr_runtime_66eafea_full_r1-20260811|833814528|3544
DELETED_PERMANENT|058|2026-08-11T09:30:59.567071Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_7412a0f_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-058-shohin_ettr_runtime_7412a0f_full_r1-20260811|832061440|3498
DELETED_PERMANENT|059|2026-08-11T09:31:00.857459Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_7d7695b_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-059-shohin_ettr_runtime_7d7695b_full_r1-20260811|832065536|3498
DELETED_PERMANENT|060|2026-08-11T09:31:02.179305Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_3da6f44_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-060-shohin_ettr_runtime_3da6f44_full_r1-20260811|832094208|3500
DELETED_PERMANENT|061|2026-08-11T09:31:03.464036Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_821351b_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-061-shohin_ettr_runtime_821351b_full_r1-20260811|832131072|3502
DELETED_PERMANENT|062|2026-08-11T09:31:04.711853Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_0cc7e05_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-062-shohin_ettr_runtime_0cc7e05_full_r1-20260811|832131072|3502
DELETED_PERMANENT|063|2026-08-11T09:31:06.089858Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_c53fa8d_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-063-shohin_ettr_runtime_c53fa8d_full_r1-20260811|832135168|3502
DELETED_PERMANENT|064|2026-08-11T09:31:07.345979Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_b2fc0ba_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-064-shohin_ettr_runtime_b2fc0ba_full_r1-20260811|832135168|3502
DELETED_PERMANENT|065|2026-08-11T09:31:08.617513Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_68c8780_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-065-shohin_ettr_runtime_68c8780_full_r1-20260811|832180224|3504
DELETED_PERMANENT|066|2026-08-11T09:31:09.872577Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_634dc76_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-066-shohin_ettr_runtime_634dc76_full_r1-20260811|831795200|3503
DELETED_PERMANENT|067|2026-08-11T09:31:11.112571Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_634dc76_full_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-067-shohin_ettr_runtime_634dc76_full_r2-20260811|832196608|3504
DELETED_PERMANENT|068|2026-08-11T09:31:12.411715Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_88336a5_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-068-shohin_ettr_runtime_88336a5_full_r1-20260811|831803392|3503
DELETED_PERMANENT|069|2026-08-11T09:31:13.702270Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_88336a5_full_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-069-shohin_ettr_runtime_88336a5_full_r2-20260811|832200704|3504
DELETED_PERMANENT|070|2026-08-11T09:31:14.947842Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_dev_anchor_20260730_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-070-shohin_ettr_dev_anchor_20260730_r1-20260811|836669440|3636
DELETED_PERMANENT|071|2026-08-11T09:31:16.275232Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_92f8f0a_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-071-shohin_ettr_runtime_92f8f0a_full_r1-20260811|831803392|3503
DELETED_PERMANENT|072|2026-08-11T09:31:17.564574Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_92f8f0a_full_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-072-shohin_ettr_runtime_92f8f0a_full_r2-20260811|832200704|3504
DELETED_PERMANENT|073|2026-08-11T09:31:18.823856Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_b315875_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-073-shohin_ettr_runtime_b315875_full_r1-20260811|832208896|3504
DELETED_PERMANENT|074|2026-08-11T09:31:20.172522Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_549c7bd_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-074-shohin_ettr_runtime_549c7bd_full_r1-20260811|832200704|3504
DELETED_PERMANENT|075|2026-08-11T09:31:21.455590Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_47c4fb2_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-075-shohin_ettr_runtime_47c4fb2_full_r1-20260811|832241664|3506
DELETED_PERMANENT|076|2026-08-11T09:31:22.700413Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_096f703_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-076-shohin_ettr_runtime_096f703_full_r1-20260811|832282624|3512
DELETED_PERMANENT|077|2026-08-11T09:31:23.977487Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_29b393d_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-077-shohin_ettr_runtime_29b393d_full_r1-20260811|832294912|3512
DELETED_PERMANENT|078|2026-08-11T09:31:25.328549Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_runtime_b981eb4_full_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-078-shohin_ettr_runtime_b981eb4_full_r1-20260811|832303104|3512
DELETED_PERMANENT|079|2026-08-11T09:31:25.797104Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_22ed5e1_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-079-shohin_ettr_joint_runtime_22ed5e1_r1-20260811|31383552|1971
DELETED_PERMANENT|080|2026-08-11T09:31:26.321193Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_9ea7981_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-080-shohin_ettr_joint_runtime_9ea7981_r1-20260811|31395840|1972
DELETED_PERMANENT|081|2026-08-11T09:31:27.611616Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_89fc0ad_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-081-shohin_ettr_joint_runtime_89fc0ad_r1-20260811|832286720|3532
DELETED_PERMANENT|082|2026-08-11T09:31:28.847734Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_89fc0ad_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-082-shohin_ettr_joint_runtime_89fc0ad_r2-20260811|832552960|3532
DELETED_PERMANENT|083|2026-08-11T09:31:28.856211Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_fd78cbd_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-083-shohin_ettr_joint_runtime_fd78cbd_r1-20260811|4096|1
DELETED_PERMANENT|084|2026-08-11T09:31:30.167225Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_fd78cbd_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-084-shohin_ettr_joint_runtime_fd78cbd_r2-20260811|832565248|3532
DELETED_PERMANENT|085|2026-08-11T09:31:32.167575Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_479472d_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-085-shohin_ettr_joint_runtime_479472d_r2-20260811|847343616|7065
DELETED_PERMANENT|086|2026-08-11T09:31:33.633458Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_a13caf3_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-086-shohin_ettr_joint_runtime_a13caf3_r1-20260811|832905216|3535
DELETED_PERMANENT|087|2026-08-11T09:31:35.123233Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_47008dc_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-087-shohin_ettr_joint_runtime_47008dc_r1-20260811|832614400|3535
DELETED_PERMANENT|088|2026-08-11T09:31:36.571080Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_9e1b107_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-088-shohin_ettr_joint_runtime_9e1b107_r1-20260811|832622592|3535
DELETED_PERMANENT|089|2026-08-11T09:31:38.005954Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_dbe27ef_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-089-shohin_ettr_joint_runtime_dbe27ef_r1-20260811|832622592|3535
DELETED_PERMANENT|090|2026-08-11T09:31:39.612338Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_e1f8e3f_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-090-shohin_ettr_joint_runtime_e1f8e3f_r1-20260811|832626688|3535
DELETED_PERMANENT|091|2026-08-11T09:31:40.752078Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_e44976b_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-091-shohin_ettr_joint_runtime_e44976b_r1-20260811|832622592|3535
DELETED_PERMANENT|092|2026-08-11T09:31:42.210401Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_743ece1_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-092-shohin_ettr_joint_runtime_743ece1_r1-20260811|832626688|3535
DELETED_PERMANENT|093|2026-08-11T09:31:43.659057Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_42c3456_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-093-shohin_ettr_joint_runtime_42c3456_r1-20260811|832626688|3535
DELETED_PERMANENT|094|2026-08-11T09:31:45.106243Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_de79470_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-094-shohin_ettr_joint_runtime_de79470_r1-20260811|832630784|3535
DELETED_PERMANENT|095|2026-08-11T09:31:46.576576Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_685f685_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-095-shohin_ettr_joint_runtime_685f685_r1-20260811|832630784|3535
DELETED_PERMANENT|096|2026-08-11T09:31:48.078600Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_a630392_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-096-shohin_ettr_joint_runtime_a630392_r1-20260811|832634880|3535
DELETED_PERMANENT|097|2026-08-11T09:31:49.507991Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_0a6e4e8_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-097-shohin_ettr_joint_runtime_0a6e4e8_r1-20260811|832655360|3535
DELETED_PERMANENT|098|2026-08-11T09:31:50.959157Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_e818e8d_rejected_wrong_source_commit|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-098-shohin_ettr_joint_runtime_e818e8d_rejected_wrong_source_commit-20260811|832655360|3535
DELETED_PERMANENT|099|2026-08-11T09:31:52.529339Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_e818e8d_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-099-shohin_ettr_joint_runtime_e818e8d_r2-20260811|832655360|3535
DELETED_PERMANENT|100|2026-08-11T09:31:53.763837Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_0bba846_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-100-shohin_ettr_joint_runtime_0bba846_r1-20260811|832667648|3535
DELETED_PERMANENT|101|2026-08-11T09:31:55.002121Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_f529fdb_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-101-shohin_ettr_joint_runtime_f529fdb_r2-20260811|832675840|3535
DELETED_PERMANENT|102|2026-08-11T09:31:56.328513Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_ac55889_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-102-shohin_ettr_joint_runtime_ac55889_r1-20260811|832688128|3536
DELETED_PERMANENT|103|2026-08-11T09:31:57.663587Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_3ba93f2_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-103-shohin_ettr_joint_runtime_3ba93f2_r1-20260811|832688128|3536
DELETED_PERMANENT|104|2026-08-11T09:31:58.957919Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_fb8725b_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-104-shohin_ettr_joint_runtime_fb8725b_r1-20260811|832688128|3536
DELETED_PERMANENT|105|2026-08-11T09:32:00.257334Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_fb8725b_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-105-shohin_ettr_joint_runtime_fb8725b_r2-20260811|832688128|3537
DELETED_PERMANENT|106|2026-08-11T09:32:01.509758Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_78c8e3d_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-106-shohin_ettr_joint_runtime_78c8e3d_r1-20260811|832716800|3540
DELETED_PERMANENT|107|2026-08-11T09:32:02.795137Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_dfe4d28_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-107-shohin_ettr_joint_runtime_dfe4d28_r1-20260811|832724992|3541
DELETED_PERMANENT|108|2026-08-11T09:32:04.031894Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_f8282c4_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-108-shohin_ettr_joint_runtime_f8282c4_r1-20260811|832733184|3541
DELETED_PERMANENT|109|2026-08-11T09:32:05.241391Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_90f698a_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-109-shohin_ettr_joint_runtime_90f698a_r1-20260811|832757760|3541
DELETED_PERMANENT|110|2026-08-11T09:32:06.481757Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_4f5fc69_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-110-shohin_ettr_joint_runtime_4f5fc69_r1-20260811|832749568|3541
DELETED_PERMANENT|111|2026-08-11T09:32:07.765530Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_e341136_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-111-shohin_ettr_joint_runtime_e341136_r1-20260811|832753664|3542
DELETED_PERMANENT|112|2026-08-11T09:32:09.041537Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_25383c2_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-112-shohin_ettr_joint_runtime_25383c2_r1-20260811|832757760|3542
DELETED_PERMANENT|113|2026-08-11T09:32:10.344601Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_c7b672a_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-113-shohin_ettr_joint_runtime_c7b672a_r1-20260811|832765952|3542
DELETED_PERMANENT|114|2026-08-11T09:32:11.634382Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_joint_runtime_9947374_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-114-shohin_ettr_joint_runtime_9947374_r1-20260811|832761856|3542
DELETED_PERMANENT|115|2026-08-11T09:32:12.178279Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_smollm2_runtime_7881d8e_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-115-shohin_ettr_smollm2_runtime_7881d8e_r1-20260811|539504640|1128
DELETED_PERMANENT|116|2026-08-11T09:32:13.461275Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_smollm2_runtime_7881d8e_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-116-shohin_ettr_smollm2_runtime_7881d8e_r2-20260811|832815104|3547
DELETED_PERMANENT|117|2026-08-11T09:32:14.683901Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_smollm2_runtime_59734ed_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-117-shohin_ettr_smollm2_runtime_59734ed_r2-20260811|832827392|3547
DELETED_PERMANENT|118|2026-08-11T09:32:15.917500Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_smollm2_runtime_8071de7_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-118-shohin_ettr_smollm2_runtime_8071de7_r1-20260811|832827392|3547
DELETED_PERMANENT|119|2026-08-11T09:32:17.257840Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_smollm2_runtime_6255ce0_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-119-shohin_ettr_smollm2_runtime_6255ce0_r1-20260811|832827392|3547
DELETED_PERMANENT|120|2026-08-11T09:32:18.582364Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_smollm2_runtime_e21a312_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-120-shohin_ettr_smollm2_runtime_e21a312_r1-20260811|832827392|3547
DELETED_PERMANENT|121|2026-08-11T09:32:19.892002Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_smollm2_runtime_bbaf9f3_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-121-shohin_ettr_smollm2_runtime_bbaf9f3_r1-20260811|832839680|3547
DELETED_PERMANENT|122|2026-08-11T09:32:20.765218Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_8a68e4a_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-122-shohin_ettr_native_disposition_runtime_8a68e4a_r2-20260811|209514496|2728
DELETED_PERMANENT|123|2026-08-11T09:32:21.673295Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_5dd10f0_r3|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-123-shohin_ettr_native_disposition_runtime_5dd10f0_r3-20260811|209522688|2729
DELETED_PERMANENT|124|2026-08-11T09:32:22.536459Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_5dd10f0_r4|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-124-shohin_ettr_native_disposition_runtime_5dd10f0_r4-20260811|209522688|2729
DELETED_PERMANENT|125|2026-08-11T09:32:23.373251Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_dfef0c2_r5|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-125-shohin_ettr_native_disposition_runtime_dfef0c2_r5-20260811|209518592|2729
DELETED_PERMANENT|126|2026-08-11T09:32:24.191496Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_45b5246_r6|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-126-shohin_ettr_native_disposition_runtime_45b5246_r6-20260811|209530880|2730
DELETED_PERMANENT|127|2026-08-11T09:32:24.976530Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_5a6e935_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-127-shohin_ettr_native_disposition_runtime_5a6e935_r1-20260811|209518592|2730
DELETED_PERMANENT|128|2026-08-11T09:32:25.852394Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_446411d_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-128-shohin_ettr_native_disposition_runtime_446411d_r1-20260811|209526784|2730
DELETED_PERMANENT|129|2026-08-11T09:32:26.684694Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_7f347ad_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-129-shohin_ettr_native_disposition_runtime_7f347ad_r1-20260811|209539072|2730
DELETED_PERMANENT|130|2026-08-11T09:32:27.607680Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_57e2713_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-130-shohin_ettr_native_disposition_runtime_57e2713_r1-20260811|209530880|2730
DELETED_PERMANENT|131|2026-08-11T09:32:28.405952Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_45d0412_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-131-shohin_ettr_native_disposition_runtime_45d0412_r1-20260811|209559552|2732
DELETED_PERMANENT|132|2026-08-11T09:32:29.664605Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_3f9add1_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-132-shohin_ettr_native_disposition_runtime_3f9add1_r1-20260811|832970752|3559
DELETED_PERMANENT|133|2026-08-11T09:32:30.928035Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_3f9add1_r3|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-133-shohin_ettr_native_disposition_runtime_3f9add1_r3-20260811|832937984|3554
DELETED_PERMANENT|134|2026-08-11T09:32:32.260976Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_f174a58_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-134-shohin_ettr_native_disposition_runtime_f174a58_r1-20260811|832937984|3554
DELETED_PERMANENT|135|2026-08-11T09:32:33.483998Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_native_disposition_runtime_3cda9ca_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-135-shohin_ettr_native_disposition_runtime_3cda9ca_r1-20260811|832954368|3554
DELETED_PERMANENT|136|2026-08-11T09:32:34.799597Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_typed_query_runtime_88d7215_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-136-shohin_ettr_typed_query_runtime_88d7215_r2-20260811|833024000|3559
DELETED_PERMANENT|137|2026-08-11T09:32:36.173154Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_typed_query_runtime_8b84b10_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-137-shohin_ettr_typed_query_runtime_8b84b10_r1-20260811|833024000|3559
DELETED_PERMANENT|138|2026-08-11T09:32:37.437515Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_algebraic_query_runtime_727fd04_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-138-shohin_ettr_algebraic_query_runtime_727fd04_r1-20260811|832647168|3560
DELETED_PERMANENT|139|2026-08-11T09:32:38.695898Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_algebraic_query_runtime_727fd04_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-139-shohin_ettr_algebraic_query_runtime_727fd04_r2-20260811|833056768|3561
DELETED_PERMANENT|140|2026-08-11T09:32:39.918005Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_algebraic_query_runtime_68599da_r3|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-140-shohin_ettr_algebraic_query_runtime_68599da_r3-20260811|833056768|3561
DELETED_PERMANENT|141|2026-08-11T09:32:41.257743Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_algebraic_query_runtime_562942f_r4|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-141-shohin_ettr_algebraic_query_runtime_562942f_r4-20260811|833060864|3561
DELETED_PERMANENT|142|2026-08-11T09:32:42.469831Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_algebraic_joint_runtime_7a83623_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-142-shohin_ettr_algebraic_joint_runtime_7a83623_r1-20260811|833097728|3563
DELETED_PERMANENT|143|2026-08-11T09:32:43.692290Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_algebraic_joint_runtime_7a83623_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-143-shohin_ettr_algebraic_joint_runtime_7a83623_r2-20260811|833089536|3563
DELETED_PERMANENT|144|2026-08-11T09:32:44.792611Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_algebraic_joint_runtime_7a83623_r3|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-144-shohin_ettr_algebraic_joint_runtime_7a83623_r3-20260811|833093632|3563
DELETED_PERMANENT|145|2026-08-11T09:32:46.233057Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_algebraic_joint_runtime_f27b1b8_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-145-shohin_ettr_algebraic_joint_runtime_f27b1b8_r1-20260811|833093632|3563
DELETED_PERMANENT|146|2026-08-11T09:32:47.688535Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_algebraic_joint_runtime_3365d01_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-146-shohin_ettr_algebraic_joint_runtime_3365d01_r1-20260811|833093632|3563
DELETED_PERMANENT|147|2026-08-11T09:32:49.169615Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_state_semantic_runtime_115c69b_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-147-shohin_ettr_state_semantic_runtime_115c69b_r1-20260811|833118208|3565
DELETED_PERMANENT|148|2026-08-11T09:32:50.741077Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_state_semantic_runtime_938610c_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-148-shohin_ettr_state_semantic_runtime_938610c_r1-20260811|833118208|3565
DELETED_PERMANENT|149|2026-08-11T09:32:51.399604Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_algebraic_joint_runtime_cf875bc_r4|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-149-shohin_ettr_algebraic_joint_runtime_cf875bc_r4-20260811|327950336|1881
DELETED_PERMANENT|150|2026-08-11T09:32:52.679410Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_algebraic_joint_runtime_cf875bc_r5|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-150-shohin_ettr_algebraic_joint_runtime_cf875bc_r5-20260811|833134592|3566
DELETED_PERMANENT|151|2026-08-11T09:32:53.946789Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_state_owner_runtime_7e50a6d_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-151-shohin_ettr_state_owner_runtime_7e50a6d_r1-20260811|833146880|3567
DELETED_PERMANENT|152|2026-08-11T09:32:54.664582Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_state_owner_runtime_20d6394_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-152-shohin_ettr_state_owner_runtime_20d6394_r1-20260811|537690112|2022
DELETED_PERMANENT|153|2026-08-11T09:32:55.950022Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_state_owner_runtime_20d6394_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-153-shohin_ettr_state_owner_runtime_20d6394_r2-20260811|835633152|3637
DELETED_PERMANENT|154|2026-08-11T09:32:57.346723Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_state_owner_runtime_89b41c6_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-154-shohin_ettr_state_owner_runtime_89b41c6_r1-20260811|835637248|3637
DELETED_PERMANENT|155|2026-08-11T09:32:58.174794Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_state_owner_runtime_a569c32_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-155-shohin_ettr_state_owner_runtime_a569c32_r1-20260811|637181952|2158
DELETED_PERMANENT|156|2026-08-11T09:32:59.456625Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_state_owner_runtime_a569c32_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-156-shohin_ettr_state_owner_runtime_a569c32_r2-20260811|835641344|3637
DELETED_PERMANENT|157|2026-08-11T09:33:00.696069Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_state_basis_runtime_3e0aecc_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-157-shohin_ettr_state_basis_runtime_3e0aecc_r1-20260811|835661824|3637
DELETED_PERMANENT|158|2026-08-11T09:33:02.116861Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_state_basis_runtime_bb65958_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-158-shohin_ettr_state_basis_runtime_bb65958_r2-20260811|835686400|3639
DELETED_PERMANENT|159|2026-08-11T09:33:03.356591Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_state_quotient_runtime_2d8c91d_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-159-shohin_ettr_state_quotient_runtime_2d8c91d_r1-20260811|835702784|3640
DELETED_PERMANENT|160|2026-08-11T09:33:04.613860Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_parallel_runtime_b45c046_r3.partial.725237|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-160-shohin_ettr_parallel_runtime_b45c046_r3.partial.725237-20260811|833253376|3576
DELETED_PERMANENT|161|2026-08-11T09:33:05.883861Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_parallel_runtime_62082b4_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-161-shohin_ettr_parallel_runtime_62082b4_r1-20260811|835788800|3647
DELETED_PERMANENT|162|2026-08-11T09:33:07.130207Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_cross_runtime_293545f_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-162-shohin_ettr_cross_runtime_293545f_r1-20260811|835760128|3646
DELETED_PERMANENT|163|2026-08-11T09:33:08.285340Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_grounded_runtime_58f504d_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-163-shohin_ettr_grounded_runtime_58f504d_r1-20260811|836005888|3663
DELETED_PERMANENT|164|2026-08-11T09:33:09.757827Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_semantic_prefix_runtime_82394b8_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-164-shohin_ettr_semantic_prefix_runtime_82394b8_r1-20260811|836034560|3663
DELETED_PERMANENT|165|2026-08-11T09:33:11.256304Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_ensemble_runtime_03472e8_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-165-shohin_ettr_ensemble_runtime_03472e8_r1-20260811|833556480|3598
DELETED_PERMANENT|166|2026-08-11T09:33:12.722788Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_deployed_state_runtime_fb5a3bd_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-166-shohin_ettr_deployed_state_runtime_fb5a3bd_r1-20260811|833576960|3598
DELETED_PERMANENT|167|2026-08-11T09:33:14.224646Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_terminal_state_runtime_08cfaa4_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-167-shohin_ettr_terminal_state_runtime_08cfaa4_r1-20260811|833662976|3606
DELETED_PERMANENT|168|2026-08-11T09:33:15.836458Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_terminal_state_runtime_08cfaa4_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-168-shohin_ettr_terminal_state_runtime_08cfaa4_r2-20260811|833863680|3630
DELETED_PERMANENT|169|2026-08-11T09:33:17.187299Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_causal_delta_runtime_0fe3d5c_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-169-shohin_ettr_causal_delta_runtime_0fe3d5c_r1-20260811|834920448|3739
DELETED_PERMANENT|170|2026-08-11T09:33:18.534024Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_sparse_residual_runtime_a1e9c77_r1_REJECTED_INCOMPLETE|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-170-shohin_ettr_sparse_residual_runtime_a1e9c77_r1_REJECTED_INCOMPLETE-20260811|834916352|3739
DELETED_PERMANENT|171|2026-08-11T09:33:19.987298Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_sparse_residual_runtime_a1e9c77_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-171-shohin_ettr_sparse_residual_runtime_a1e9c77_r2-20260811|835293184|3760
DELETED_PERMANENT|172|2026-08-11T09:33:21.443847Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_sparse_residual_runtime_a543fd7_r3_REJECTED_INEXACT_SOURCE|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-172-shohin_ettr_sparse_residual_runtime_a543fd7_r3_REJECTED_INEXACT_SOURCE-20260811|835293184|3760
DELETED_PERMANENT|173|2026-08-11T09:33:22.794030Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_sparse_residual_runtime_a543fd7_r4|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-173-shohin_ettr_sparse_residual_runtime_a543fd7_r4-20260811|835297280|3760
DELETED_PERMANENT|174|2026-08-11T09:33:24.284995Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_atomic_typed_edit_runtime_fb368f2_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-174-shohin_ettr_atomic_typed_edit_runtime_fb368f2_r1-20260811|835604480|3771
DELETED_PERMANENT|175|2026-08-11T09:33:25.690277Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_dual_rail_lexical_command_runtime_c4fa270_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-175-shohin_ettr_dual_rail_lexical_command_runtime_c4fa270_r1-20260811|836935680|3792
DELETED_PERMANENT|176|2026-08-11T09:33:27.153731Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_token_native_router_runtime_8235fc9_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-176-shohin_ettr_token_native_router_runtime_8235fc9_r1-20260811|836960256|3795
DELETED_PERMANENT|177|2026-08-11T09:33:28.292975Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_occurrence_binding_runtime_9841b7d_r1.rejected-incomplete-20260802T0045|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-177-shohin_ettr_occurrence_binding_runtime_9841b7d_r1.rejected-incomplete-20260802T0045-20260811|671576064|3050
DELETED_PERMANENT|178|2026-08-11T09:33:28.929057Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_occurrence_binding_runtime_9841b7d_r2.partial.rejected-truncated|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-178-shohin_ettr_occurrence_binding_runtime_9841b7d_r2.partial.rejected-truncated-20260811|592015360|1349
DELETED_PERMANENT|179|2026-08-11T09:33:30.395663Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_occurrence_binding_runtime_9841b7d_r3|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-179-shohin_ettr_occurrence_binding_runtime_9841b7d_r3-20260811|837029888|3800
DELETED_PERMANENT|180|2026-08-11T09:33:31.829689Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_occurrence_schedule_runtime_60ecda3_r2.rejected-wrong-source-marker-20260802T0105|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-180-shohin_ettr_occurrence_schedule_runtime_60ecda3_r2.rejected-wrong-source-marker-20260802T0105-20260811|837050368|3801
DELETED_PERMANENT|181|2026-08-11T09:33:33.221507Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_occurrence_schedule_runtime_60ecda3_r3|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-181-shohin_ettr_occurrence_schedule_runtime_60ecda3_r3-20260811|837046272|3801
DELETED_PERMANENT|182|2026-08-11T09:33:34.611359Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_syntax_graph_runtime_00eefb9_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-182-shohin_ettr_syntax_graph_runtime_00eefb9_r1-20260811|837169152|3810
DELETED_PERMANENT|183|2026-08-11T09:33:34.626796Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_program_audit_632bde0_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-183-shohin_ettr_program_audit_632bde0_r1-20260811|32768|6
DELETED_PERMANENT|184|2026-08-11T09:33:34.641549Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_program_audit_632bde0_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-184-shohin_ettr_program_audit_632bde0_r2-20260811|32768|6
DELETED_PERMANENT|185|2026-08-11T09:33:34.655534Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_program_audit_e868d3c_r3|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-185-shohin_ettr_program_audit_e868d3c_r3-20260811|36864|6
DELETED_PERMANENT|186|2026-08-11T09:33:36.182493Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_sticky_macro_runtime_960333f_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-186-shohin_ettr_sticky_macro_runtime_960333f_r1-20260811|837971968|3829
DELETED_PERMANENT|187|2026-08-11T09:33:36.469265Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_opcode_projection_runtime_1a0cd1c_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-187-shohin_ettr_opcode_projection_runtime_1a0cd1c_r2-20260811|18472960|1271
DELETED_PERMANENT|188|2026-08-11T09:33:37.916652Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_opcode_projection_runtime_1a0cd1c_r3|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-188-shohin_ettr_opcode_projection_runtime_1a0cd1c_r3-20260811|836734976|3808
DELETED_PERMANENT|189|2026-08-11T09:33:37.925709Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_effect_diagnostic_runtime_16455f3_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-189-shohin_ettr_effect_diagnostic_runtime_16455f3_r1-20260811|4096|1
DELETED_PERMANENT|190|2026-08-11T09:33:38.673608Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_effect_diagnostic_runtime_16455f3_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-190-shohin_ettr_effect_diagnostic_runtime_16455f3_r2-20260811|467746816|1393
DELETED_PERMANENT|191|2026-08-11T09:33:39.893506Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_effect_family_id_runtime_1ed1a60_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-191-shohin_ettr_effect_family_id_runtime_1ed1a60_r1-20260811|667025408|3084
DELETED_PERMANENT|192|2026-08-11T09:33:41.155476Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_family_state_runtime_2523e7e_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-192-shohin_ettr_family_state_runtime_2523e7e_r1-20260811|667549696|3095
DELETED_PERMANENT|193|2026-08-11T09:33:42.340288Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_family_state_runtime_f6e65ec_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-193-shohin_ettr_family_state_runtime_f6e65ec_r1-20260811|667590656|3098
DELETED_PERMANENT|194|2026-08-11T09:33:42.350547Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_ettr_operation_family_campaign_admission_43084cd_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-194-shohin_ettr_operation_family_campaign_admission_43084cd_r1-20260811|2846720|5
DELETED_PERMANENT|195|2026-08-11T09:33:43.071447Z|/lustre/fs1/home/sa305415/shohin/scratchpad/quarantine_capability_floor_runtime_7730e7f_r1_bad_source_receipt|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-195-quarantine_capability_floor_runtime_7730e7f_r1_bad_source_receipt-20260811|544702464|1402
DELETED_PERMANENT|196|2026-08-11T09:33:43.611841Z|/lustre/fs1/home/sa305415/shohin/scratchpad/quarantine_capability_floor_runtime_7730e7f_r2_truncated_archive|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-196-quarantine_capability_floor_runtime_7730e7f_r2_truncated_archive-20260811|297447424|1122
DELETED_PERMANENT|197|2026-08-11T09:33:44.330816Z|/lustre/fs1/home/sa305415/shohin/scratchpad/quarantine_capability_floor_runtime_ebc8483_r1_bad_source_and_truncated_archive|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-197-quarantine_capability_floor_runtime_ebc8483_r1_bad_source_and_truncated_archive-20260811|544698368|1402
DELETED_PERMANENT|198|2026-08-11T09:33:44.357404Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_neural_reranker_runtime.JG6Yn0|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-198-shohin_neural_reranker_runtime.JG6Yn0-20260811|397312|34
DELETED_PERMANENT|199|2026-08-11T09:33:44.381055Z|/lustre/fs1/home/sa305415/shohin/scratchpad/shohin_process_verifier_runtime.KhKI8o|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-199-shohin_process_verifier_runtime.KhKI8o-20260811|348160|21
DELETED_PERMANENT|200|2026-08-11T09:33:44.403782Z|/lustre/fs1/home/sa305415/shohin/scratchpad/code_selector_runtime_5fc7127_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-200-code_selector_runtime_5fc7127_r1-20260811|286720|20
DELETED_PERMANENT|201|2026-08-11T09:33:44.428074Z|/lustre/fs1/home/sa305415/shohin/scratchpad/code_eval_runtime_824e2fa_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-201-code_eval_runtime_824e2fa_r1-20260811|335872|24
DELETED_PERMANENT|202|2026-08-11T09:33:44.453692Z|/lustre/fs1/home/sa305415/shohin/scratchpad/code_eval_runtime_7889aa7_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-202-code_eval_runtime_7889aa7_r1-20260811|344064|25
DELETED_PERMANENT|203|2026-08-11T09:33:44.478988Z|/lustre/fs1/home/sa305415/shohin/scratchpad/code_eval_runtime_daa1067_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-203-code_eval_runtime_daa1067_r1-20260811|344064|25
DELETED_PERMANENT|204|2026-08-11T09:33:44.493185Z|/lustre/fs1/home/sa305415/shohin/scratchpad/function_graph_runtime_4c8d2d1_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-204-function_graph_runtime_4c8d2d1_r1-20260811|86016|7
DELETED_PERMANENT|205|2026-08-11T09:33:44.504308Z|/lustre/fs1/home/sa305415/shohin/scratchpad/function_graph_runtime_849c6d4_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-205-function_graph_runtime_849c6d4_r1-20260811|36864|4
DELETED_PERMANENT|206|2026-08-11T09:33:44.515794Z|/lustre/fs1/home/sa305415/shohin/scratchpad/function_curriculum_runtime_16e6cc6_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-206-function_curriculum_runtime_16e6cc6_r1-20260811|20480|4
DELETED_PERMANENT|207|2026-08-11T09:33:44.527677Z|/lustre/fs1/home/sa305415/shohin/scratchpad/function_curriculum_runtime_9ec54d8_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-207-function_curriculum_runtime_9ec54d8_r1-20260811|20480|4
DELETED_PERMANENT|208|2026-08-11T09:33:44.539482Z|/lustre/fs1/home/sa305415/shohin/scratchpad/function_curriculum_runtime_912b8fb_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-208-function_curriculum_runtime_912b8fb_r1-20260811|24576|4
DELETED_PERMANENT|209|2026-08-11T09:33:44.555667Z|/lustre/fs1/home/sa305415/shohin/scratchpad/function_graph_v2_runtime_ea52bf8_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-209-function_graph_v2_runtime_ea52bf8_r1-20260811|126976|10
DELETED_PERMANENT|210|2026-08-11T09:33:44.580508Z|/lustre/fs1/home/sa305415/shohin/scratchpad/code_candidate_merge_runtime_6009e48_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-210-code_candidate_merge_runtime_6009e48_r1-20260811|368640|28
DELETED_PERMANENT|211|2026-08-11T09:33:44.598278Z|/lustre/fs1/home/sa305415/shohin/scratchpad/code_eval_runtime_9f00eb9_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-211-code_eval_runtime_9f00eb9_r1-20260811|184320|11
DELETED_PERMANENT|212|2026-08-11T09:33:44.609524Z|/lustre/fs1/home/sa305415/shohin/scratchpad/visible_code_repair_runtime_3321004_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-212-visible_code_repair_runtime_3321004_r2-20260811|20480|4
DELETED_PERMANENT|213|2026-08-11T09:33:44.620979Z|/lustre/fs1/home/sa305415/shohin/scratchpad/visible_code_repair_runtime_35b10f4_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-213-visible_code_repair_runtime_35b10f4_r1-20260811|20480|4
DELETED_PERMANENT|214|2026-08-11T09:33:44.633037Z|/lustre/fs1/home/sa305415/shohin/scratchpad/visible_code_repair_aggregate_cef5169_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-214-visible_code_repair_aggregate_cef5169_r1-20260811|20480|4
DELETED_PERMANENT|215|2026-08-11T09:33:44.644105Z|/lustre/fs1/home/sa305415/shohin/scratchpad/verified_code_repair_runtime_d20823d_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-215-verified_code_repair_runtime_d20823d_r1-20260811|28672|4
DELETED_PERMANENT|216|2026-08-11T09:33:44.658847Z|/lustre/fs1/home/sa305415/shohin/scratchpad/model_failure_repair_runtime_d238755_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-216-model_failure_repair_runtime_d238755_r1-20260811|73728|8
DELETED_PERMANENT|217|2026-08-11T09:33:44.683257Z|/lustre/fs1/home/sa305415/shohin/scratchpad/code_teacher_runtime_9878ba3_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-217-code_teacher_runtime_9878ba3_r1-20260811|339968|24
DELETED_PERMANENT|218|2026-08-11T09:33:44.707847Z|/lustre/fs1/home/sa305415/shohin/scratchpad/code_teacher_runtime_2fa929f_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-218-code_teacher_runtime_2fa929f_r1-20260811|339968|24
DELETED_PERMANENT|219|2026-08-11T09:33:44.719511Z|/lustre/fs1/home/sa305415/shohin/scratchpad/model_failure_preference_runtime_f4b430f_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-219-model_failure_preference_runtime_f4b430f_r1-20260811|24576|5
DELETED_PERMANENT|220|2026-08-11T09:33:44.735610Z|/lustre/fs1/home/sa305415/shohin/scratchpad/code_preference_runtime_f4b430f_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-220-code_preference_runtime_f4b430f_r1-20260811|90112|8
DELETED_PERMANENT|221|2026-08-11T09:33:44.756841Z|/lustre/fs1/home/sa305415/shohin/scratchpad/pcsd_runtime_a15e121|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-221-pcsd_runtime_a15e121-20260811|90112|13
DELETED_PERMANENT|222|2026-08-11T09:33:44.776507Z|/lustre/fs1/home/sa305415/shohin/scratchpad/fcpt_runtime_313ffea|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-222-fcpt_runtime_313ffea-20260811|73728|9
DELETED_PERMANENT|223|2026-08-11T09:33:44.796037Z|/lustre/fs1/home/sa305415/shohin/scratchpad/fcpt_runtime_650cc21|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-223-fcpt_runtime_650cc21-20260811|77824|9
DELETED_PERMANENT|224|2026-08-11T09:33:44.814996Z|/lustre/fs1/home/sa305415/shohin/scratchpad/cgsgr_runtime_fb60f2d|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-224-cgsgr_runtime_fb60f2d-20260811|106496|11
DELETED_PERMANENT|225|2026-08-11T09:33:44.839549Z|/lustre/fs1/home/sa305415/shohin/scratchpad/qvesr_runtime_b0c6f25|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-225-qvesr_runtime_b0c6f25-20260811|229376|23
DELETED_PERMANENT|226|2026-08-11T09:33:44.865833Z|/lustre/fs1/home/sa305415/shohin/scratchpad/qvesr_runtime_e3d4999|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-226-qvesr_runtime_e3d4999-20260811|282624|26
DELETED_PERMANENT|227|2026-08-11T09:33:44.891911Z|/lustre/fs1/home/sa305415/shohin/scratchpad/ceer_runtime_0e40f3f|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-227-ceer_runtime_0e40f3f-20260811|282624|26
DELETED_PERMANENT|228|2026-08-11T09:33:44.913389Z|/lustre/fs1/home/sa305415/shohin/scratchpad/pcdl_runtime_39ece41|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-228-pcdl_runtime_39ece41-20260811|106496|12
DELETED_PERMANENT|229|2026-08-11T09:33:44.928644Z|/lustre/fs1/home/sa305415/shohin/scratchpad/pspa_runtime_8ea9d4b|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-229-pspa_runtime_8ea9d4b-20260811|57344|7
DELETED_PERMANENT|230|2026-08-11T09:33:44.943108Z|/lustre/fs1/home/sa305415/shohin/scratchpad/pspa_runtime_071229a|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-230-pspa_runtime_071229a-20260811|61440|7
DELETED_PERMANENT|231|2026-08-11T09:33:44.965885Z|/lustre/fs1/home/sa305415/shohin/scratchpad/learned_pspa_dev|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-231-learned_pspa_dev-20260811|409600|31
DELETED_PERMANENT|232|2026-08-11T09:33:44.982042Z|/lustre/fs1/home/sa305415/shohin/scratchpad/learned_pspa_runtime_8fb5d61|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-232-learned_pspa_runtime_8fb5d61-20260811|90112|9
DELETED_PERMANENT|233|2026-08-11T09:33:44.995238Z|/lustre/fs1/home/sa305415/shohin/scratchpad/dwpc_runtime_e99b4ef|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-233-dwpc_runtime_e99b4ef-20260811|90112|8
DELETED_PERMANENT|234|2026-08-11T09:33:45.023610Z|/lustre/fs1/home/sa305415/shohin/scratchpad/csdc_runtime_6121e2f|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-234-csdc_runtime_6121e2f-20260811|102400|9
DELETED_PERMANENT|235|2026-08-11T09:33:45.041754Z|/lustre/fs1/home/sa305415/shohin/scratchpad/csdc_semantic_runtime_152d42f|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-235-csdc_semantic_runtime_152d42f-20260811|139264|12
DELETED_PERMANENT|236|2026-08-11T09:33:45.058721Z|/lustre/fs1/home/sa305415/shohin/scratchpad/csdc_copy_runtime_6359453|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-236-csdc_copy_runtime_6359453-20260811|159744|13
DELETED_PERMANENT|237|2026-08-11T09:33:45.075632Z|/lustre/fs1/home/sa305415/shohin/scratchpad/csdc_smollm2_bridge_96217cb_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-237-csdc_smollm2_bridge_96217cb_r1-20260811|212992|14
DELETED_PERMANENT|238|2026-08-11T09:33:45.090161Z|/lustre/fs1/home/sa305415/shohin/scratchpad/csdc_smollm2_diag_2850b94_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-238-csdc_smollm2_diag_2850b94_r1-20260811|217088|13
DELETED_PERMANENT|239|2026-08-11T09:33:45.872844Z|/lustre/fs1/home/sa305415/shohin/scratchpad/csdc_span_quotient_f1b91e9_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-239-csdc_span_quotient_f1b91e9_r1-20260811|203706368|2668
DELETED_PERMANENT|240|2026-08-11T09:33:46.669404Z|/lustre/fs1/home/sa305415/shohin/scratchpad/csdc_span_quotient_405fcbb_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-240-csdc_span_quotient_405fcbb_r2-20260811|203706368|2668
DELETED_PERMANENT|241|2026-08-11T09:33:47.467637Z|/lustre/fs1/home/sa305415/shohin/scratchpad/csdc_span_quotient_e28f6d3_r3|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-241-csdc_span_quotient_e28f6d3_r3-20260811|203726848|2671
DELETED_PERMANENT|242|2026-08-11T09:33:47.482565Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_v0_smollm2_bd34385_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-242-diverge_v0_smollm2_bd34385_r1-20260811|192512|11
DELETED_PERMANENT|243|2026-08-11T09:33:47.497074Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_v0_smollm2_bd34385_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-243-diverge_v0_smollm2_bd34385_r2-20260811|208896|12
DELETED_PERMANENT|244|2026-08-11T09:33:47.509693Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_v0_smollm2_5241a56_audit_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-244-diverge_v0_smollm2_5241a56_audit_r1-20260811|208896|11
DELETED_PERMANENT|245|2026-08-11T09:33:47.525649Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_v0_role_copy_536f29c_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-245-diverge_v0_role_copy_536f29c_r1-20260811|233472|13
DELETED_PERMANENT|246|2026-08-11T09:33:47.541266Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_v0_role_copy_e1bdf8b_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-246-diverge_v0_role_copy_e1bdf8b_r2-20260811|233472|13
DELETED_PERMANENT|247|2026-08-11T09:33:47.557003Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_v0_matched_gate_0249b2c_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-247-diverge_v0_matched_gate_0249b2c_r2-20260811|245760|13
DELETED_PERMANENT|248|2026-08-11T09:33:47.572814Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_v0_matched_gate_0249b2c_r3|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-248-diverge_v0_matched_gate_0249b2c_r3-20260811|290816|15
DELETED_PERMANENT|249|2026-08-11T09:33:47.589013Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_v0_matched_gate_8e846c3_r4|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-249-diverge_v0_matched_gate_8e846c3_r4-20260811|299008|15
DELETED_PERMANENT|250|2026-08-11T09:33:47.607004Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_v0_matched_gate_fe2e5a8_r5|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-250-diverge_v0_matched_gate_fe2e5a8_r5-20260811|339968|17
DELETED_PERMANENT|251|2026-08-11T09:33:47.624538Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_v0_matched_gate_0de0afe_r6|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-251-diverge_v0_matched_gate_0de0afe_r6-20260811|339968|17
DELETED_PERMANENT|252|2026-08-11T09:33:47.641376Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_v0_matched_gate_56a2aa5_r7|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-252-diverge_v0_matched_gate_56a2aa5_r7-20260811|319488|16
DELETED_PERMANENT|253|2026-08-11T09:33:47.657931Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_v0_matched_gate_e56a37f_r8|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-253-diverge_v0_matched_gate_e56a37f_r8-20260811|319488|16
DELETED_PERMANENT|254|2026-08-11T09:33:47.675464Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_sc1_e59fe33_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-254-diverge_sc1_e59fe33_r1-20260811|147456|17
DELETED_PERMANENT|255|2026-08-11T09:33:47.690746Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_sc1_e59fe33_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-255-diverge_sc1_e59fe33_r2-20260811|118784|10
DELETED_PERMANENT|256|2026-08-11T09:33:47.707294Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_sc1_d0af1d1_audit_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-256-diverge_sc1_d0af1d1_audit_r1-20260811|143360|13
DELETED_PERMANENT|257|2026-08-11T09:33:47.740271Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_wra1_4f07bdf_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-257-diverge_wra1_4f07bdf_r1-20260811|2441216|37
DELETED_PERMANENT|258|2026-08-11T09:33:47.755129Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_hsc1_d7360cd_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-258-diverge_hsc1_d7360cd_r1-20260811|28672|7
DELETED_PERMANENT|259|2026-08-11T09:33:47.782963Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_hsc1_d7360cd_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-259-diverge_hsc1_d7360cd_r2-20260811|397312|23
DELETED_PERMANENT|260|2026-08-11T09:33:47.812272Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_hsc1_fce7efb_r3|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-260-diverge_hsc1_fce7efb_r3-20260811|581632|28
DELETED_PERMANENT|261|2026-08-11T09:33:49.493963Z|/lustre/fs1/home/sa305415/shohin/runtimes/diverge_ulc1_hsc1_2e7b326|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-261-diverge_ulc1_hsc1_2e7b326-20260811|844697600|4297
DELETED_PERMANENT|262|2026-08-11T09:33:51.075062Z|/lustre/fs1/home/sa305415/shohin/runtimes/diverge_ulc1_hsc1_b4dc261|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-262-diverge_ulc1_hsc1_b4dc261-20260811|844697600|4297
DELETED_PERMANENT|263|2026-08-11T09:33:52.677743Z|/lustre/fs1/home/sa305415/shohin/runtimes/diverge_ulc1_hsc1_fbc8623|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-263-diverge_ulc1_hsc1_fbc8623-20260811|844697600|4297
DELETED_PERMANENT|264|2026-08-11T09:33:54.304297Z|/lustre/fs1/home/sa305415/shohin/runtimes/diverge_ulc1_hsc1_bf049df|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-264-diverge_ulc1_hsc1_bf049df-20260811|845361152|4304
DELETED_PERMANENT|265|2026-08-11T09:33:54.329329Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_vmt1_b4f5766_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-265-diverge_vmt1_b4f5766_r1-20260811|225280|23
DELETED_PERMANENT|266|2026-08-11T09:33:54.352744Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_vcr1_3224d1e_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-266-diverge_vcr1_3224d1e_r1-20260811|184320|22
DELETED_PERMANENT|267|2026-08-11T09:33:54.376597Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_vcr1_20c94f4_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-267-diverge_vcr1_20c94f4_r1-20260811|184320|22
DELETED_PERMANENT|268|2026-08-11T09:33:54.394164Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_vcr1_c5fe1d4_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-268-diverge_vcr1_c5fe1d4_r1-20260811|126976|10
DELETED_PERMANENT|269|2026-08-11T09:33:54.412302Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_vcr1_c5fe1d4_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-269-diverge_vcr1_c5fe1d4_r2-20260811|139264|11
DELETED_PERMANENT|270|2026-08-11T09:33:54.427884Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_vcr1_c5fe1d4_r3|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-270-diverge_vcr1_c5fe1d4_r3-20260811|122880|11
DELETED_PERMANENT|271|2026-08-11T09:33:54.443846Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_vcr1_620a8c3_eval_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-271-diverge_vcr1_620a8c3_eval_r1-20260811|176128|12
DELETED_PERMANENT|272|2026-08-11T09:33:54.455004Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_vcr1_7f38109_score_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-272-diverge_vcr1_7f38109_score_r1-20260811|24576|4
DELETED_PERMANENT|273|2026-08-11T09:33:54.469028Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_vcr1_7f38109_finalizer_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-273-diverge_vcr1_7f38109_finalizer_r1-20260811|20480|5
DELETED_PERMANENT|274|2026-08-11T09:33:54.482826Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_vcr1_2c64021_finalizer_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-274-diverge_vcr1_2c64021_finalizer_r1-20260811|20480|5
DELETED_PERMANENT|275|2026-08-11T09:33:54.510268Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_crp1_43d81d0_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-275-diverge_crp1_43d81d0_r1-20260811|19415040|25
DELETED_PERMANENT|276|2026-08-11T09:33:54.535057Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_rsm1_f19c6dc_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-276-diverge_rsm1_f19c6dc_r1-20260811|278528|26
DELETED_PERMANENT|277|2026-08-11T09:33:54.559306Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_ats1_a9790ee_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-277-diverge_ats1_a9790ee_r1-20260811|5226496|25
DELETED_PERMANENT|278|2026-08-11T09:33:54.583844Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_ats1_ecc07eb_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-278-diverge_ats1_ecc07eb_r2-20260811|5169152|22
DELETED_PERMANENT|279|2026-08-11T09:33:54.601015Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_ats1_207fbc1_eval_r3|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-279-diverge_ats1_207fbc1_eval_r3-20260811|81920|11
DELETED_PERMANENT|280|2026-08-11T09:33:54.626594Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_fta1_8eeb136_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-280-diverge_fta1_8eeb136_r1-20260811|6991872|24
DELETED_PERMANENT|281|2026-08-11T09:33:54.652139Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_tol1_89622d5_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-281-diverge_tol1_89622d5_r1-20260811|167936|24
DELETED_PERMANENT|282|2026-08-11T09:33:54.667803Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_iem1_eval_fb23f6b_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-282-diverge_iem1_eval_fb23f6b_r1-20260811|106496|11
DELETED_PERMANENT|283|2026-08-11T09:33:55.503744Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_sot1_baa50a1_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-283-diverge_sot1_baa50a1_r1-20260811|222588928|2906
DELETED_PERMANENT|284|2026-08-11T09:33:55.512704Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_sot1_eval_1b755f4_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-284-diverge_sot1_eval_1b755f4_r1-20260811|8192|2
DELETED_PERMANENT|285|2026-08-11T09:33:55.520992Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_npw1_2ba2837_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-285-diverge_npw1_2ba2837_r1-20260811|4096|1
DELETED_PERMANENT|286|2026-08-11T09:33:56.418339Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_srp1_ce8b1c4_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-286-diverge_srp1_ce8b1c4_r1-20260811|206528512|2924
DELETED_PERMANENT|287|2026-08-11T09:33:57.325521Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_srp1_906b6f1_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-287-diverge_srp1_906b6f1_r1-20260811|206528512|2924
DELETED_PERMANENT|288|2026-08-11T09:33:58.248684Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_ccr1_ec6d2f3_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-288-diverge_ccr1_ec6d2f3_r1-20260811|207429632|2973
DELETED_PERMANENT|289|2026-08-11T09:33:59.166603Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_ccr1_c2ee33f_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-289-diverge_ccr1_c2ee33f_r1-20260811|207429632|2973
DELETED_PERMANENT|290|2026-08-11T09:34:00.001498Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_ccr1_b0f0a6e_diag_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-290-diverge_ccr1_b0f0a6e_diag_r1-20260811|206712832|2945
DELETED_PERMANENT|291|2026-08-11T09:34:00.110461Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_nls1_eff7d06_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-291-diverge_nls1_eff7d06_r1-20260811|2555904|28
DELETED_PERMANENT|292|2026-08-11T09:34:00.135894Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_ncp1_f84cabc_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-292-diverge_ncp1_f84cabc_r1-20260811|2584576|28
DELETED_PERMANENT|293|2026-08-11T09:34:00.161240Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_jrb1_1ab21c9_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-293-diverge_jrb1_1ab21c9_r1-20260811|2666496|29
DELETED_PERMANENT|294|2026-08-11T09:34:00.485978Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_cab1_6820c49_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-294-diverge_cab1_6820c49_r1-20260811|24805376|1438
DELETED_PERMANENT|295|2026-08-11T09:34:00.802044Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_oqb1_2ead185_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-295-diverge_oqb1_2ead185_r1-20260811|22577152|1441
DELETED_PERMANENT|296|2026-08-11T09:34:01.383712Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_sve1_bf7656f_r3|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-296-diverge_sve1_bf7656f_r3-20260811|38621184|2416
DELETED_PERMANENT|297|2026-08-11T09:34:03.264040Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_snl1_2451988_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-297-diverge_snl1_2451988_r1-20260811|851664896|4827
DELETED_PERMANENT|298|2026-08-11T09:34:03.280558Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_qst1_0973e7e_r2|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-298-diverge_qst1_0973e7e_r2-20260811|159744|13
DELETED_PERMANENT|299|2026-08-11T09:34:03.304276Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_qpt1_402fa8e_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-299-diverge_qpt1_402fa8e_r1-20260811|2551808|18
DELETED_PERMANENT|300|2026-08-11T09:34:03.326826Z|/lustre/fs1/home/sa305415/shohin/scratchpad/diverge_qpt1_controls_d50e262_r1|/lustre/fs1/home/sa305415/shohin/scratchpad/.pcf1-age-delete-B2b-300-diverge_qpt1_controls_d50e262_r1-20260811|2560000|18
DELETED_PERMANENT|301|2026-08-11T09:34:04.262631Z|/lustre/fs1/home/sa305415/shohin/runtime/cvg1_rollouts_cb48f18_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-301-cvg1_rollouts_cb48f18_r1-20260811|209580032|3221
DELETED_PERMANENT|302|2026-08-11T09:34:05.170589Z|/lustre/fs1/home/sa305415/shohin/runtime/cvg1_rollouts_bc1d185_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-302-cvg1_rollouts_bc1d185_r1-20260811|209580032|3221
DELETED_PERMANENT|303|2026-08-11T09:34:06.113611Z|/lustre/fs1/home/sa305415/shohin/runtime/cvg1_rollouts_dd4ef87_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-303-cvg1_rollouts_dd4ef87_r1-20260811|209604608|3223
DELETED_PERMANENT|304|2026-08-11T09:34:06.132041Z|/lustre/fs1/home/sa305415/shohin/runtime/cvg1_critic_a960206_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-304-cvg1_critic_a960206_r1-20260811|159744|11
DELETED_PERMANENT|305|2026-08-11T09:34:06.149922Z|/lustre/fs1/home/sa305415/shohin/runtime/cvg1_apply_88498a1_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-305-cvg1_apply_88498a1_r1-20260811|167936|12
DELETED_PERMANENT|306|2026-08-11T09:34:06.167228Z|/lustre/fs1/home/sa305415/shohin/runtime/pcj1_beb127f_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-306-pcj1_beb127f_r1-20260811|200704|14
DELETED_PERMANENT|307|2026-08-11T09:34:06.185505Z|/lustre/fs1/home/sa305415/shohin/runtime/vcr1_8152cc8_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-307-vcr1_8152cc8_r1-20260811|225280|16
DELETED_PERMANENT|308|2026-08-11T09:34:06.203901Z|/lustre/fs1/home/sa305415/shohin/runtime/sdr1_d11b231_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-308-sdr1_d11b231_r1-20260811|241664|17
DELETED_PERMANENT|309|2026-08-11T09:34:06.219894Z|/lustre/fs1/home/sa305415/shohin/runtime/idr1_drafts_e7b5be3_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-309-idr1_drafts_e7b5be3_r1-20260811|167936|16
DELETED_PERMANENT|310|2026-08-11T09:34:06.235637Z|/lustre/fs1/home/sa305415/shohin/runtime/idr1_drafts_e7b5be3_r2|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-310-idr1_drafts_e7b5be3_r2-20260811|143360|10
DELETED_PERMANENT|311|2026-08-11T09:34:06.258104Z|/lustre/fs1/home/sa305415/shohin/runtime/idr1_ad790b0_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-311-idr1_ad790b0_r1-20260811|253952|20
DELETED_PERMANENT|312|2026-08-11T09:34:06.280494Z|/lustre/fs1/home/sa305415/shohin/runtime/idr1_d3fbff5_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-312-idr1_d3fbff5_r1-20260811|262144|22
DELETED_PERMANENT|313|2026-08-11T09:34:06.302547Z|/lustre/fs1/home/sa305415/shohin/runtime/idr1_bc2bb10_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-313-idr1_bc2bb10_r1-20260811|262144|22
DELETED_PERMANENT|314|2026-08-11T09:34:06.324508Z|/lustre/fs1/home/sa305415/shohin/runtime/idr1_55b6476_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-314-idr1_55b6476_r1-20260811|274432|22
DELETED_PERMANENT|315|2026-08-11T09:34:06.347413Z|/lustre/fs1/home/sa305415/shohin/runtime/idr1_0ecf5f0_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-315-idr1_0ecf5f0_r1-20260811|282624|23
DELETED_PERMANENT|316|2026-08-11T09:34:06.371289Z|/lustre/fs1/home/sa305415/shohin/runtime/aqc1_192a7f7_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-316-aqc1_192a7f7_r1-20260811|327680|27
DELETED_PERMANENT|317|2026-08-11T09:34:06.394480Z|/lustre/fs1/home/sa305415/shohin/runtime/aqc1_61da147_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-317-aqc1_61da147_r1-20260811|327680|27
DELETED_PERMANENT|318|2026-08-11T09:34:06.418521Z|/lustre/fs1/home/sa305415/shohin/runtime/aqc1_38bc1f3_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-318-aqc1_38bc1f3_r1-20260811|327680|27
DELETED_PERMANENT|319|2026-08-11T09:34:06.538413Z|/lustre/fs1/home/sa305415/shohin/runtime/idr4_4b81c22_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-319-idr4_4b81c22_r1-20260811|75325440|481
DELETED_PERMANENT|320|2026-08-11T09:34:06.563272Z|/lustre/fs1/home/sa305415/shohin/runtime/idr4_6dddeaf_r2|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-320-idr4_6dddeaf_r2-20260811|2625536|23
DELETED_PERMANENT|321|2026-08-11T09:34:06.586079Z|/lustre/fs1/home/sa305415/shohin/runtime/idr4_6dddeaf_r3|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-321-idr4_6dddeaf_r3-20260811|266240|23
DELETED_PERMANENT|322|2026-08-11T09:34:06.608388Z|/lustre/fs1/home/sa305415/shohin/runtime/idr4_8891621_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-322-idr4_8891621_r1-20260811|282624|26
DELETED_PERMANENT|323|2026-08-11T09:34:06.630955Z|/lustre/fs1/home/sa305415/shohin/runtime/idr4_687a91a_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-323-idr4_687a91a_r1-20260811|290816|27
DELETED_PERMANENT|324|2026-08-11T09:34:06.654311Z|/lustre/fs1/home/sa305415/shohin/runtime/idr4_d681daa_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-324-idr4_d681daa_r1-20260811|290816|27
DELETED_PERMANENT|325|2026-08-11T09:34:06.677639Z|/lustre/fs1/home/sa305415/shohin/runtime/idr4_eval_54454d7_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-325-idr4_eval_54454d7_r1-20260811|290816|27
DELETED_PERMANENT|326|2026-08-11T09:34:06.702714Z|/lustre/fs1/home/sa305415/shohin/runtime/idr08_28c1246_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-326-idr08_28c1246_r1-20260811|315392|32
DELETED_PERMANENT|327|2026-08-11T09:34:06.728288Z|/lustre/fs1/home/sa305415/shohin/runtime/idr08_28c1246_r2_incomplete_20260808T171128|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-327-idr08_28c1246_r2_incomplete_20260808T171128-20260811|315392|32
DELETED_PERMANENT|328|2026-08-11T09:34:06.744294Z|/lustre/fs1/home/sa305415/shohin/runtime/phase2_near_3ab418c_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-328-phase2_near_3ab418c_r1-20260811|208896|12
DELETED_PERMANENT|329|2026-08-11T09:34:06.760401Z|/lustre/fs1/home/sa305415/shohin/runtime/phase2_near_residual_3ab418c_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-329-phase2_near_residual_3ab418c_r1-20260811|249856|14
DELETED_PERMANENT|330|2026-08-11T09:34:06.772123Z|/lustre/fs1/home/sa305415/shohin/runtime/tokenizer_compare_fbf81cf_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-330-tokenizer_compare_fbf81cf_r1-20260811|20480|4
DELETED_PERMANENT|331|2026-08-11T09:34:06.787526Z|/lustre/fs1/home/sa305415/shohin/runtime/phase2_retokenize_0f167fa_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-331-phase2_retokenize_0f167fa_r1-20260811|122880|9
DELETED_PERMANENT|332|2026-08-11T09:34:06.814065Z|/lustre/fs1/home/sa305415/shohin/runtime/idr08_28c1246_r2|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-332-idr08_28c1246_r2-20260811|339968|33
DELETED_PERMANENT|333|2026-08-11T09:34:06.823169Z|/lustre/fs1/home/sa305415/shohin/runtime/checkpoint_normalize_1a01705_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-333-checkpoint_normalize_1a01705_r1-20260811|16384|3
DELETED_PERMANENT|334|2026-08-11T09:34:06.838082Z|/lustre/fs1/home/sa305415/shohin/runtime/phase2_retokenize_3d6e140_r2|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-334-phase2_retokenize_3d6e140_r2-20260811|126976|9
DELETED_PERMANENT|335|2026-08-11T09:34:06.853108Z|/lustre/fs1/home/sa305415/shohin/runtime/phase2_retokenize_4abf367_r3|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-335-phase2_retokenize_4abf367_r3-20260811|126976|9
DELETED_PERMANENT|336|2026-08-11T09:34:06.867653Z|/lustre/fs1/home/sa305415/shohin/runtime/phase2_retokenize_a481bc4_r4|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-336-phase2_retokenize_a481bc4_r4-20260811|122880|9
DELETED_PERMANENT|337|2026-08-11T09:34:06.884649Z|/lustre/fs1/home/sa305415/shohin/runtime/phase2_near49k_f9aeeef_r1|/lustre/fs1/home/sa305415/shohin/runtime/.pcf1-age-delete-B2b-337-phase2_near49k_f9aeeef_r1-20260811|204800|13
DELETED_PERMANENT|338|2026-08-11T09:34:06.905363Z|/lustre/fs1/home/sa305415/shohin/runtimes/ttr1_ddba463_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-338-ttr1_ddba463_r1-20260811|155648|14
DELETED_PERMANENT|339|2026-08-11T09:34:06.931081Z|/lustre/fs1/home/sa305415/shohin/runtimes/ttr1_8659590_r2|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-339-ttr1_8659590_r2-20260811|335872|31
DELETED_PERMANENT|340|2026-08-11T09:34:06.955928Z|/lustre/fs1/home/sa305415/shohin/runtimes/ttr1_b3d7cb9_r3|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-340-ttr1_b3d7cb9_r3-20260811|356352|33
DELETED_PERMANENT|341|2026-08-11T09:34:06.981118Z|/lustre/fs1/home/sa305415/shohin/runtimes/ttr1_6c1db22_r4|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-341-ttr1_6c1db22_r4-20260811|364544|34
DELETED_PERMANENT|342|2026-08-11T09:34:07.031442Z|/lustre/fs1/home/sa305415/shohin/runtimes/ttr1_d0b7730_r5|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-342-ttr1_d0b7730_r5-20260811|376832|36
DELETED_PERMANENT|343|2026-08-11T09:34:07.059164Z|/lustre/fs1/home/sa305415/shohin/runtimes/ttr1_5f4a83b_r6|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-343-ttr1_5f4a83b_r6-20260811|376832|36
DELETED_PERMANENT|344|2026-08-11T09:34:08.816616Z|/lustre/fs1/home/sa305415/shohin/runtimes/ttr1_bf58208_r7|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-344-ttr1_bf58208_r7-20260811|855810048|5052
DELETED_PERMANENT|345|2026-08-11T09:34:10.598220Z|/lustre/fs1/home/sa305415/shohin/runtimes/ttr1_4ebdfe8_r8|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-345-ttr1_4ebdfe8_r8-20260811|855830528|5054
DELETED_PERMANENT|346|2026-08-11T09:34:12.430293Z|/lustre/fs1/home/sa305415/shohin/runtimes/ttr1_d03eafc_r9|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-346-ttr1_d03eafc_r9-20260811|855838720|5054
DELETED_PERMANENT|347|2026-08-11T09:34:14.227207Z|/lustre/fs1/home/sa305415/shohin/runtimes/ttr1_7b5415d_r10|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-347-ttr1_7b5415d_r10-20260811|855834624|5054
DELETED_PERMANENT|348|2026-08-11T09:34:15.957985Z|/lustre/fs1/home/sa305415/shohin/runtimes/ttr1_1c96116_r11|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-348-ttr1_1c96116_r11-20260811|855830528|5054
DELETED_PERMANENT|349|2026-08-11T09:34:16.853599Z|/lustre/fs1/home/sa305415/shohin/runtimes/sctr1_7ac0a94_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-349-sctr1_7ac0a94_r1-20260811|210575360|3354
DELETED_PERMANENT|350|2026-08-11T09:34:17.785094Z|/lustre/fs1/home/sa305415/shohin/runtimes/sctr1_bbe3a7a_r2|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-350-sctr1_bbe3a7a_r2-20260811|210587648|3356
DELETED_PERMANENT|351|2026-08-11T09:34:18.697542Z|/lustre/fs1/home/sa305415/shohin/runtimes/sctr1_0a92c76_r3|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-351-sctr1_0a92c76_r3-20260811|210595840|3357
DELETED_PERMANENT|352|2026-08-11T09:34:19.647942Z|/lustre/fs1/home/sa305415/shohin/runtimes/sctr1_75f99ae_r5|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-352-sctr1_75f99ae_r5-20260811|210624512|3361
DELETED_PERMANENT|353|2026-08-11T09:34:20.577592Z|/lustre/fs1/home/sa305415/shohin/runtimes/sctr1_1260cce_r6|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-353-sctr1_1260cce_r6-20260811|210554880|3351
DELETED_PERMANENT|354|2026-08-11T09:34:21.503221Z|/lustre/fs1/home/sa305415/shohin/runtimes/sctr1_f6f7f55_r7|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-354-sctr1_f6f7f55_r7-20260811|210554880|3351
DELETED_PERMANENT|355|2026-08-11T09:34:22.433304Z|/lustre/fs1/home/sa305415/shohin/runtimes/esr1_9253961_r8|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-355-esr1_9253961_r8-20260811|210595840|3358
DELETED_PERMANENT|356|2026-08-11T09:34:23.369430Z|/lustre/fs1/home/sa305415/shohin/runtimes/q36moe_e39a53f_r9|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-356-q36moe_e39a53f_r9-20260811|210608128|3360
DELETED_PERMANENT|357|2026-08-11T09:34:23.383876Z|/lustre/fs1/home/sa305415/shohin/runtimes/q36kernels_761a972_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-357-q36kernels_761a972_r1-20260811|20480|5
DELETED_PERMANENT|358|2026-08-11T09:34:24.305297Z|/lustre/fs1/home/sa305415/shohin/runtimes/q36moe_9eaeeff_r10|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-358-q36moe_9eaeeff_r10-20260811|210612224|3361
DELETED_PERMANENT|359|2026-08-11T09:34:25.224258Z|/lustre/fs1/home/sa305415/shohin/runtimes/q36kernels_5d4b09e_r2|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-359-q36kernels_5d4b09e_r2-20260811|210612224|3361
DELETED_PERMANENT|360|2026-08-11T09:34:26.131535Z|/lustre/fs1/home/sa305415/shohin/runtimes/q36kernels_36e56f6_r3|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-360-q36kernels_36e56f6_r3-20260811|210612224|3361
DELETED_PERMANENT|361|2026-08-11T09:34:27.063239Z|/lustre/fs1/home/sa305415/shohin/runtimes/q36kernels_0381541_r4|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-361-q36kernels_0381541_r4-20260811|210612224|3361
DELETED_PERMANENT|362|2026-08-11T09:34:27.996289Z|/lustre/fs1/home/sa305415/shohin/runtimes/q36kernels_65b81e3_r5|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-362-q36kernels_65b81e3_r5-20260811|210612224|3361
DELETED_PERMANENT|363|2026-08-11T09:34:28.895441Z|/lustre/fs1/home/sa305415/shohin/runtimes/q36rollouts_4d74a59_r11|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-363-q36rollouts_4d74a59_r11-20260811|210616320|3361
DELETED_PERMANENT|364|2026-08-11T09:34:29.834062Z|/lustre/fs1/home/sa305415/shohin/runtimes/q36steady_05fdbbe_r12|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-364-q36steady_05fdbbe_r12-20260811|210616320|3361
DELETED_PERMANENT|365|2026-08-11T09:34:30.849545Z|/lustre/fs1/home/sa305415/shohin/runtimes/mtr1_2bf7023_r2.failed_copy_1786249064|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-365-mtr1_2bf7023_r2.failed_copy_1786249064-20260811|216023040|3370
DELETED_PERMANENT|366|2026-08-11T09:34:31.810296Z|/lustre/fs1/home/sa305415/shohin/runtimes/mtr1_6809591_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-366-mtr1_6809591_r1-20260811|216023040|3370
DELETED_PERMANENT|367|2026-08-11T09:34:32.791260Z|/lustre/fs1/home/sa305415/shohin/runtimes/mtr1_2bf7023_r2|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-367-mtr1_2bf7023_r2-20260811|216027136|3371
DELETED_PERMANENT|368|2026-08-11T09:34:33.738425Z|/lustre/fs1/home/sa305415/shohin/runtimes/mtr1_32bc0df_r3|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-368-mtr1_32bc0df_r3-20260811|216039424|3373
DELETED_PERMANENT|369|2026-08-11T09:34:34.657379Z|/lustre/fs1/home/sa305415/shohin/runtimes/mtr1_6ee8506_r4|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-369-mtr1_6ee8506_r4-20260811|216068096|3378
DELETED_PERMANENT|370|2026-08-11T09:34:35.596905Z|/lustre/fs1/home/sa305415/shohin/runtimes/mtr1_592a35e_r5|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-370-mtr1_592a35e_r5-20260811|216084480|3380
DELETED_PERMANENT|371|2026-08-11T09:34:36.526294Z|/lustre/fs1/home/sa305415/shohin/runtimes/mtr1_14275aa_r6|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-371-mtr1_14275aa_r6-20260811|216088576|3380
DELETED_PERMANENT|372|2026-08-11T09:34:37.460334Z|/lustre/fs1/home/sa305415/shohin/runtimes/mtr1_14275aa_r7|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-372-mtr1_14275aa_r7-20260811|216088576|3380
DELETED_PERMANENT|373|2026-08-11T09:34:38.393424Z|/lustre/fs1/home/sa305415/shohin/runtimes/mtr1_277e1ca_r8|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-373-mtr1_277e1ca_r8-20260811|216088576|3380
DELETED_PERMANENT|374|2026-08-11T09:34:39.345087Z|/lustre/fs1/home/sa305415/shohin/runtimes/rcr1_1c44a27_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-374-rcr1_1c44a27_r1-20260811|216100864|3381
DELETED_PERMANENT|375|2026-08-11T09:34:40.296440Z|/lustre/fs1/home/sa305415/shohin/runtimes/rcr1_67ac03f_r2|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-375-rcr1_67ac03f_r2-20260811|216117248|3384
DELETED_PERMANENT|376|2026-08-11T09:34:41.229847Z|/lustre/fs1/home/sa305415/shohin/runtimes/mtr2_ffa7e06_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-376-mtr2_ffa7e06_r1-20260811|216117248|3384
DELETED_PERMANENT|377|2026-08-11T09:34:42.160287Z|/lustre/fs1/home/sa305415/shohin/runtimes/mtr2_750800e_r2|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-377-mtr2_750800e_r2-20260811|216117248|3384
DELETED_PERMANENT|378|2026-08-11T09:34:43.095877Z|/lustre/fs1/home/sa305415/shohin/runtimes/moeattr_cb0ce73_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-378-moeattr_cb0ce73_r1-20260811|216133632|3386
DELETED_PERMANENT|379|2026-08-11T09:34:43.651694Z|/lustre/fs1/home/sa305415/shohin/runtimes/drem1_e532105_r1_INVALID_INCOMPLETE_ARCHIVE|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-379-drem1_e532105_r1_INVALID_INCOMPLETE_ARCHIVE-20260811|299970560|1145
DELETED_PERMANENT|380|2026-08-11T09:34:44.587117Z|/lustre/fs1/home/sa305415/shohin/runtimes/drem1_e532105_r2|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-380-drem1_e532105_r2-20260811|216182784|3389
DELETED_PERMANENT|381|2026-08-11T09:34:44.611490Z|/lustre/fs1/home/sa305415/shohin/runtimes/ridr1_7cc2292_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-381-ridr1_7cc2292_r1-20260811|380928|39
DELETED_PERMANENT|382|2026-08-11T09:34:44.632515Z|/lustre/fs1/home/sa305415/shohin/runtimes/ridr1_7cc2292_r2|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-382-ridr1_7cc2292_r2-20260811|311296|22
DELETED_PERMANENT|383|2026-08-11T09:34:44.653952Z|/lustre/fs1/home/sa305415/shohin/runtimes/kr2_236c8e0_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-383-kr2_236c8e0_r1-20260811|352256|27
DELETED_PERMANENT|384|2026-08-11T09:34:44.675401Z|/lustre/fs1/home/sa305415/shohin/runtimes/kr2_87a37c8_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-384-kr2_87a37c8_r1-20260811|364544|29
DELETED_PERMANENT|385|2026-08-11T09:34:44.696384Z|/lustre/fs1/home/sa305415/shohin/runtimes/kr2_56579a2_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-385-kr2_56579a2_r1-20260811|364544|29
DELETED_PERMANENT|386|2026-08-11T09:34:44.717898Z|/lustre/fs1/home/sa305415/shohin/runtimes/kr2_78b4715_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-386-kr2_78b4715_r1-20260811|364544|29
DELETED_PERMANENT|387|2026-08-11T09:34:44.739158Z|/lustre/fs1/home/sa305415/shohin/runtimes/kr2_8197da6_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-387-kr2_8197da6_r1-20260811|364544|29
DELETED_PERMANENT|388|2026-08-11T09:34:44.759396Z|/lustre/fs1/home/sa305415/shohin/runtimes/kr2_abdf2ac_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-388-kr2_abdf2ac_r1-20260811|364544|29
DELETED_PERMANENT|389|2026-08-11T09:34:44.771141Z|/lustre/fs1/home/sa305415/shohin/runtimes/tcs1_e25b858_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-389-tcs1_e25b858_r1-20260811|40960|5
DELETED_PERMANENT|390|2026-08-11T09:34:44.782333Z|/lustre/fs1/home/sa305415/shohin/runtimes/tcs1_5c06b92_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-390-tcs1_5c06b92_r1-20260811|40960|5
DELETED_PERMANENT|391|2026-08-11T09:34:44.804509Z|/lustre/fs1/home/sa305415/shohin/runtimes/tcs1_semantic_9d0f109_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-391-tcs1_semantic_9d0f109_r1-20260811|389120|31
DELETED_PERMANENT|392|2026-08-11T09:34:45.770184Z|/lustre/fs1/home/sa305415/shohin/runtimes/vfr1_514ce86_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-392-vfr1_514ce86_r1-20260811|211337216|3451
DELETED_PERMANENT|393|2026-08-11T09:34:46.688946Z|/lustre/fs1/home/sa305415/shohin/runtimes/vfr1_699ddf6_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-393-vfr1_699ddf6_r1-20260811|211337216|3451
DELETED_PERMANENT|394|2026-08-11T09:34:47.641985Z|/lustre/fs1/home/sa305415/shohin/runtimes/vfr1_3dc9d29_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-394-vfr1_3dc9d29_r1-20260811|211353600|3453
DELETED_PERMANENT|395|2026-08-11T09:34:48.596936Z|/lustre/fs1/home/sa305415/shohin/runtimes/vfr1_671eca0_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-395-vfr1_671eca0_r1-20260811|211423232|3462
DELETED_PERMANENT|396|2026-08-11T09:34:49.581268Z|/lustre/fs1/home/sa305415/shohin/runtimes/vfr1_1da7300_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-396-vfr1_1da7300_r1-20260811|211435520|3465
DELETED_PERMANENT|397|2026-08-11T09:34:50.541364Z|/lustre/fs1/home/sa305415/shohin/runtimes/cfr1_6dc5eff_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-397-cfr1_6dc5eff_r1-20260811|211456000|3467
DELETED_PERMANENT|398|2026-08-11T09:34:51.495378Z|/lustre/fs1/home/sa305415/shohin/runtimes/cfr1_fe80947_r2|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-398-cfr1_fe80947_r2-20260811|211472384|3470
DELETED_PERMANENT|399|2026-08-11T09:34:52.429642Z|/lustre/fs1/home/sa305415/shohin/runtimes/cfr1_cda70e5_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-399-cfr1_cda70e5_r1-20260811|211488768|3471
DELETED_PERMANENT|400|2026-08-11T09:34:53.382645Z|/lustre/fs1/home/sa305415/shohin/runtimes/cfr1_79b5272_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-400-cfr1_79b5272_r1-20260811|211492864|3472
DELETED_PERMANENT|401|2026-08-11T09:34:54.319041Z|/lustre/fs1/home/sa305415/shohin/runtimes/ndr1_c3ebdb3_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-401-ndr1_c3ebdb3_r1-20260811|211521536|3475
DELETED_PERMANENT|402|2026-08-11T09:34:55.263951Z|/lustre/fs1/home/sa305415/shohin/runtimes/ndr1_31606c8_r1|/lustre/fs1/home/sa305415/shohin/runtimes/.pcf1-age-delete-B2b-402-ndr1_31606c8_r1-20260811|211525632|3476
```

The immediate completion observation was
`839,119,116 KiB / 768,478` inodes. Lustre accounting then settled at
`838,918,136 KiB / 768,478` inodes, identical across observations at
`2026-08-11T09:35:17Z`, `09:35:38Z`, and `09:35:57Z`; each observation
was accompanied by an empty user scheduler listing. Relative to the settled
pre-mutation baseline, actual recovery is
`228,108,156 KiB / 700,823` quota inodes. Actual byte recovery exceeds the
conservative hardlink-adjusted projection by `1,212 KiB`, attributable to
allocation released from the three retained parent directories; inode
recovery exactly matches projection.

Settled usage is now `838,918,136 KiB / 768,478` inodes, which is below both
quota limits and leaves `220,143,624 KiB / 241,522` inodes of hard-limit
headroom. This exceeds the 128-GiB / 150,000-inode PCF1 admission target by
`85,925,896 KiB / 91,522` inodes. B2b stops here: no additional target was
selected, renamed, or deleted.

An independent read-only postcheck then reproduced the settled quota and
scheduler state, found all 20 protected anchors present, found no
`.pcf1-age-delete-B2b-*` remnant at depth one beneath any of the three target
roots, and extracted exactly 402 unique deletion records with indices
`001`--`402`. Rehashing those records with one terminating newline reproduced
`a3a58a160206a0f676cf2faad18234ef7dbb448871651bb4a097cdd1c68b907a`.
No write or deletion was made by the independent postcheck.

### PCF17 terminal headroom maintenance — 2026-08-12

After PCF17 stopped and the scheduler was empty, two explicitly bounded
cleanup transactions restored the frozen 128-GiB/150,000-inode admission
margin. Both transactions used literal absolute paths, verified ownership,
nonsymlink roots, resolved-path containment, protected-anchor exclusion, and
same-parent quarantine before one-filesystem permanent deletion.

The first transaction removed only the regenerable Bazel cache
`/lustre/fs1/home/sa305415/.cache/bazel`: 274,456 entries. The second removed
eight old closed Shohin scratch runtimes, totaling 17,077 entries and
2,137,333,760 allocated bytes:

- `.invalid_shohin_ettr_smollm2_runtime_59734ed_r1`
- `.shohin_ettr_native_disposition_runtime_3f9add1_r2_build`
- `hsc1_rank_c7b8aed_r2`
- `hsc1_rank_249ec7f_r1`
- `shohin_finepdf_policy_runtime_bed4596_r1`
- `shohin_finepdf_policy_runtime_bed4596_r2`
- `shohin_finepdf_policy_runtime_e587807_r2`
- `shohin_finepdf_policy_runtime_e587807_r1`

All originals and quarantine names were confirmed absent, so these deletions
are permanently nonrecoverable locally. The PCF17 evidence root, qualified 9B
release, pinned Ministral host/runtime, repositories, credentials, and current
research documents remained present. Settled quota after cleanup was
`850,234,668 / 1,059,061,760 KiB` and `857,586 / 1,010,000` inodes, leaving
`208,827,092 KiB` and `152,414` inodes of hard-limit headroom. Cleanup stopped
at that target.
