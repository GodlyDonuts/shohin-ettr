# Q36-MTR storage reclamation ledger — 2026-08-13

This ledger records the bounded storage repair performed after the
`q36-mtr-bb06a34a-r1` draft fan-out terminated before candidate publication.
The scientific host, source data, prompts, arms, controls, seeds, update
budgets, and gate are not changed by storage repair.

## Admission failure

At `2026-08-13T11:31:22Z`, Lustre usage was `888,200,808 KiB` and
`1,724,279` inodes against hard limits of `1,059,061,760 KiB` and
`1,010,000` inodes. The byte limit remained healthy, but the inode limit was
exceeded. The Q36 run root contained no draft candidate files and was not the
source of the approximately 944,000-inode increase from the preceding
admission snapshot.

## Destructive batch C1 — recorded before mutation

Scheduler job `754705`, an unrelated TensorFlow/Bazel validation allocation,
had completed successfully and no process owned by `sa305415` matched Bazel
or Bazelisk. The following two literal roots were regular directories owned by
`sa305415`, were not symlinks, and contained only regenerable Bazel caches:

| Literal target | Root mtime | Allocated bytes | Inodes |
|---|---:|---:|---:|
| `/lustre/fs1/home/sa305415/.cache/bazel` | `2026-08-13T09:33:47Z` | `25,120,747,520` | `644,309` |
| `/lustre/fs1/home/sa305415/tf-bazel-disk-cache` | `2026-08-13T09:41:33Z` | `9,627,353,088` | `107,865` |

The exact combined projection is `34,748,100,608` allocated bytes and
`752,174` inodes. TensorFlow source/worktrees, Shohin repositories, model
roots, scientific artifacts, qualified runtimes, Q36 owner/mechanics evidence,
credentials, and configuration are outside the targets and must remain.

## Destructive batch C2 — recorded before mutation

After C1, quota settled at `854,296,028 KiB` and `972,105` inodes,
leaving only `37,895` inodes below the hard limit. Batch C2 contains the 62
literal direct-child runtime packages below. They are closed historical Shohin
lanes, every root and descendant is owned by `sa305415`, no root is a symlink,
and no hard-linked inode has a link outside this exact target set. The two
qualified release runtimes
`/lustre/fs1/home/sa305415/shohin/runtimes/idr_aqc_8f0bd8d_r1` and
`/lustre/fs1/home/sa305415/shohin/runtimes/idr_aqc_8f0bd8d_r2` are explicitly
excluded and preserved.

| Literal target | Unix mtime | Allocated bytes | Path inodes |
|---|---:|---:|---:|
| `/lustre/fs1/home/sa305415/shohin/runtimes/mpr2_01d2d96_r1` | `1786300396` | `211714048` | `3504` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/mpr2_378a10a_r1` | `1786300735` | `211714048` | `3504` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/dpr1_70441df_r1` | `1786301691` | `211738624` | `3509` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/dpr1_be79bd8_r1` | `1786302630` | `211767296` | `3514` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/obr1_45be1b7_r1` | `1786303478` | `211816448` | `3522` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/obr1_aeb3541_r1` | `1786303813` | `211841024` | `3525` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/obr1_41b3298_r1` | `1786304053` | `211841024` | `3525` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/mpr3_b0c4ec8_r1` | `1786304348` | `211849216` | `3526` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/dseo1_3cf187a_r1` | `1786308944` | `211968000` | `3542` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/dseo1_efffcdb_r1` | `1786309118` | `211968000` | `3542` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/dseo1_76bddea_r1` | `1786309440` | `211984384` | `3545` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/dseo1_3467a6c_r1` | `1786309522` | `211984384` | `3545` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/dseo1_d17b093_r1` | `1786309925` | `211984384` | `3545` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/dsec0_local_r1` | `1786311925` | `16384` | `3` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/dset1_data_r1` | `1786312366` | `45056` | `8` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/dset1_data_r2` | `1786312695` | `45056` | `8` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/dset1_data_r3` | `1786312873` | `45056` | `8` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/dset1_data_r4` | `1786313030` | `45056` | `8` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/dset1_data_r5` | `1786313133` | `45056` | `8` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/dset1_data_r6` | `1786313350` | `45056` | `8` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/dset1_1e4ce9d_r2` | `1786313409` | `212074496` | `3556` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/dset1_49709a1_r1` | `1786313409` | `212054016` | `3554` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/dset_q35_52984a8_r1` | `1786313409` | `212197376` | `3576` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/dset_q35t_2ea44dc_r1` | `1786313409` | `212250624` | `3586` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/dset_q35t_79c0e24_r4` | `1786313409` | `212307968` | `3596` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/dset_q35t_aff59ea_r2` | `1786313409` | `212271104` | `3590` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/dset_q35t_c3689d4_r5` | `1786313409` | `212328448` | `3600` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/dset_q35t_c3fa029_r3` | `1786313409` | `212271104` | `3590` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/pset1_50fd89c_r4` | `1786313409` | `212164608` | `3568` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/pset1_char_data_r2` | `1786313409` | `212090880` | `3558` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/pset1_char_train_r3` | `1786313409` | `212119552` | `3562` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/pset1_fa278a9_data_r1` | `1786313409` | `212090880` | `3558` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/gset1_f02dc45_r1` | `1786326901` | `212410368` | `3611` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/gset1_f02dc45_r2` | `1786326933` | `212410368` | `3611` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/iset1_6a611f5_r1` | `1786328491` | `212430848` | `3614` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/wtv1_2128cfd_r1` | `1786339028` | `212529152` | `3625` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/wtv1_8ed48d4_r2` | `1786339028` | `212525056` | `3625` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/wtv1_8ed48d4_r3` | `1786339219` | `212504576` | `3624` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/tsvc1_923369b_r1` | `1786339623` | `152010752` | `3580` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/tsvc1_25f1885_r2` | `1786339726` | `212443136` | `3620` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/tsvc1_data_dfab010_r3` | `1786339779` | `24576` | `4` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/tsvc1_data_21a1ace_r4` | `1786339869` | `24576` | `4` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/tsvc1_data_57c50db_r5` | `1786339914` | `24576` | `4` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/tsvc1_compare_879d65b_r1` | `1786340722` | `20480` | `4` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/etv1_15c7ad9_r1` | `1786340873` | `212455424` | `3621` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/etv1_b8482f4_r2` | `1786340978` | `212459520` | `3621` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/etv1_11457b3_r3` | `1786341187` | `212447232` | `3621` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/etv1_compare_2a415b3_r1` | `1786349882` | `28672` | `6` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/fret1_1397e67_r1` | `1786352180` | `212353024` | `3604` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/fret1_compare_e51d142_r1` | `1786352311` | `28672` | `6` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/rift1_36b8de0_r1` | `1786353149` | `212377600` | `3607` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/ocet1_data_17eaa99_r1` | `1786353564` | `24576` | `6` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/ocet1_materialize_1ace2e3_r1` | `1786354418` | `32768` | `6` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/ocet1_materialize_a412eb59d117322c6abffc26ad8e3ad857624e5b_r2` | `1786354796` | `28672` | `6` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/ocet1_materialize_a593a463c73cdc2e16abd9869b424b6fde48f82b_r3` | `1786354855` | `36864` | `8` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/ocet1_5d33fa18ed739ede6cc75d90c81f4874887ecb76_r1` | `1786355114` | `176128` | `19` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/ocet1_eval_99d3657a873d53de58b5007547decba66f796f90_r1` | `1786355468` | `225280` | `19` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/ocet1_compare_78a5d4331b34ff2e0d8f1d665d6830ba5ec73911_r1` | `1786364370` | `24576` | `6` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/rsot1_eval_dae357e032c0e80f518951db11f0247909b47d4a_r1` | `1786364545` | `212992` | `17` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/bsot1_eval_abd1d35_r1` | `1786365776` | `217088` | `18` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/bsot1_eval_b922115_r1` | `1786365887` | `221184` | `18` |
| `/lustre/fs1/home/sa305415/shohin/runtimes/bsot1_eval_c365beb_r1` | `1786366025` | `221184` | `18` |

The exact conservative aggregate is `8,003,608,576` allocated bytes and
`135,946` quota inodes. There are zero outside-linked inodes. The expected
post-batch headroom is `173,841` inodes, above the frozen 150,000-inode
admission floor.

## Execution receipts

### Batch C1

The first launch failed closed before mutation because a broad process-pattern
check matched the audit shell itself. A direct executable-name and scheduler
check found no live Bazel process or job. The identical two targets were then
renamed to unique same-parent quarantines and permanently deleted one
filesystem at a time. Both original and quarantine paths are absent. The
settled post-C1 observation was `854,296,028 KiB` and `972,105` inodes, an
observed recovery of `33,904,780 KiB` and exactly `752,174` inodes. These cache
contents are not locally recoverable.

### Batch C2

The target list contained exactly 62 entries and matched SHA-256
`8501ba54bf780dfa08aeed293174d4fc7486244cefcabd4aff7f084639e9a557`.
The first target was atomically renamed, after which removal failed closed
because the immutable package was non-writable. No second target had been
renamed. Owner write permission was added only inside that exact quarantine;
it was permanently removed, and the remaining 61 targets matched digest
`03373821893be19c8118183277589605f6bf7ad4d1e337a1690028478eef0161`
before the same rename, owner-permission, and one-filesystem deletion sequence.
All 62 originals and quarantines are absent and are not locally recoverable.

Three settled quota observations at `2026-08-13T12:07:44Z`,
`2026-08-13T12:08:04Z`, and `2026-08-13T12:08:24Z` were identical:
`846,451,092 KiB` and `836,159` inodes. Final hard-limit headroom is therefore
`212,610,668 KiB` and `173,841` inodes, exceeding the frozen 128-GiB and
150,000-inode admission floors. Total observed recovery from the pre-C1 state
is `41,749,716 KiB` and `888,120` inodes.

The scheduler was empty after cleanup. Both qualified `idr_aqc` runtimes, the
pinned Qwen3.6 and Qwen3.5-9B model roots, the qualified 9B release, the sealed
Q36 owner checkpoint, both repository anchors, and Shohin documentation were
all independently rechecked present. No `.q36-*` quarantine remains.
