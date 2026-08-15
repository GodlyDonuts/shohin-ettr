# Q36 upward-MoE storage reclamation — 2026-08-15

This manifest was recorded before mutation.  It contains only obsolete external
model caches from closed dense, small-MoE, control-family, and prohibited Q35
lanes.  Dense scaling is complete; these weights are reproducible from their
upstream revisions, while their compact conclusions and hashes remain in Git.

The executable batch contains exactly 8 resolved, nonsymbolic, regular-file-only
directories owned by `sa305415`: 61,247,893,504 allocated bytes (57.04 GiB)
and 255 path inodes.  An initial precheck failed closed before mutation because
the proposed 229,376-byte `hf_cache` contained one interior Hugging Face
symlink; that cache was removed from the batch and remains untouched.

| Mtime (EDT) | Allocated bytes | Inodes | Exact target |
|---|---:|---:|---|
| 2026-07-31 17:45:20 | 272,490,496 | 13 | `/lustre/fs1/home/sa305415/shohin/artifacts/external/smollm2-135m-instruct-83212e1` |
| 2026-08-02 21:46:00 | 1,769,992,192 | 35 | `/lustre/fs1/home/sa305415/shohin/artifacts/external/qwen3.5-0.8b-2fc0636` |
| 2026-08-03 04:10:28 | 6,167,617,536 | 32 | `/lustre/fs1/home/sa305415/shohin/artifacts/external/smollm3-3b-a07cc9a` |
| 2026-08-05 01:22:53 | 15,242,944,512 | 50 | `/lustre/fs1/home/sa305415/shohin/artifacts/external/qwen2.5-coder-7b-instruct-c03e6d3` |
| 2026-08-05 07:11:09 | 2,109,440 | 2 | `/lustre/fs1/home/sa305415/shohin/artifacts/external/smollm2_135m_instruct_83212e1e` |
| 2026-08-07 17:58:50 | 9,342,881,792 | 31 | `/lustre/fs1/home/sa305415/shohin/artifacts/external/qwen3.5-4b-851bf6e` |
| 2026-08-08 20:54:55 | 14,607,130,624 | 49 | `/lustre/fs1/home/sa305415/shohin/artifacts/external/olmo2-7b-instruct-470b1fba` |
| 2026-08-09 00:06:29 | 13,842,726,912 | 43 | `/lustre/fs1/home/sa305415/shohin/artifacts/external/olmoe-1b-7b-0125-instruct-b89a7c4` |

Explicit exclusions are the active Qwen3.6 35B-A3B host; the staged Nemotron
Super 120B-A12B host and overlay; the qualified Qwen3.5 9B release; the pinned
Ministral PCF host; all current Q36 checkpoints, sources, candidates, scores,
and runtimes; repository/configuration/credentials; and unrelated workspaces.

Deletion is authorized as permanent and locally nonrecoverable.  Each target
must be revalidated byte/inode-identical, resolved inside the literal external
artifact root, nonsymbolic, same-user owned, and disjoint from the exclusions;
then it may be renamed to a unique same-parent quarantine and only that exact
quarantine removed one filesystem at a time.

## Execution receipt

The eight amended targets revalidated exactly and were renamed one at a time to
literal same-parent `.q36-upward-delete-20260815-*` quarantines.  Only those
quarantines were deleted, and every original/quarantine is absent.  The
deletion is permanent and not locally recoverable.

- Pre-delete quota: 976,289,460 KiB and 700,938 inodes.
- Stable post-delete quota at 10:10:14Z, 10:10:34Z, and 10:10:54Z:
  916,477,064 KiB and 700,683 inodes.
- Exact recovered quota: 59,812,396 KiB (61,247,893,504 bytes) and 255
  inodes, matching the amended manifest.
- Current hard-limit headroom: 142,584,696 KiB and 309,317 inodes.
- The excluded symbolic `hf_cache` remains present and untouched.
- Every explicit Q36/Nemotron/Qwen9B/Ministral protected anchor remained
  present after the transaction.

## Batch 2 manifest — closed dense pretraining shards

This manifest was recorded before mutation. Dense scaling and the phase-2
small-model lane are closed, and none of the active Q36 or Nemotron Super batch
scripts references `/lustre/fs1/home/sa305415/shohin/artifacts/shards`. The 25
literal child directories below are resolved, nonsymbolic, owned by
`sa305415`, and contain 78,086,721,536 allocated bytes (72.72 GiB) across 668
path inodes. The parent shard root itself is not a deletion target.

| Mtime (EDT) | Allocated bytes | Inodes | Exact target |
|---|---:|---:|---|
| 2026-07-31 07:05:15 | 404,815,872 | 7 | `/lustre/fs1/home/sa305415/shohin/artifacts/shards/essential_web_phase2_exact_r1` |
| 2026-08-08 17:39:57 | 395,104,256 | 6 | `/lustre/fs1/home/sa305415/shohin/artifacts/shards/essential_web_phase2_exact_smollm2_49k_r4` |
| 2026-07-30 05:03:01 | 404,819,968 | 7 | `/lustre/fs1/home/sa305415/shohin/artifacts/shards/essential_web_reasoning_core_1b` |
| 2026-08-08 17:44:19 | 126,889,984 | 4 | `/lustre/fs1/home/sa305415/shohin/artifacts/shards/finepdf_core_smollm2_49k_r2` |
| 2026-07-29 21:37:33 | 8,044,449,792 | 64 | `/lustre/fs1/home/sa305415/shohin/artifacts/shards/finepdfs_edu_eng_core_10b` |
| 2026-07-30 00:26:37 | 129,871,872 | 4 | `/lustre/fs1/home/sa305415/shohin/artifacts/shards/finepdfs_edu_eng_policy_core_100m_bed4596` |
| 2026-07-29 23:23:45 | 4,096 | 2 | `/lustre/fs1/home/sa305415/shohin/artifacts/shards/finepdfs_edu_eng_policy_core_100m_e587807.partial` |
| 2026-07-30 00:05:17 | 131,366,912 | 4 | `/lustre/fs1/home/sa305415/shohin/artifacts/shards/finepdfs_edu_eng_policy_residual_100m_bed4596` |
| 2026-07-29 23:35:21 | 131,366,912 | 4 | `/lustre/fs1/home/sa305415/shohin/artifacts/shards/finepdfs_edu_eng_policy_residual_100m_e587807.partial` |
| 2026-07-31 07:24:59 | 13,418,856,448 | 91 | `/lustre/fs1/home/sa305415/shohin/artifacts/shards/fineweb_edu_phase2_exact_r1` |
| 2026-08-08 20:27:47 | 13,079,957,504 | 88 | `/lustre/fs1/home/sa305415/shohin/artifacts/shards/fineweb_edu_phase2_exact_smollm2_49k_r2` |
| 2026-07-28 21:27:03 | 4,096 | 1 | `/lustre/fs1/home/sa305415/shohin/artifacts/shards/fineweb_edu_score4_core_10b.partial` |
| 2026-07-29 02:52:37 | 8,327,168 | 2 | `/lustre/fs1/home/sa305415/shohin/artifacts/shards/fineweb_edu_score4_core_10b_r1.partial` |
| 2026-07-30 07:29:13 | 13,418,790,912 | 91 | `/lustre/fs1/home/sa305415/shohin/artifacts/shards/fineweb_edu_score4_core_10b_r2` |
| 2026-07-29 02:59:56 | 6,062,080 | 10 | `/lustre/fs1/home/sa305415/shohin/artifacts/shards/fineweb_edu_tokenizer_batch_canary_754466` |
| 2026-07-31 07:01:43 | 78,684,160 | 4 | `/lustre/fs1/home/sa305415/shohin/artifacts/shards/formal_logic_phase2_exact_r1` |
| 2026-07-29 07:44:40 | 78,696,448 | 5 | `/lustre/fs1/home/sa305415/shohin/artifacts/shards/nemotron_formal_logic_13fa979_challenger_128m` |
| 2026-07-29 10:28:54 | 5,331,808,256 | 47 | `/lustre/fs1/home/sa305415/shohin/artifacts/shards/pes2o_domain_balanced_50m_r1` |
| 2026-07-29 14:01:18 | 5,331,804,160 | 47 | `/lustre/fs1/home/sa305415/shohin/artifacts/shards/pes2o_domain_balanced_50m_r1_sensitive_residual_v1` |
| 2026-08-11 03:55:34 | 20,869,120 | 2 | `/lustre/fs1/home/sa305415/shohin/artifacts/shards/pes2o_selected_core_10b.partial` |
| 2026-07-29 09:30:01 | 11,900,739,584 | 103 | `/lustre/fs1/home/sa305415/shohin/artifacts/shards/pes2o_selected_core_10b_r2` |
| 2026-08-08 18:53:46 | 5,233,700,864 | 46 | `/lustre/fs1/home/sa305415/shohin/artifacts/shards/pes2o_sensitive_smollm2_49k_r2` |
| 2026-07-29 04:07:00 | 4,743,168 | 10 | `/lustre/fs1/home/sa305415/shohin/artifacts/shards/pes2o_tokenizer_batch_canary_754483` |
| 2026-08-08 16:35:14 | 4,096 | 1 | `/lustre/fs1/home/sa305415/shohin/artifacts/shards/phase2_holdouts_3ab418c_r1` |
| 2026-08-08 16:52:26 | 404,983,808 | 18 | `/lustre/fs1/home/sa305415/shohin/artifacts/shards/phase2_holdouts_3ab418c_r2` |

Every target must be revalidated against the exact bytes/inodes above and
checked disjoint from the live Qwen3.6, Nemotron Super, Q36 sources,
checkpoints, candidates, scores, runtimes, and repository/configuration roots.
Each target may then be renamed to a unique same-parent quarantine and only
that literal quarantine permanently removed with one-filesystem protection.
