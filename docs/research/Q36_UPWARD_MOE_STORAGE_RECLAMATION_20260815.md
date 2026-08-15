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
