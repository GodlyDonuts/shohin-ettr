# Shohin upward-MoE execution resumption — 2026-08-16

## Outcome

The user ended the temporary execution hold after restoring the UCF VPN path.
Newton access was recovered at `2026-08-16T04:15:46-04:00`, and the two
already-staged cross-family upward-MoE graphs were selectively released without
resubmission, reshaping, or scientific-byte changes:

- Nemotron Super-120B-A12B: mechanics `760382`, fit `760384`, matched
  unchanged/self-refinement/revision arrays `760385`–`760387`, score `760388`.
- Mixtral-141B-A39B: mechanics `760565`, fit `760566`, matched
  unchanged/self-refinement/revision arrays `760567`–`760569`, score `760571`.
- Joint scaling analysis `760575` and publication renderer `760588`.

Both mechanics roots are pending healthy two-H100 capacity. Downstream jobs are
released but remain protected by their original `afterok` dependencies. Every
job retains `Requeue=0`; no duplicate job or output was created.

## Immutable input replay

Before release, the following exact inputs were present and read-only:

| Artifact | SHA-256 |
|---|---|
| Nemotron runtime manifest | `1184ea2b254a807cb8517e6c9ae239023b79b45c84498db2b4e71186d54194cb` |
| Nemotron model manifest | `8bb8bb898794651791de9d79c1041fe0ec6ad0f54a97b03f52620bd6e245ce92` |
| Nemotron GLIBC-2.28 overlay manifest | `cde0fa5b91d50d1509872cbc577cf016d0a6c6697bfb066d607f420c1b568e84` |
| Mixtral runtime manifest | `c98f7df71d89de763db101d2eadea5ae75e48505d735f2d1de432950063aadd4` |
| Mixtral model manifest | `46b8475d98e2a49f9a81329287beb9d450dfd4d7a74886e8780708764a8f3fe7` |
| Matched revision training data | `802c85662570c5bcb72f3e4430dbd093e901081f114213831292750894c3feff` |
| Shared 256-row screen source | `f0b7830814762c6917363642e86edaaf192a8ab2834911c13c0cae9255ceefa9` |

All mechanics reports, 256-update checkpoints, and score outputs were absent
before release, proving that this is activation of the preserved graph rather
than duplicate execution.

## `evc33` qualification result

`evc33` reported `IDLE` with two H100 PCIe devices after a Slurmd restart, but
it remained on the frozen exclusion list because of repeated historical CUDA
failures. Two bounded, non-scientific diagnostics tested the apparent repair:

1. Job `760725` requested both H100s. It received `CUDA_VISIBLE_DEVICES=0,1`
   but hung at the first `nvidia-smi -L`, emitting no device row.
2. Array `760726` requested the GPUs as two independent one-H100 allocations,
   as tasks `760726` and `760727`. Both received one visible device and hung
   before PyTorch could publish its CUDA/GEMM result.

The diagnostics were canceled after 25 and 57 seconds respectively, with zero
restarts and no model or benchmark access. Their immutable stdout hashes are:

- dual-H100 stdout: `4a1a6cc76e68fa10ffaa8963f2f07cbba0c10f55f58b9b9ac3f068cd7ba9a212`
- single-H100 task 0 stdout: `96d1bd563becaee79932c96b252561ed067f6c81b2be1ddd0acd74e2b6a37140`
- single-H100 task 1 stdout: `755e54f554d05ab0a1fdf6b4796d38c644054967485024292342628a175cdd47`

Conclusion: scheduler availability did not imply usable CUDA hardware.
`evc33` remains excluded. No scientific graph was exposed to it.

## Capacity and storage

At release, all non-excluded H100 nodes had at least one GPU allocated by
another user; therefore no healthy two-H100 node was immediately available.
Nemotron mechanics was pending `Resources`, and Mixtral mechanics was pending
`Priority`. The dependency graphs are now eligible and will admit
opportunistically when a healthy full node clears.

Lustre usage was `866,439,700 / 1,059,061,760 KiB` and
`535,785 / 1,010,000` files, leaving `192,622,060 KiB` and `474,215` inodes of
hard-limit headroom.

## Scientific boundary

The 35B anchor remains Qwen3.6-35B-A3B temporal causal gating at `143/256`
versus unchanged `111/256`. The released graphs test whether the same
model-owned temporal-revision principle transfers across both family and
scale: NVIDIA Nemotron at 120B total/12B active and Mistral Mixtral at 141B
total/39B active. No scaling claim is made until both matched score artifacts
close.
