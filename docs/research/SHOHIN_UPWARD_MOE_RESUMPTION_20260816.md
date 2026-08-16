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

Fresh pre-execution replay also verified the complete Nemotron Super input
closure: model revision `7d7e5797b8a3c7abbab54033b6004e93e8b6bc91`, config SHA-256
`ff5d6d643b288d4149b0bf820ecb5fe87dd9bbc08b6b811241c57840e11e30e3`,
151/151 runtime members, and 3,802/3,802 GLIBC/FP8 overlay members. The
training input has exactly 9,655 rows; the matched source and assessor inputs
have exactly 256 rows each, with assessor SHA-256
`ac665433d40c0f492744e1152bfabc0e960dfb2d2e4ced8c15c7385a1e387351`.
The custom `NemotronH` config exposes no native tensor-parallel plan, so its
preserved two-local-H100 graph remains the executable semantics; the
independent-node TP fallback is used only for Mixtral, whose native TP plan is
qualified by mechanics.

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

## Automatic temporal-architecture continuation

The first free H100 on `evc31` passed job `760728` in 30 seconds: PyTorch saw
one `NVIDIA H100 PCIe` with 85,017,624,576 bytes, completed an 8192-square
FP16 GEMM with finite output, and observed 816 GiB free on the node-local
filesystem. Because the scientific jobs require both devices, this single-GPU
pass does not by itself admit the node. Dual-H100 qualification `760729` is
dependency-staged after the existing one-GPU occupant `760640`; CPU custodian
`760730` will re-parse the exact two-device result and remove only `evc31`
from the ten Super/Mixtral GPU-job exclusion lists if the dual test completes
`0:0` with zero restarts. A failed qualification cannot alter the lists.

The result-dependent temporal promotion path was also activated in place:

- selector `760596` waits for scores `760388` and `760571`;
- launcher `760598` waits for selector `760596` and can create the exact
  owner/draft/aligned-revision/temporal-gate graph only for the selected host.

Before release, selector runtime manifest
`7d1de76fcd44b63de94d8ff517134bf5bb054a033eeeec3af978a566265374ca`
and launcher runtime manifest
`08db2f1282a406e8d7dd3ee9c5691294b071992e1d4fa24426a0c433b96398dd`
passed full manifest replay; their promotion, run, and automation roots were
absent. The exact selector/launcher/analyzer/renderer suite passed 37 tests.
This makes the staged work an end-to-end architecture transfer rather than a
plain host-baseline comparison: matched revision screens choose the stronger
large family, and only then does that host receive the 35B-leading temporal
causal architecture.

## Independent-H100 tensor-parallel fallback

The scheduler does not need to place every model shard on one node. A fresh
fallback now accumulates independent one-H100 allocations and joins them with
PyTorch native tensor parallelism over NCCL/InfiniBand. This path does not
alter or cancel the frozen two-local-H100 graphs above and reads no benchmark
rows during qualification.

The first two-rank launch exposed two bounded infrastructure facts:

- jobs `760731`/`760732` were admitted simultaneously on `evc31`/`evc48`,
  proving separate one-H100 requests can form the distributed world; they
  exited in two to three seconds because the pinned environment has no
  `torchrun` console script. Runtime `6d08d7b9` repaired only the launcher to
  use `python -m torch.distributed.run`;
- jobs `760733`/`760734` then verified the full 281 GB model manifest,
  initialized NCCL across the same two nodes, and loaded 317 of 507 weight
  groups before both ranks reached 79.13 GiB. The exact traceback showed that
  Transformers native TP had materialized BF16 shards rather than applying
  BitsAndBytes NF4. Both failed after 601 seconds with zero restarts and before
  model forward, training data, or benchmark access. Stderr hashes are
  `779bf326c37556eb90f42198f5f1cfa4fab133bf9eeb8d0cafe60c8bda38c7c6`
  and `69a16185ba1c7f667082aaa5e134fb16aa8bfa3b4476ab3e5c42a9f6ada40574`.

The corrected geometry is explicit BF16 TP4: four independent one-H100 ranks,
approximately 70.3 GB of immutable weight bytes per rank, identical controls
and treatment precision, replicated Shohin trainables with gradient
all-reduction, and native routers/experts frozen. Runtime manifest
`931d64a5f6fcd0b874bba91449da8395cadde1c76281e048fe945dca93001a3f`
binds source commit `ca994c07d6e1a89144ee185edd6059cabe991068`.
Jobs `760739`--`760741` are currently admitted on `evc31`, `evc48`, and
`evc49`; independent request `760742` is accumulating the fourth rank. The
three admitted jobs wait inside a two-hour rendezvous bound and perform no
model or benchmark access until all four ranks are present.

The remainder of this independent-H100 graph is dependency-staged, with no
idle reservation and no duplicate model load inside evaluation:

- revision-training ranks `760743`--`760746` wait for all four mechanics
  ranks and preserve the fixed 9,655-presentation, 2,048-consumption,
  256-update geometry (`revision_train.jsonl` SHA-256
  `802c85662570c5bcb72f3e4430dbd093e901081f114213831292750894c3feff`);
- matched-evaluation ranks `760747`--`760750` wait for all four training ranks
  and evaluate unchanged, self-refinement, and trained revision in that fixed
  order from one TP4 BF16 model load over the same source-disjoint 256-row
  screen (`screen_sources.jsonl` SHA-256
  `f0b7830814762c6917363642e86edaaf192a8ab2834911c13c0cae9255ceefa9`);
- CPU score job `760751` waits for all four evaluation ranks and is the only
  job bound to the 256-row assessor board (SHA-256
  `ac665433d40c0f492744e1152bfabc0e960dfb2d2e4ced8c15c7385a1e387351`).
- CPU curve job `760752` waits for the new Mixtral score and preserved
  Nemotron score, then combines those points with the immutable 35B Qwen
  trained-revision screen (SHA-256
  `dbf6caa1c4f0546c1d1d11d0490edfa5e74ae573613c2978577a7ea5ab1941bb`);
  renderer `760753` waits for that analysis and emits the hash-manifested SVG
  and point CSV under the separate TP4-BF16 publication root.

The training runtime manifest is
`716a6bd5bc6d79f498abe5b2812e13a37d6f97bd660d9b2ade78a6c71a8765b1`;
the one-load matched-evaluation/scoring runtime manifest is
`a89bc0f2ee55cc252b989a85cf03d0ff1a4931b235e5d4369f0b141c673e606e`.
Every GPU stage is four independent single-H100 Slurm requests joined across
nodes by NCCL/InfiniBand. Rank `760742` has a scheduler reservation for
2026-08-16 06:24:30 EDT; its one-hour allocation still exceeds the measured
post-rendezvous mechanics duration by more than threefold.
