# DSET-Q35T: Trained Strong-MoE Transfer

Status: frozen mechanics-first development contract, 2026-08-09. The earlier
untrained Qwen3.6 ceiling is closed and is not reinterpreted by this lane.

## Hypothesis

The DSET edit interface transfers to a current stronger MoE once its compact
script language is learned. The intervention is a host-agnostic shared
post-MLP residual in the final 16 decoder layers. Qwen routers, native experts,
embeddings, attention, and language head remain frozen.

## Frozen treatment and control

- host: pinned Qwen3.6-35B-A3B, loaded in NF4 with BF16 compute;
- quantizer: isolated `bitsandbytes==0.50.0`, manifest-bound in every job;
- adapter: rank 18 after each of the final 16 MLP/MoE blocks, alpha 18,
  exactly 1,179,648 trainables;
- data: a deterministic unchanged-text view of immutable DSET1 `dset1_r6`;
  complete pairs that exceed 4,096 Qwen tokens or the frozen 32-token script
  budget are dropped and receipted before any model output; no truncation,
  shortening, replacement, or source migration is allowed;
- optimization: 256 updates, four paired identities/update, LR `5e-5` cosine,
  identical seeds and token presentation in both arms;
- aligned arm sees exact model-owned drafts;
- hidden arm preserves token/position geometry but causally masks the draft;
- greedy 32-token script decoding and deterministic execution are unchanged.

One update must first demonstrate finite loss/gradient/update, exact parameter
count, zero protected trainables, complete token retention, and peak memory
below one H100. No setting changes are allowed after capability output.

## Gate

On the already-open development diagnostic, aligned must achieve at least 95%
exact executed trajectories, at least 90% exact scripts in numeric and choice
families, at least 99% clean copy, at least 90% fault repair, at least 90%
paired consistency, at least 13 more correct trajectories than hidden, and
zero malformed execution or decode exhaustion. A pass opens one separately
frozen source-disjoint confirmation. A miss closes this exact transfer without
rank, seed, duration, prompt, threshold, or quantization rescue.
