# ETTR Phase 2 Data and Compute Plan

Status: active operational plan, 2026-07-28. This plan authorizes measured
Phase 2 preparation and training only after its named gates pass. The protected
step-300k checkpoint remains a read-only trust root.

## Decision

Shohin needs two distinct training streams.

1. **General pretraining stream.** High-quality language, math, code, science,
   and procedural documents train the 125.08M-parameter trunk's language,
   knowledge, and representation substrate.
2. **ETTR-native stream.** Exact WORLD/COMMAND/QUERY episodes, terminal
   packets, generic transactions, interventions, and invariant views train the
   67.70M new compiler/reactor/query-reader parameters to use the architecture.

The streams are complementary. General text cannot identify the ETTR state
machine, while ETTR episodes are too narrow to supply broad language and world
knowledge.

## Current Inventory

### General data

The historically mounted, evaluation-decontaminated corpus contains
62,426,256,278 manifest tokens:

| source | tokens | current status |
|---|---:|---|
| FineMath 4+ | 6,600,235,115 | historical admitted stream |
| OpenWebMath | 14,063,689,153 | historical admitted stream |
| Python code | 16,762,327,600 | historical admitted stream |
| FineMath 3+ | 25,000,004,410 | historical admitted stream |

Two future candidates add 30,000,001,936 tokens:

| source | tokens | current gate |
|---|---:|---|
| DCLM Baseline residual | 25,000,001,792 | full shard scan and reviewed approval |
| OpenMathInstruct-2 PT | 5,000,000,144 | full shard scan and semantic sample review |

The combined available inventory would be 92,426,258,214 tokens if both
candidates were approved. All six full-corpus structural scans completed on
2026-07-28. Every scanned shard had zero byte-fallback fraction. DCLM's
entropy range was 10.500--10.590 bits/token and OpenMath's was
7.995--8.003 bits/token, with tight token-frequency distributions. Those
measurements establish token-stream integrity, not semantic quality.

Deterministic manual decoding of five DCLM shards found coherent long-form
prose but also low-value forum conversation, dated news, and awkward/SEO-like
text. Five OpenMath shards contained useful worked mathematics but also at
least one visibly unreliable geometry derivation and generated prose that can
sound authoritative while self-correcting or remaining incomplete. Therefore
neither candidate is admitted wholesale. DCLM requires document-level quality
stratification; OpenMath requires answer verification and rejection of
unverifiable or inconsistent solutions before either approval record is
written.

The 4.60B-token FineWeb-Edu `sample-10BT` artifact is an undersized diagnostic
pilot and is not a 25B replacement. The quality-first target remains at least
100--120B admitted unique general tokens, with cross-source near-dedup and
equal-token source ablations before a long run.

### ETTR-native data

The frozen target is 62,500 semantic cores. Each core expands into 64
architecture-native rows and 528 charged positions:

| split | cores | expanded rows | charged positions |
|---|---:|---:|---:|
| train | 40,000 | 2,560,000 | 1,351,680,000 |
| main total | 56,250 | 3,600,000 | 1,900,800,000 |
| sealed confirmation | 6,250 | 400,000 | 211,200,000 |
| total | 62,500 | 4,000,000 | 2,112,000,000 |

As of this plan, 207/216 main receiver-qualification reports and all 108/108
sealed-confirmation reports are durable. Nine long main cells are running
under 48-hour recovery job `753823`. Selector `753824`, task manifest
`753825`, materialization arrays `753826`/`753827`, aggregate audits
`753828`/`753829`, and separation audit `753830` are dependency-held.

No partial or pre-audit ETTR payload may be consumed by training.

## Training Sequence

1. **Resource and transport qualification.** Use the 10-H100 backfill
   allocation for H100 memory sweeps, a bounded multi-node DDP/NCCL canary,
   and ETTR composite-objective throughput measurements. Do not infer quality
   from a transport canary.
2. **Architecture bootstrap.** Run a bounded ETTR-native pilot with the base
   frozen. Require finite losses and gradients, exact checkpoint resume, and
   held-out causal-control improvement before increasing the budget.
3. **Joint continuation pilot.** Interleave general LM updates and ETTR-native
   updates while training the complete 192.78M-parameter system. Freeze
   validation sets and compare at least two stream ratios at equal presented
   positions.
4. **Scale only the winning schedule.** Continue to the long run only if
   general NLL is retained and ETTR treatment improves against query-only,
   state-reset, deranged-binding, and dense-state controls.
5. **Post-training follows later.** Instruction following and user-visible
   reasoning traces are a separate phase and cannot substitute for native
   ETTR causal qualification.

Initial stream ratios are experiment arms, not assumptions. Start with an
ETTR-only architecture bootstrap, then compare 95/5 and 85/15 general/ETTR
charged-position ratios. Repetition, unique positions, and per-source epochs
must be reported separately.

## Newton Capacity

| job | request | expected window at submission | purpose |
|---|---|---|---|
| `719497` | 5 nodes, 10 H100 PCIe, 12h | 2026-07-29 | backfill profiling and scaling |
| `719496` | 10 nodes, 20 H100 PCIe, 72h | 2026-07-30 | gated Phase 2 training |
| `719591` | 4 nodes, 4 H100 PCIe, 4h | scheduler pending | four independent pilot lanes |

Both jobs request two H100s, four CPUs, and 32 GiB per node. Newton advertises
HDR InfiniBand. A same-switch-only constraint delayed both test-only start
estimates, so the live requests remain unconstrained and topology is measured
instead of assumed.

The 20-GPU shape is promoted only if the measured canary is healthy and
efficient. Otherwise the allocation is partitioned into independent 2-, 4-,
or 10-GPU training/ablation lanes. Reserving 20 GPUs does not require forcing
one inefficient 20-way model replica.

## Hard Launch Gates

- exact private source commit installed in a clean shared Newton checkout;
- all requested GPUs pass real BF16 allocation and InfiniBand device checks;
- multi-node rendezvous and NCCL complete without rank loss or hang;
- general shard manifests, scans, approvals, and source weights are frozen;
- ETTR selection, both materializations, both aggregate audits, and separation
  audit pass;
- production record-to-`ETTRContinuationBatch` loader re-hashes every batch;
- exact-resume checkpoint save/load passes after a real optimizer update;
- output directory is fresh, isolated, no-replace, and never aliases the
  protected checkpoint or historical flagship output;
- evaluation and rollback checkpoints are scheduled before the long run.

## Active Jobs

- Newton DCLM scan: `719553`
- Newton OpenMath scan: `719554`
- Newton historical-stream scans: `719561`--`719564`
- Stokes ETTR long-cell recovery and downstream chain:
  `753823`--`753830`

The exact inverse ETTR record materializer is implemented as
`rematerialize_record`. It reconstructs all 64 rows from one frozen
`SemanticCoreRecord`, verifies source/view/token receipts, and requires the
resulting tensor batch to reproduce the stored materialization SHA-256.
All three ontology families and hostile source/target mutations are covered.
The remaining training critical path is a streaming sharded dataset wrapper
and the distributed ETTR trainer/launcher. The bounded generic DDP transport
launcher is `train/jobs/run_reserved_multinode_ddp_canary.sh`.
