# Shohin Execution Hold — 2026-08-15

Status: user-requested cluster pause applied at 2026-08-15 18:59 EDT.

## Scheduler state

All pending Shohin work owned by `sa305415` was placed in Slurm user hold,
including Q36 validation, its downstream score/analysis jobs, Nemotron Super,
and Mixtral mechanics and dependents. The settled queue contains:

- `210` pending tasks, all with reason `JobHeldUser`;
- `0` running tasks; and
- `0` completing tasks.

No dense public benchmark campaign job was submitted.

## Active allocation disposition

At the hold boundary one Q36 revision shard was active: array identity
`759816_5`, represented by replacement allocation `760606` on `evc30`.
Slurm rejected `requeuehold` because the scientific jobs were intentionally
submitted with requeue disabled. The parent graph was held first, then only
this active allocation was canceled so the GPU would not remain occupied for
the multi-day pause.

Final accounting for `760606` is:

- state: `CANCELLED by 1227834669`;
- elapsed: `457` seconds;
- start: `2026-08-15T18:52:19`;
- end: `2026-08-15T18:59:56`; and
- node: `evc30`.

It produced no completed scientific result. Resumption must explicitly submit
exactly one replacement for Q36 revision shard 5, using the already-frozen
identity partition and scientific bytes. It must not release both a replacement
and the canceled allocation, and it must not treat the canceled attempt as a
benchmark observation.

## Preserved graph

The following root jobs and their dependencies remain held rather than
deleted: Q36 validation arrays `759816`, `759817`, `759818`; Nemotron Super
mechanics `760382` through its staged dependents; Mixtral mechanics `760565`
through its staged dependents; and the earlier held Q36 variant/validation
graph. No job definition, identity partition, checkpoint, prompt, threshold,
or output was changed by the pause.

At the settled hold boundary Lustre usage was `866,439,696 KiB` and `535,785`
inodes against hard limits of `1,059,061,760 KiB` and `1,010,000` inodes.

## Resume checklist

1. Re-read this receipt and verify the queue still contains only intended held
   work.
2. Verify every completed output and immutable input hash before release.
3. Submit one exact replacement for canceled revision shard 5.
4. Release only the desired graph roots; do not bulk-release obsolete held
   variants.
5. Recheck quota, node exclusions, dependencies, and `--no-requeue` before
   admitting GPU work.
