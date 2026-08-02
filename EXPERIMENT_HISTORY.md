# Shohin ETTR Experiment History

This file is the index for experimental Git histories that have been folded
into the private repository's canonical `main` history. It exists so rejected,
superseded, incomplete, and custody-only work remains discoverable without
reactivating obsolete source files in the live architecture.

## Archive policy

- Private `main` is the only canonical implementation line.
- A historical branch is incorporated with an ancestry-preserving merge when
  its working tree is obsolete. Its commits remain inspectable with
  `git show <tip>` and `git log <tip>`, but its old tree does not overwrite the
  current architecture.
- `NO-GO`, `rejected`, and `superseded` mean "do not resume without a new
  hypothesis and a new preregistered falsifier." They do not mean that the
  evidence should be deleted.
- Preservation and repository-topology branches are retained as provenance,
  not counted as scientific results.
- The sanitized public repository is a separate publication surface. Private
  ETTR history must never be pushed there.

## Canonical line

| History | Tip before consolidation | Disposition |
| --- | --- | --- |
| `codex/ssqac-law-compiler` | `960333f` | Current private ETTR architecture and evidence line. This becomes private `main`. |
| `codex/acw-g2` | `7433062` | Already contained in the canonical ancestry. Historical ACW custody baseline. |
| `codex/carry-recovery-v9` | `a0c258e` | Already contained in the canonical ancestry. Historical carry-recovery candidate. |
| `codex/public-ssqac-pre-sanitization-20260727` | `78b7653` | Already contained in private ancestry; records the point before public sanitization. |

## Folded experimental histories

| Historical branch | Preserved tip | Classification | Why it is not the live tree |
| --- | --- | --- | --- |
| `codex/cgrfc-mechanics` | `dcfe501` | Superseded mechanics / neural-fit `NO-GO` | Preserves the source-sealed joint compiler, counterfactual-repair, equilibrium, and custody mechanics. The branch's own runbook records that mechanics existed while neural fitting and reasoning promotion remained `NO-GO`. |
| `codex/er-tt-ordinal-route` | `7d1b936` | Rejected route with useful diagnostics | Preserves ordinal witness routing, fresh-board receipts, opcode coupling, and the near-gate marginal-route evidence. Fresh ordinal-route v1 was formally rejected; later work replaced this compiler family. |
| `codex/private-wip-acw-20260727` | `4215256` | Incomplete preservation snapshot | ACW G2 development work was preserved before the public scrub. Its runbook states commit/install/release remained `NO-GO` pending independent review; it is evidence and unfinished machinery, not a promoted result. |
| `codex/private-wip-carry-recovery-20260727` | `539e01d` | Incomplete preservation snapshot | Preserves the causal-carry motor-recovery preregistration, implementation, job, and test that were not promoted into the current architecture. |
| `codex/private-wip-er-ordinal-20260727` | `1528b7f` | Rejected-route preservation snapshot | Extends the ordinal-route history with the final pre-scrub handoff. It is retained for reconsideration but remains downstream of the rejected ordinal-route result. |

## Folded repository-provenance histories

| Historical branch | Preserved tip | Classification | Purpose |
| --- | --- | --- | --- |
| `codex/main-integration` | `182738c` | Repository integration provenance | Records the earlier ETTR/public integration sequence. It introduces no unique canonical tree relative to its parents. |
| `codex/public-pre-sanitization-20260727` | `062cc0c` | Public-scrub provenance | Records the public repository state before sanitization. Its ancestry is retained privately, but its tree is not allowed to replace private ETTR `main`. |

## Preserved non-Git evidence

The obsolete ER worktree contained two unique local result directories. They
were moved intact into the canonical private evidence store before that
worktree was removed:

- `artifacts/r12/er_dual_stream_fresh_score_5499768532556522119`
  (`compiler.pt` SHA-256
  `01cbaa8c9de2c59ce75ff0eb95b6414e1cbb4c2636ca59fca03ee59d07ef3106`;
  assessment SHA-256
  `ddd607f326bf1b8cab6a5a67989ba5d5908a3323f4efba2b6c22d1184cf2e483`).
- `artifacts/r12/er_dual_stream_ordinal_1790361034717866861`
  (`compiler.pt` SHA-256
  `99be7b89e0b7dfe35f745abf1320c6640ad61f2fb62624b288fb8f9502cd97e7`;
  report SHA-256
  `6cc31e67734afa2320bc825d8bc368c780125fb7e6bb98409841e08950a2dece`).

Large tensors remain intentionally ignored by Git; the paths and hashes above
make their local custody explicit rather than leaving them hidden in an old
branch worktree.

## How to revisit an archived experiment

1. Read the relevant result/preregistration and its contemporaneous section in
   `AGENT_RUNBOOK.md` from the preserved tip.
2. Inspect the exact historical tree without changing the live workspace:
   `git show <tip>:<path>` or `git diff <tip>^ <tip>`.
3. State what new evidence invalidates the old rejection or what defect the new
   design repairs.
4. Create a fresh branch from private `main`; do not resurrect the old branch
   as the canonical line.
5. Preserve the old evaluator and add a preregistered comparison before using
   new compute.

This structure keeps every failed idea recoverable while preventing abandoned
implementations from cluttering or silently changing the current architecture.
