# Shohin Reasoning Phase Handoff

Status: phase closed on 2026-08-11 after the one authorized VTE1 evaluation.
No successor experiment is frozen, queued, or authorized by this handoff.

## Surviving qualified architecture

The surviving practical architecture is the immutable dense 9B same-family
temporal-revision release:

```text
source -> model-owned draft -> trained same-family revision
       -> learned whole-trajectory commit -> one coherent answer
```

On the protected 538-row product board, unchanged continuation solves 316,
trained revision 374, and learned commitment 383; the coherent oracle ceiling
is 399. The package pins Qwen3.5-9B revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, release manifest SHA-256
`554e841f71edd3a19063411348340e337532db2db05dd5e1e2adc25a3d347e7b`,
release `SHA256SUMS` SHA-256
`0dad031312dec0859e35bb7e9daea8aef688ef350b9053f587fba5acdc9c58c5`,
and product-result SHA-256
`3e86751bb234ee29465885206da5316890060ad8b0b88ea752c4fb012bbf7187`.
This is a qualified multi-pass dense system, not a claim about the original
125M checkpoint or universal reasoning.

LAM1 remains the strongest controlled source-to-terminal mechanism: a learned
byte compiler, grammar-constrained program selection, and learned recurrent
digit microcode solve 3,917/3,917 source-disjoint development programs. It is
not holdout-qualified and retains explicit grammar/stack/rational scaffolding.

## Final bounded result

VTE1 trained a Qwen3.5-9B final-four rank-8 reviser with a set-valued loss over
independently verified KEEP/CONTINUE/RESTART transactions. Its immutable
source-disjoint result is:

- semantic correctness: `1285/1566 = 82.0562%`, versus KCR1 parent
  `1294/1566 = 82.6309%` (`-9` answers);
- per presentation: natural `379/522`, continue `414/522`, keep `492/522`;
- all three states correct: `324/522`;
- valid transactions: `1566/1566`, malformed: zero;
- emitted actions: RESTART `1566`, KEEP `0`, CONTINUE `0`;
- exact execution: `69/1566`; KEEP byte preservation: `0/692`;
- decode-limit exhaustions: `197`, generated tokens: `359,269`.

VTE1 is a conjunctive failure. Equivalence supervision removed arbitrary
label penalties but induced universal regeneration and did not improve
semantics. Draft-hidden control, broad development, and holdout were not
opened.

## Closed lanes and prohibited retries

The exact NDR1, KCR1, and VTE1 natural-draft/transaction family is closed.
Do not retry source mix, parser, prompt, delimiter, action ontology,
equivalence temperature, candidate family, rank, layer, duration, seed,
decoding, or threshold variants. DTMC1, DTC1, CTE1, LTR1, CTF1, and ECTR0
close the current natural-language-to-microcode bridge. DSEO1, DSET1, PSET1,
and later edit variants close their exact objectives/interfaces. Small-OLMoE
MTR/RCR/ECR/MPR/OBR/DPR routes do not authorize large-MoE scaling. Historical
synthetic ETTR lanes remain evidence, not qualified public capability.

## Exact VTE1 custody

- KCR1 parent checkpoint:
  `07e08abe2480782afc77e35031d23bea71a737d019f307066af2bde786dd2ebd`
- VTE1 training data/report:
  `cc312363f880e9048622b57cb0cb609acaf92a1ab9ed0552ec8383ea20da1c33` /
  `d1c828f4645a15175535d7712805b489019948bf11e0ab2c3899881c4c2705fb`
- mechanics checkpoint/report:
  `e74e929cb17ffc8825e538996aba9b06b36ce5dee633d4384e3c9b65e308b370` /
  `b91f218f0e9dd4f75522cf9c705cad3aa5f1cbf1b23ea3c1c0891b61996d12e0`
- fit checkpoint/report:
  `8d8f813341aa8eeab3b78cd010908e80fac75ec8d8ad690a18818deb5543381a` /
  `efc78413c319160cdbd942c69a07b65243504a1340772bdbdfa0273a9c03b846`
- canary data/report:
  `b9c97c28438b6a87e89c816c177f1a23818564bb16fed0c9551dd07c05d7e139` /
  `78fef3045c75a8aa1c61afb172b6e1474f2a4b2888c7e8c2cec1f5fd4b8638a5`
- merged candidates/report:
  `45efcefded3d6a06633c478bf6b265ebd8fd2a44d313cf9db780ca33374d03d3` /
  `e47a5f0e8cfcbdad920e02fab1aeabda49e5e12b8ac4103a278b524132070189`

Full paths, shard hashes, compute receipts, and frozen gate booleans are in
`docs/research/SHOHIN_VTE1_RESULT.json`.

## Unresolved bottleneck

The system can generate and consume useful complete trajectories on capable
dense hosts, and it can execute typed arithmetic programs causally. It still
lacks a reliable model-owned transaction planner that preserves correct
drafts, repairs semantic faults, and commits efficiently. KCR forced an
arbitrary canonical branch; VTE quotiented that ambiguity but learned the
easiest objective solution, universal RESTART. The remaining issue is not
transaction syntax or verifier quality. It is semantic planning and
capability preservation under a causally draft-dependent action.

## Compute state and one recommendation

VTE1 evaluation used 10,652.98 aggregate GPU-seconds (2.9592 H100-hours); its
256-update fit used 1,017 Slurm seconds (0.2825 H100-hours). No holdout or
successor compute was consumed. A final `squeue -u sa305415` check returned no
queued or running jobs; the VTE1 data, mechanics, fit, dispatcher, four
evaluation shards, and merge all report `COMPLETED` in Slurm accounting.

The single recommended next publication gate, if a later phase is authorized,
is an unopened cross-family confirmation of the already-qualified immutable
9B temporal-revision release contract on a capable dense host, with the same
model-owned draft, trained revision, unchanged-pass control, whole-trajectory
commit accounting, and a sealed source-disjoint benchmark. This tests the
surviving transferable claim without reopening the failed transaction family.
