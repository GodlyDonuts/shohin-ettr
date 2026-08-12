# PCF15: Ministral Publication Confirmation Successor

Status: prospectively frozen on 2026-08-12 before admission or scientific
compute.

PCF14 is terminal-null. It completed preparation, mechanics, B1, all 16 draft
shards, the exact draft merge, materialization, and revision training. During
revision calibration, shard 2 stopped on MBPP identity `8d17a4...eb7f7`
because pinned Python failed before trusted READY. All peer allocations and
descendants were cancelled. The prepared confirmation assessor board exists
as required but has zero semantic reads; no confirmation candidate, score
authorization, score, final comparison, or formal scientific result exists.
Preserve `SHOHIN_PCF14_TERMINAL_INFRASTRUCTURE_RECEIPT_20260812.json` and its
clarifying amendment. Do not replay PCF14.

PCF15 is a separately named infrastructure successor under the user's standing
authorization. Its only change removes Python `preexec_fn` from the
per-candidate launcher. Python explicitly documents that callback as unsafe in
the presence of threads because the child can deadlock before `exec`; PCF14's
failure occurred at exactly that boundary in a resident PyTorch/CUDA process.
PCF15 instead executes exact host binary `/usr/bin/prlimit` (SHA-256
`2c1c7948...41d5`, util-linux 2.32.1), which applies the unchanged CPU `3:4`
seconds, address-space `1 GiB:1 GiB`, and file-size `1 MiB:1 MiB` limits and
then execs the unchanged Bubblewrap command. No Python code runs between fork
and exec.

The bootstrap SHA-256 remains `726af7a5...a60`, the generated-candidate policy
remains `f27124db...f1e`, and the three-second scientific candidate limit is
unchanged. Fresh Newton job `752701` passed all 41 sandbox probes. H100 job
`752702` loaded the exact dense host and PCF14 revision adapter on the same
node, then completed 2,000 consecutive executions of the exact failed MBPP
assessor context with zero infrastructure failures and no model generation or
confirmation access. Bind admission to
`SHOHIN_PCF15_SANDBOX_LAUNCH_QUALIFICATION_20260812.json`.

PCF15 must start from a fresh root and replay the complete graph. It may not
reuse any PCF14 draft, adapter, calibration, or partial output. The pinned
dense host and tokenizer, source banks, split and identity order, prompts,
unchanged/revision/self-refinement/commit arms, seeds, 256-update training
geometry, generation settings, candidate policy, sandbox filesystem and
network isolation, three-second candidate limit, custody rules, sealed board,
and sole 1,289-row conjunctive gate remain byte-for-byte unchanged. Requeue,
retries, partial continuation, automatic successors, and protected-split
access are disabled.

The sole falsifiable gate remains:

- unchanged reaches at least `387/1289` with every domain nonzero;
- trained revision exceeds unchanged by at least `65` correct and
  self-refinement by at least `39`, with no per-domain loss against either;
- learned whole-trajectory commit exceeds revision by at least `13`, retains
  at least 95% of both the revision-correct and unchanged-correct identity
  sets, and loses no domain against revision;
- all four arms cover the exact 1,289 identities in frozen order, with zero
  truncation, zero malformed selections, exact runtime/model/data/checkpoint/
  scheduler custody, one consumed assessor authorization, and zero
  holdout/public/product access.

Any infrastructure failure, formal `PASS`, or formal `FAIL` ends PCF15. A
formal result must be preserved exactly and authorizes no successor.
