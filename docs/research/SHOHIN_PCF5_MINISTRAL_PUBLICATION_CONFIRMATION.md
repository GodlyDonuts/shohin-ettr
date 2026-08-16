# PCF5: Ministral Publication Confirmation Successor

Status: prospectively frozen on 2026-08-11 before PCF5 admission or compute.

PCF1 through PCF4 remain closed with formal scientific result `null`. PCF4's
CPU-only reference canary passed all 40 sandbox probes, then stopped before
executing any reference because standalone setup qualification attempted to
execute a frozen setup that constructs `Node` instances before the candidate
has defined `Node`. It did not publish data, submit a scientific graph, load a
model, use an H100, open an assessor, or emit a scientific score. The exact
evidence is in
`SHOHIN_PCF4_TERMINAL_INFRASTRUCTURE_RECEIPT_20260811.json`.

The user subsequently directed the campaign to continue fixing pre-science
infrastructure in pursuit of Shohin. PCF5 is the separately named,
prospectively declared successor. It inherits every scientific and security
clause of PCF1 through PCF4. The host, source bytes and hashes, split, prompts,
arms, seeds, training geometry, generated-candidate policy SHA-256
`f27124db3d134a1e3dbde06958ab03220cd5e9585abcc356baa6a49d9edd1f1e`,
trusted-reference sandbox mode, thresholds, single assessor open, and terminal
gate are unchanged.

Its only repair is the setup-qualification lifecycle. Exact hash-pinned
supervisor setup and official-test sources are compiled inside the qualified
Bubblewrap runtime before candidate execution. Successful compilation is
receipted as `compile_only_before_candidate`; setup source is never evaluated
during that preflight. During an actual reference or candidate assessment, the
already-compiled setup executes in the established order after the candidate,
so candidate-defined classes and functions are available. A compile/runtime
fault before candidate execution is infrastructure failure. A failure caused
when setup executes after candidate code remains a scientific candidate
failure. Generated candidates retain the unchanged capability policy and
cannot select this trusted setup mode.

Run the full infrastructure-only nonsealed-reference canary once. It must
produce a fresh 40-probe sandbox receipt, compile every unique setup, execute
every nonsealed frozen MBPP reference in trusted-reference mode, and record
zero holdout-reference payload access. If it passes, preserve its immutable
receipt, repackage the exact pushed runtime, replay storage, repository, model,
source, environment, and live admission, and submit one PCF5 scientific graph.
Disable requeue, retries, and automatic successors. Any infrastructure
failure, formal `PASS`, or formal `FAIL` ends PCF5. Do not change or open an
alternate host, data split, prompt, arm, seed, threshold, or protected board.
