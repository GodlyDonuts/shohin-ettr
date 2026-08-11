# PCF4: Ministral Publication Confirmation Successor

Status: prospectively frozen on 2026-08-11 before PCF4 admission or compute.

PCF1, PCF2, and PCF3 remain closed with formal scientific result `null`.
PCF3's infrastructure-only reference canary stopped before opening its source
because the historical directory name `product_reasoning` triggered the
post-freeze lexical firewall. It did not start the sandbox, publish data,
submit a scientific graph, load a model, use an H100, or open an assessor.

PCF4 inherits every scientific and security clause of PCF1, PCF2, and PCF3.
The host, source bytes/hashes, split, prompts, arms, seeds, training geometry,
generated-candidate policy SHA-256 `f27124db...f1e`, trusted-reference sandbox
mode, thresholds, single assessor open, and terminal gate are unchanged.

Its only repair is an exact control-plane exception already used by the main
preparation job: the reference canary may read the one historical source file
only when its complete SHA-256 equals
`0b6d068b4d71f407cb234579b9278dc640df09139ea906dd0f52a6ab71e05398`.
No other `product`, `public`, or `holdout` path is admitted. The path is never
exported to a GPU, model-visible artifact, prompt, candidate, or score.

Run the full infrastructure-only reference canary once. It must produce a
fresh 40-probe sandbox receipt and pass every nonsealed frozen MBPP reference
with zero holdout-reference access. If it passes, repackage the exact pushed
runtime, replay storage/repository/model/source/live preflight, and submit one
PCF4 graph. Disable requeue, retries, and automatic successors. Infrastructure
failure, formal `PASS`, or formal `FAIL` ends PCF4. Do not open holdout,
product, public, an alternate host, or a successor after PCF4.
