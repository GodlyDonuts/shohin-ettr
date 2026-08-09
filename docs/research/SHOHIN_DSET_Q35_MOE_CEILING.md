# DSET-Q35: Strong-MoE Script Capacity Ceiling

Status: frozen read-only development ceiling, 2026-08-09. No Qwen3.6 output
exists at freeze time. No training or holdout is authorized by this contract.

## Question

DSET1 on OLMoE is already causal and strong on numeric edits (`98.05%`) but
weak on choice edits (`71.48%`). PSET1 shows that replacing the host LM with a
small isolated value decoder destroys corrected-value synthesis. DSET-Q35 asks
whether a current stronger MoE can follow the exact same model-owned edit
protocol without training.

## Frozen implementation

- host: local pinned Qwen3.6-35B-A3B MoE artifact;
- data: exactly the 256 already-opened PSET1 diagnostic identities, mapped
  back to their immutable DSET1 clean/fault presentations;
- prompt, script grammar, deterministic executor, maximum 32 generated tokens,
  greedy decoding, and exact complete-trajectory scoring unchanged from DSET1;
- no demonstrations, prompt variants, verifier, repair, router, tool, answer
  label, adapter, or benchmark-specific selection;
- eight disjoint H100 shards, merged before inspection.

The ceiling passes only if exact execution is `>=95%` overall, exact scripts
are `>=90%` in both numeric and choice families, clean copy is `>=99%`, fault
repair is `>=90%`, and there are zero execution errors or exhausted rows. A
miss closes this exact untrained transfer prompt. A pass authorizes a separate
prospective trained DSET transfer contract; it is not itself a reasoning claim.
