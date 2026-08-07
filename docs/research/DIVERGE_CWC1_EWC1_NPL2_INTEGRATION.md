# DIVERGE CWC1 -> EWC1 -> NPL2 Integration

Status: frozen before integrated WORLD or late-query scoring.

## Capability hypothesis

The confirmed CWC1 whole-world selector can remove EWC1's semantic shortcut
by choosing one complete candidate before structural extraction. Frozen EWC1
then acts only as a transcription layer inside that selected candidate. If
the selected typed WORLD is exact, unchanged confirmed NPL2 should retain its
late-query reasoning score.

No component is retrained. This is one bounded composition test, not an EWC1
retry and not authorization for continuation pretraining.

## Frozen path

```text
two complete WORLD candidates + natural directive
  -> confirmed CWC1 involution selector
  -> one physical complete candidate
  -> frozen EWC1 structural extractor
  -> typed WORLD; source deleted
  -> unchanged NVE1/EIC1/NPL2/executor/verifier
  -> late QUERY
```

CWC1 checkpoint `cae4d896...ab1a` and confirmation result
`f42802ce...5a9d` are immutable. EWC1 checkpoint `0816ed1c...d1ed` is used
only under its documented boundary: normal structure was `4096/4096`, but it
failed semantic source scrub and is not a semantic owner. The confirmed NPL2,
EIC1, NVE1, STI1, executor, verifier, base checkpoint, and tokenizer remain
unchanged and hash-checked.

## Wrapper board

Every existing NPL2 WORLD program is paired with a deterministic decoy using
the same aliases, registers, depth, and surface grammar but different initial
state and operation sequence. The true candidate position is exactly balanced
within each of all 64 positive/negative CWC renderer pairs. Candidate labels,
sources, and identities are unique. The original program and all downstream
assessor data are unchanged.

Development wraps the fixed 256-episode NPL1/NPL2 development split, yielding
7,168 candidate decisions. Five conditional confirmation wrappers use the
five already-frozen NPL2 confirmation seeds. All wrapper JSONL files and input
hashes must be materialized and audited before the first integrated score.

## Conjunctive development gate

- CWC normal and mapped-counterfactual selection at least 99%;
- mapped counterfactual selects the original decoy at least 99%;
- directive scrub between 49% and 51%, maximum absolute margin at most
  `1e-6`, exact projection;
- selected EWC typed WORLD at least 99% joint exact;
- forced-opposite typed WORLD at most 1% exact against the true WORLD while
  EWC transcribes the decoy itself at least 99%;
- NPL2 late-query exactness at least 80%, within five points of PL1 oracle,
  and at least ten points above every non-oracle arm;
- unchanged NPL2 reset, shuffled-credit, wrong-branch, transplant,
  eligibility, rollback, source-deletion, and protected-owner gates;
- all source, data, checkpoint, runtime, and result hashes exact.

Only a conjunctive pass opens the five fixed confirmation jobs. A failure
closes this exact composition without seed, width, threshold, renderer,
duration, parser, or loss variants.

## Claim boundary

A confirmation pass would establish controlled end-to-end composition of a
learned natural directive selector, learned structural transcription, and
confirmed source-deleted NPL2 reasoning. Candidate generation, the
mini-language, execution, and verification remain engineered. It would not
establish unrestricted language reasoning, an involution score advantage, or
permission for a long pretraining run.
