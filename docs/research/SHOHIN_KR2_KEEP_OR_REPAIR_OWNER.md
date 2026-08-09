# KR2: Keep-or-Repair Stage Owner

Status: frozen before predecessor generation or training.

## Capability hypothesis

RIDR1 showed that a second application of the one-shot revision owner repairs
15 errors but breaks 65 correct answers. KR2 trains a distinct later role on
the actual output distribution of IDR1. For verified-correct predecessor
revisions, the target is one discrete action, `<KEEP_PREVIOUS>`. For incorrect
predecessors, the target is the same verifier-backed full replacement used by
IDR1. At inference the model either preserves one complete predecessor
trajectory or emits one complete replacement; fields are never averaged.

## Matched experiment

Generate one greedy IDR1 predecessor for each of 5,824 unique train identities,
then expand them back over the unchanged 9,655 IDR1 train presentations.
Initialize both arms from the exact qualified IDR1 checkpoint and train each
for 256 updates with the same model, source, predecessor, targets, optimizer,
batch geometry, sequence length, LoRA geometry, seed, and charged data.

- **Treatment:** correct predecessors target `<KEEP_PREVIOUS>`; errors target a
  full verified repair. Exact sentinel output selects the predecessor.
- **Direct-rewrite control:** every row targets the full verified answer and
  sentinel output has no special meaning.

Evaluation uses the existing 1,289-row recurrent development corpus, greedy
768-token decoding, and the unchanged exact assessor. Holdout stays sealed.

## Frozen gate

All conditions are conjunctive:

- treatment at least `615/1,289`;
- treatment at least 26 answers above direct rewrite;
- math at least 223, logic/science at least 349, and code at least 17;
- at least 98% retention of the 589 depth-one-correct answers;
- keep-action precision at least 95%; and
- at least 64 keep actions, preventing a vacuous always-rewrite policy.

A pass unlocks one holdout run. Failure closes KR2 without prompt, rank,
duration, seed, or threshold rescue. The claim is bounded to a model-owned
stage-specific keep/repair policy, not generic recurrence or novelty.
