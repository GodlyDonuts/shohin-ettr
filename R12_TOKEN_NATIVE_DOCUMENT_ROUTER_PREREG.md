# R12 Token-Native Document Router Preregistration

## Status

Preregistered after matched lexical-command H100 job `725519` failed and
before optimization of a syntax-routed arm.

## Failure Localized

Each ETTR COMMAND is a token-native AST followed by deterministic cover up to
an exact 96-token transport width. The source attention mask currently marks
all 96 positions as valid because the cover itself is model-visible. Shohin's
causal residual at AST positions cannot depend on later cover, but every
replacement compiler cross-attends the entire residual and lexical transport.
For a representative local-rewrite command, only 13/96 positions are the AST.
The rest are public-codebook cover, not padding and not command semantics.

## Architecture Treatment

`TokenNativeDocumentMask` is a zero-parameter, tensor-native syntax router. It
maps token IDs to their public codebook indices and recovers the end of the
leading AST from call arities, fused reification arities, renderer preambles,
and the admitted ETTR root heads. It then intersects that exact document span
with the existing transport mask. The compiler receives the unchanged frozen
layer-19 residual plus unchanged raw lexical embeddings, but cross-attention
cannot see transport cover.

The router does not use a symbol sidecar, family or ontology label, QUERY,
answer, target, terminal packet, transaction trace, oracle program, candidate
score, host semantic executor, or verifier. It recognizes only the public
transport grammar already required to tokenize the source. The same tensor
path runs during training, interface evaluation, and fully autonomous model
execution. Its codebook inverse is checkpointed as a non-trainable buffer.

## Mechanical Gates

Before launch:

1. all four renderers must recover the exact same document length on packed
   prefix and postfix examples;
2. both legacy head-14 and local head-15 ETTR roots must route;
3. non-codebook and malformed sources must fail closed;
4. changing only masked cover hidden states and embeddings must leave compiler
   output unchanged;
5. the full exact source-deleted evaluator must receive routed tokens through
   the production `execute` path;
6. parameter count remains below 200M and the protected checkpoint remains
   immutable.

## Matched H100 Gate

Use architecture seed 31, data seed 11, oracle initial WORLD state, H100 BF16,
1,000 updates, LR `3e-4`, clip `1.0`, causal-delta weight `1.0`, atomic-action
weight `1.0`, and 32 held-out batches. The protected checkpoint, ETTR release,
query compiler/reader, random initialization, and all train/evaluation rows
match job `725519`. The only treatment is exact COMMAND-document masking.

The router is retained only if it materially exceeds job `725519`'s post-fit
oracle value accuracy of `70.196507%` without structural regression. Restoring
value accuracy above the untouched oracle-initial `74.344978%` is the useful
minimum; nonzero exact terminal packets are the stronger interface gate.
Fully autonomous reasoning still requires simultaneous nonzero strict WORLD
and COMMAND gates on the unchanged evaluator and cannot be claimed from this
oracle-initial isolation alone.

If routed attention fails, generic pooling is closed. The successor must
predict explicit syntax-node occurrences and bind selected codeword identity
to typed state operations, rather than add another dense loss or wider latent.

Decision:
`delete_public_transport_cover_then_keep_only_measured_binding_gain_or_move_to_explicit_syntax_occurrence_pointers`.
