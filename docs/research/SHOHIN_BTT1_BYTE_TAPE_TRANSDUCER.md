# BTT1: Raw Byte-Tape Transduction Compiler

Status: frozen successor contract; CPU extensional audit pending

Date: 2026-08-10

Predecessor: MLTC1 closed despite 100% normal because pre-extracted candidate
surface/position metadata retained 85.19% under contextual-state permutation

Holdout: sealed

## Hypothesis

MLTC1 proves that monotonic lexical selection plus generic precedence
execution can eliminate PSTC1's source-position drift, but its candidate
extractor owns too much syntax. BTT1 removes every pre-extracted number,
operator, parenthesis, and surface-type input. A standalone bidirectional byte
transducer receives only the complete raw ASCII question and absolute tape
position. It labels every byte as:

`IGNORE, NUM_BEGIN, NUM_CONT, NEGATE, ADD, SUB, MUL, DIV, LPAREN, RPAREN`.

A generic executor copies bytes from each predicted numeric span and applies
shunting-yard precedence to the predicted operator/parenthesis labels. It
cannot inspect an answer, invoke a verifier, infer symbols from raw bytes, or
repair malformed output. A complete program is therefore impossible unless
the model locates expression ownership, number boundaries, unary signs,
operators, and scope on the raw source tape.

## Frozen model and budget

- byte IDs `0..255`, one padding ID, and learned absolute positions;
- width 256, six bidirectional Transformer blocks, eight heads, FFN width
  1024, no pretrained backbone and fewer than 10M trainable parameters;
- maximum tape length admitted by CPU audit, hard-capped at 512;
- per-byte cross entropy with `IGNORE` weight `0.1` and every selected role
  weight `1.0`;
- 1,024 updates, batch 64, AdamW LR `3e-4`, betas `(0.9,0.95)`, weight decay
  `0.01`, gradient clip 1, one seed;
- exactly 65,536 charged examples; no recurrence, decoding search, answer
  labels, arithmetic values, or verifier feedback.

## Controls

1. same-family/depth full-source shuffle under unchanged weights;
2. all active input byte IDs zeroed while tape length/positions and original
   executor copy source remain fixed;
3. identical predicted byte roles with precedence and parentheses removed by
   a flat left-to-right executor;
4. frozen MLTC1/PSTC1/FSTC1 references.

## Development gate

- complete byte-role sequence exact `>=99%`;
- selected byte-role sequence exact `>=99%`;
- generic-executor valid program `>=99.5%`;
- exact materialized operation skeleton `>=97%`;
- every family `>=95%`;
- mixed precedence, unary groups, and three-plus-parenthesis each `>=90%`;
- source shuffle and zero-byte controls each `<=25%` exact with aligned margin
  `>=70` points;
- flat execution loses `>=35` points on hierarchical rows;
- zero byte truncation, malformed numeric copies, invalid programs, or
  fallback.

One development pass opens exactly one sealed holdout. Failure closes BTT1
without width, depth, duration, seed, LR, role vocabulary, loss, or threshold
variants. A pass opens arithmetic-state learning over the sealed compiled
program but does not by itself establish arithmetic reasoning.

## CPU admission

Job `749685` admitted all `75,935` training and all `3,917` development rows
with exact byte-role-to-program parity and no exclusions. The maximum raw tape
length is `174` training bytes / `172` development bytes, below the frozen
512-byte cap. Train/development SHA-256 are
`8deabd1ad1cdd93466cf9a53bde3012886fd63ff327914897bd70cbd9e13bc6e`
and `de16d5dc8dd3676f4c3f4a69c306ec19a0ea8a42e9750bf93a2aaf16f4be0bf8`;
report SHA-256 is
`a69c5dd401fb91bf6da057cd38c4c1834c834de79da9f51594fbf27bbf80993d`.
