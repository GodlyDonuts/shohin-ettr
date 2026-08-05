# CSDC Lexical Backbone Transfer Gate

**Status:** frozen fail-fast development gate; no CSDC language-integration
claim until this gate passes

## Capability hypothesis

The promoted CSDC mechanism already performs complete-hypothesis construction,
counterexample falsification, one-lineage commitment, and late execution at
`99.723%` under a held renderer. Its remaining boundary is lexical: the
successful compiler receives a small controlled vocabulary whose state and
operator tokens preserve identity.

The next question is whether a pretrained language residual can assign unseen
directional phrases to the correct copied operator role while preserving the
same binding and ordered-copy mechanism. This is tested before building a
larger natural-language CSDC front end.

## Paired board

One factorized semantic corpus is generated once. A derivation pass retokenizes
the exact same UTF-8 questions and character spans for the second tokenizer.
It must prove byte-bound source artifacts, exact semantic-payload identity,
zero cross-split prompt/name/13-gram overlap, exact dual-executor agreement,
and unchanged quartet labels. The model sees only token IDs and the source
length mask.

The development splits are:

- known-atom compositional transfer;
- lexical OOD, where every direction phrase is absent from training but is
  built from meaningful directional words;
- order and binding twins with matched token bags.

Confirmation is not evaluated by this gate.

## Matched arms

1. protected Shohin step-300k, frozen through layer 19;
2. exact imported `HuggingFaceTB/SmolLM2-135M-Instruct` backbone, frozen
   through layer 19;
3. the SmolLM2 arm with pointer and kind labels shuffled within equal-length
   buckets.

All arms use the existing ordinary token-role compiler: width 384, eight
heads, five source encoder layers, identical optimizer, one pass, example
order, batch size, seed, labels, and exact host executor. The backbones share
the same 576-wide, 30-layer geometry. Token counts and elapsed compute are
reported because the tokenizers differ.

## Fixed decision

The lexical substrate passes only if:

- both real-label arms reach at least 98% exact programs on known-atom
  compositional development;
- SmolLM2 reaches at least 90% exact lexical-OOD programs and 95% lexical-OOD
  answers;
- SmolLM2 exceeds matched Shohin lexical-OOD exact programs by at least 15
  percentage points;
- SmolLM2 reaches at least 410/512 all-four lexical-OOD exact groups; and
- shuffled SmolLM2 remains at or below 5% exact programs.

A pass authorizes wiring the same copy predictions into the frozen CSDC
candidate constructor and testing end-to-end lexical challenge falsification.
A miss means pretrained residuals alone do not ground unseen operators. The
next interface must provide explicit in-prompt definitions or contrastive
operator grounding; it does not authorize more epochs, width, seeds, or a
relaxed score on this board.

## Resource envelope

Corpus generation and paired retokenization request eight CPU cores, 64 GiB,
and at most two hours. The three independent neural arms each request one H100
for at most one hour and are expected to finish training plus both evaluations
in roughly 12--20 minutes. The maximum gate is therefore three H100-hours; the
expected use is below one H100-hour. No additional seed or duration follows a
miss.

## Claim boundary

This is a language-substrate and exact-program test. The host still
dereferences copied tokens and executes a frozen list machine. Passing does
not establish unrestricted natural-language reasoning, public-benchmark
ability, or a new backbone result.
