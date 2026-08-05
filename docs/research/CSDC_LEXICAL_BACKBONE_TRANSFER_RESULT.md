# CSDC Lexical Backbone Transfer Result

**Decision:** pass; advance model-owned SmolLM2 role-copy predictions into the
frozen CSDC candidate constructor

## Result

The capability-floor hypothesis passes every frozen condition. On exactly the
same semantic questions and labels:

| Arm | Known program | Lexical-OOD program | Lexical-OOD answer | Lexical all-four |
|---|---:|---:|---:|---:|
| protected Shohin step-300k | 100.000% | 77.344% | 85.352% | 221/512 |
| exact SmolLM2-135M-Instruct | 100.000% | **95.947%** | **96.582%** | **443/512** |
| shuffled-label SmolLM2 | 0.342% | 0.293% | 0.391% | 0/512 |

SmolLM2 improves exact unseen-lexical programs by **18.604 points** over
Shohin, above the fixed 15-point margin. Its 95.947% program score, 96.582%
answer score, 443 exact quartets, and 0.293% shuffled control all clear the
remaining gates.

## What changed

The UTF-8 questions, character spans, semantic programs, answers, split
identities, and matched twins are identical. Semantic identity SHA-256 is
`3ec090657cd0ee90d509d7313147a156ec65b617b13b9c7edb4f32fd47d18cab`.
Only the frozen backbone and its tokenizer representation differ. Both real
arms use the same 8,607,886-parameter ordinary role-copy compiler at layer 19,
the same 96,000 examples once, seed, optimizer, labels, and exact executor.

Shohin tokenizes the 96,000 training questions into 9,624,734 tokens;
SmolLM2 uses 9,493,338. Length-bucket packing produces 1,517 versus 1,515
optimizer updates. This 0.13% schedule difference is disclosed rather than
described as exact update matching. The result isolates the combined practical
effect of pretrained backbone plus tokenizer, not backbone weights alone.

## Jobs and cost

| Job | Purpose | State | Elapsed |
|---:|---|---|---:|
| `739553` | paired corpus build and retokenization | completed | 3:58 CPU |
| `739556` | real imported-parent load probe | completed | 0:08 CPU |
| `739590` | Shohin role-copy arm | completed | 9:36 H100 |
| `739591` | SmolLM2 role-copy arm | completed | 9:41 H100 |
| `739592` | shuffled SmolLM2 arm | completed | 9:49 H100 |
| `739615` | automatic assessment | completed | <1s CPU |

Total neural allocation is 29:06, or approximately **0.485 H100-hours**.

## Artifact receipts

- runtime archive SHA-256:
  `3946ab3c7bef4303bc876f291f28ad9e99221c3b1f8cd9ddf0703db866c2470f`;
- Shohin/Smol corpus report SHA-256:
  `f102629f2e6c5cec1770ec2ca585b948d1f2052bf8279bd5f1eb194c3261c6a6` /
  `59b886fd2268cff25b488b5dca98aac01450e4b2532b2a142da6b27886d78098`;
- assessment SHA-256:
  `6ac70d3d0a1938366342411fab380182ea3c3c9ab5ab6269a9971232ccfa3fa7`;
- Shohin/Smol/shuffled compiler SHA-256:
  `1439b5499f612cef1cc4ce52eeea27ca931fdfbd27d6c9e3f7bd03856b470567` /
  `abd22528da0d8dc4718c7a89d9c94520540a2f38b7f0b1d9a9e623d0af23cf4d` /
  `16a7b12b5ec499b6c83912b18aa9918197612abc5e23f836e8f10b0544c32c40`.

All Newton outputs are read-only. The compact assessment is preserved in the
private repository.

## Interpretation and next gate

This is a meaningful semantic-interface result: the copied source positions
are already exact, while the stronger pretrained representation supplies the
previously missing unseen-word polarity. It is not yet CSDC natural-language
reasoning, because this board ends in a host list-machine executor.

The authorized successor freezes the trained Smol role-copy compiler and the
existing CSDC reasoning core. Model-decoded source roles must construct the
complete CSDC hypotheses, source challenges must falsify them, and one lineage
must answer the late query. The end-to-end gate must preserve the shuffled-
challenge and lineage-swap failures. No confirmation or public claim follows
from the compiler result alone.
