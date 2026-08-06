# DIVERGE-PL1: Branch-Local Verified Plasticity Result

Status: oracle-typed mechanics gate passed. This is not a raw-language,
Shohin, or continuation-pretraining result.

## Result

The source-separated board contains 2,048 train episodes, 256 development
episodes, and five unopened confirmation seeds of 256 episodes each. Every
episode uses a new permutation of eight opaque symbols over eight
noncommuting transforms in `Z/97Z`, twelve acquisition attempts with eight
complete branches each, and sixteen context-free transfer programs at depths
12--20. Alias and program overlap is zero across every split pair.

On 1,280 confirmation episodes and 20,480 transfer programs:

| Arm | Exact transfer | Exact mapping |
| --- | ---: | ---: |
| STATIC | `1/20,480` (`0.005%`) | `0/1,280` |
| CONTEXT_ONLY | `1/20,480` (`0.005%`) | `0/1,280` |
| DIVERGE_ONLY | `66/20,480` (`0.322%`) | `3/1,280` |
| FAST_WEIGHT | `5/20,480` (`0.024%`) | `0/1,280` |
| TRANSIENT_GRAD | `533/20,480` (`2.603%`) | `30/1,280` |
| **PL1** | **`17,726/20,480` (`86.553%`)** | **`1,104/1,280` (`86.250%`)** |

PL1 transfer rates across the five fixed seeds are `85.205%`, `85.107%`,
`88.818%`, `83.984%`, and `89.648%`. The weakest seed remains above the
frozen 80% floor. The paired 95% bootstrap lower bound against the strongest
baseline, TRANSIENT_GRAD, is `81.948` points.

The assessor-only context-free transfer probe rises from `7/20,480` after
attempt 1 to `17,726/20,480` after attempt 12. Text, demonstrations, branch
transcripts, and verifier messages are absent at transfer; only the 64-scalar
session policy matrix remains.

## Causal controls

- plastic reset: `1/20,480`;
- shuffled branch receipts: `485/20,480`;
- wrong-branch receipts: `3/20,480`, with 122,296 mismatched credits rejected;
- unrelated episode-state transplant: `3/20,480`;
- no eligibility localization: `70/20,480`;
- poison changed behavior in `1,280/1,280`, then rollback restored exact
  pre-poison hashes and outputs in `1,280/1,280`;
- protected-owner mutation injection failed closed in `5/5` seeds; and
- normal PL1 runs had zero rejected credits or protected-owner hash changes.

Removing the write cap and score clip is exactly tied with PL1 at
`17,726/20,480`. Homeostasis is therefore not part of the causal performance
claim; it remains only a safety envelope.

Every frozen condition passes. The decisive factor is not generic mutable
state: verified first-error receipts plus branch-local eligibility allow the
policy to accumulate correct partial assignments across failed complete
hypotheses. Whole-branch retention, scalar fast weights, and full-branch
transient gradients cannot localize that credit.

## Claim boundary and decision

PL1 v0 has zero learned parameters and uses `REFERENT_ORACLE`, an exact typed
executor, and a verifier that identifies the first invalid transition. It
proves a branch-local policy-update mechanism under deterministic feedback; it
does not prove that a language model can compile the symbols, produce the
certificates, or solve open-domain problems.

SRP1 failed, so raw-language PL1 remains blocked. Preserve this mechanism and
return to the semantic source/referent interface. Only after a structurally
different compiler qualifies may one natural PL1 integration run. Do not run
PL1 rank, width, branch-count, attempt-count, seed, budget, or duration
variants.

## Receipts

- implementation/preregistration commit: `33e6a7b`;
- data report SHA-256:
  `738b60f8c837fa479a9bb1cf82b11ff06441280a50f490f2d3155e82e24f3058`;
- evaluation SHA-256:
  `0cc2a24267c38396e166dbce7e540aaba87c5acf2474ede26be0939cfe9fd5fb`.

