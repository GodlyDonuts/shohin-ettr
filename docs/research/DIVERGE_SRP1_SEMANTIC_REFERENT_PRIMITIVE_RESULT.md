# DIVERGE-SRP1: Semantic Referent Primitive Result

Status: closed negative after the one frozen training and evaluation pass. No
SRP1 width, duration, seed, renderer, optimizer, warm-start, or loss variant is
authorized.

## Result

Newton job `744234` completed exactly 1,000 joint REFERENT updates and fit both
immutable training sets at `50,000/50,000`. Evaluation job `744246` completed
normally on `evc32` in `00:16:04`.

The fresh confirmation result is:

- WORLD programs: `256/256`;
- natural EVIDENCE receipts: `3,067/3,072`;
- fully sealed episodes: `253/256`;
- natural QUERY transactions: `753/768`;
- sensitive answers: `248/256`;
- exact extensional answers: `253/256`;
- no-evidence abstention: `251/256`;
- invariant answers: `248/256`;
- partial-evidence underdetermined abstention: `249/256`;
- protected TOL3: `1,024/1,024`;
- protected NVE1: `256/256`; and
- invalid queries accepted: `0`.

Query transactions by renderer are `128/128`, `127/128`, `128/128`,
`114/128`, `128/128`, and `128/128`. Renderer 3 therefore misses the frozen
`122/128` floor.

## Matched owner comparison

Frozen SOT1 is much stronger on this deconfounded board than on its original
confirmation board:

| Owner | Exact query transactions |
| --- | ---: |
| Frozen SOT1 | `742/768` |
| SRP1 | `753/768` |
| Delta | `+11/768` (`+1.43` points) |

SRP1 needed at least `+77/768` over SOT1. It misses that causal inclusion gate
by 66 transactions. The semantic primitive improves the weak renderer from
`102/128` to `114/128`, but the gain is too small and does not meet the
renderer floor.

## Decision

SRP1 fails the conjunctive gate. It also misses full evidence sealing,
extensional parity, universal packet/query swap rejection, and post-seal
poison invariance because three episodes never seal. The shared REFERENT owner
is not a qualified raw-language interface.

Consequences:

1. close SRP1 without local retries;
2. do not launch raw-language branch-local plasticity through SRP1;
3. preserve TOL3 and NVE1 as protected stage-specific owners; and
4. allow DIVERGE-PL1 only as an explicitly oracle-typed plasticity-mechanics
   ceiling while referent/source compilation remains the active bottleneck.

## Receipts

- checkpoint SHA-256:
  `0e4365d96d86ea791b64e8bc60b46e4441acfc3d2bc043bd9339da3e4bb286cd`;
- model-state SHA-256:
  `4b54f333394c869a4ac7a8286300176167d7d4e1ddae1cb86e1afcf5a3112182`;
- fresh board SHA-256:
  `8d5bf36e5eab0d51328a1fbfeabad676efcbe697fdbe1dd64cbe592820aa0216`;
- evaluation SHA-256:
  `8b68d19bbf67007addc6c9e9ec5287414580b4cd9b108198cb199fd171257ebe`;
- immutable runtime archive SHA-256:
  `502802d0fca0f33ebc005503cfd7867e7cfc2a67a0ddc68a2e1cbe8c559eaa6c`.

