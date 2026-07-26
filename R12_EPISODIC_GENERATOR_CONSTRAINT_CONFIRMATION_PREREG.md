# R12 Episodic-Generator Constraint Confirmation Preregistration

**Post-result disposition:** formal promotion rejected. In addition to the
frozen zero-seal miss, adversarial audit established that Shohin was absent,
the inference path was neuro-symbolic, and 14/22 development target-word
instances overlapped training.

## Frozen finding

Mechanics canary `704786`, seed `20260725`, was launched before fail-closed
consensus sealing was added. It reached 11/11 exact development queries,
including 2/2 held-out random-permutation episodes, but deletion of one
necessary target witness still accidentally sealed 4/11 packets. The canary
is evidence for induction capability, not the final safety claim.

## Single confirmation

Run the same frozen board and budget at seed `20260726` after adding one
mechanical requirement: every surviving episode-local program must agree on
every state transition before a target map may seal. An ambiguous posterior
must emit a non-permutation and fail closed.

The confirmation passes only if:

- treatment, record-order reversal, and support-order recoding are all exact;
- every family and development cell is exact;
- target-law training/development overlap is zero;
- source and support-generator material are absent from every deployed packet;
- zeroed observations, deranged support semantics, and deleted target
  witnesses seal zero packets;
- shifted observations recover zero complete target maps; and
- the complete conceptual system remains below 200M parameters.

No additional architecture branch follows this run. The result will either
establish bounded episode-local finite-program induction or close the route.
