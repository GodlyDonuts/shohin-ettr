# R12 Sparse-Law Neural Microcode Result

## Decision

The supervised neural-microcode candidate **fails** the frozen sparse-law
promotion gate. Do not scale this formulation or describe it as native
reasoning.

Newton job `704760` completed 2,000 optimizer updates on one H100 in 3 minutes
10 seconds. The learned byte controller predicts a program for a fixed
finite-domain ALU; inference uses no host parser, search, solver, oracle, or
posthoc verifier. Training and development action maps are hash-disjoint.

## Results

| Arm | Transition accuracy | Complete maps | Exact queries | Invalid seals |
|---|---:|---:|---:|---:|
| Microcode treatment | 22.1250% | 0/60 | 0/60 | 52/60 |
| Direction negated | 20.8750% | 0/60 | 1/60 | 53/60 |
| Observation targets shifted | 5.2083% | 0/60 | 0/60 | 56/60 |
| Observations zeroed | 7.7917% | 0/60 | 0/60 | 60/60 |

The frozen direct-attention baseline was 46.5000% transition accuracy,
0/60 complete maps, and 4/60 exact queries. Microcode therefore loses the
baseline by 24.375 percentage points and also loses the direction-negated
same-weight control by one exact query.

Training reached 57.5263% transition accuracy, 456 complete maps, and
674/3,300 exact queries. Development program accuracy was only
1/204 = 0.4902%. The result is not an optimization-free no-signal outcome:
source direction reached 100%, loss fell from 5.742733 to 2.394297, and
shifting or removing observed transitions caused a large degradation.
Instead, the controller learned local evidence dependence without learning a
program-identification rule that transfers to unseen action maps.

## Receipts

- learned compiler parameters: 340,152
- conceptual complete system: 125,421,816 parameters
- global limit: 200,000,000 parameters
- training rows: 3,300
- development rows: 60
- training action laws: 263
- development action laws: 80
- overlap: 0
- candidate-time oracle/search/verifier calls: 0/0/0
- report SHA-256:
  `3f6f78d84c1ce46dc8975f609ef80a79a86ecc6e9d430446c40b309d6753a94e`
- model SHA-256:
  `a38ae69806e1dd394cb8b271fb541ed34a3f7bbbfcec290cf2b4b273ca7a6334`

## Scientific Conclusion

An internal ALU solves execution only after the correct instruction is
identified. It does not by itself create law induction. Direct table
completion, learned generic generators, and supervised fixed-ontology
microcode all fail on the same hash-disjoint sparse-law boundary.

The strongest demonstrated result remains the 60,613-parameter semantic
partition compiler at 360/360 on complete anonymous machines. Shohin itself
still does not demonstrate native general reasoning. Under the current usage
constraint, further proxy-specific architecture branching is not justified.
The protected 300k checkpoint and explicit pretraining hold remain unchanged.
