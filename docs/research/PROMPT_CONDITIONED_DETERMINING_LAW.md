# Prompt-Conditioned Determining Law

Status: one bounded representation falsifier; no novelty or reasoning claim.

## Capability hypothesis

CEER can lower source-consequence energy but its learned consequence head fails
when applied to the held-out query. The latent system fits observed evidence
without forming one reusable law. Prompt-Conditioned Determining Law (PCDL)
replaces the opaque state/readout split with an explicit episode law.

A shared learned basis maps every source probe and the late query into the same
rank-R coordinates. Source outcomes determine class-valued law coefficients by
a differentiable regularized solve:

```text
Phi_j = phi_theta(source, probe_j)
W = (Phi^T M Phi + lambda I)^-1 Phi^T M one_hot(outcome)
answer_logits = phi_theta(source, query)^T W
```

The query is absent from law formation. It can only evaluate the law determined
by source witnesses. The treatment cannot use a separate query reader.

## Matched control

The DENSE arm is a standard outcome-embedded set-attention reader from all
source witnesses to the query. Every model instantiates and executes both PCDL
and DENSE paths; the arm changes only which answer logits receive the answer
loss and are evaluated. Encoder, data, parameter count, module execution,
training updates, and frozen cohorts are identical. Both arms receive the same
PCDL witness-reconstruction auxiliary loss, so the shared encoder has the same
extra supervision.

This mechanism is related to deep kernel learning, conditional neural
processes, differentiable least squares, and system identification. Those are
prior art. This pilot tests the representation diagnosis; it does not claim
that a differentiable law solve is itself novel.

## Frozen pilot

- seed 41;
- 1,000 updates, 256 examples/update;
- training depths 2--4;
- rank 8, width 64, ridge 0.1;
- the same six frozen depth-5/7 cohorts used by FCPT, CGSGR, QVESR, and CEER;
- one H100 per arm, estimated under five minutes total GPU time.

PCDL advances only if it beats DENSE by at least five absolute macro points,
improves the mean of both depths in every family, exceeds 15% mean induction
accuracy, loses at least five points when witness outcomes are shuffled, and
loses at least five points when solved law coefficients are exchanged across
episodes. A miss closes this exact learned-basis law solver; it does not permit
rank, ridge, seed, width, duration, or basis-network sweeps.

## Result

Jobs `739305`/`739306` complete cleanly in 27/24 seconds. Both arms contain
exactly 51,038 parameters and see 256,000 examples.

| Family | Depth | PCDL | DENSE | Delta |
|---|---:|---:|---:|---:|
| Noncommuting | 5 | 0.000% | 22.754% | -22.754 |
| Noncommuting | 7 | 0.000% | 52.344% | -52.344 |
| Binding | 5 | 11.621% | 20.605% | -8.984 |
| Binding | 7 | 6.738% | 15.625% | -8.887 |
| Induction | 5 | 9.375% | 10.254% | -0.879 |
| Induction | 7 | 8.008% | 9.082% | -1.074 |
| **Macro** | | **5.957%** | **21.777%** | **-15.820** |

The learned law reconstructs approximately 85.5--95.2% of observed binding
and induction witnesses and 87.6--94.4% of noncommuting witnesses, while
scoring zero on unseen noncommuting queries. Its controls invert the intended
causal prediction: shuffling witness outcomes raises macro to 10.254%, and
exchanging solved law coefficients across episodes raises it to 10.677%.

Report SHA-256 values are:

- PCDL: `fc1d4810a09f07af7f7e87fb14009bd76faa53dbbb0d0fb959c43c7f9933a894`
- DENSE: `6aa1442f4d83cb4987408032d688f3c80388b40e2d4098eaddf6b42ddd563021`

PCDL is therefore closed. A low-rank learned feature map plus differentiable
coefficient solve is not a determining representation: it creates a compact
interpolation surface with no enforced algebra between probes. The next
candidate must select a restricted hypothesis family and compose a learned
generator or operation law inside that family. This is a different mechanism,
not a rank, ridge, seed, width, or duration repair.
