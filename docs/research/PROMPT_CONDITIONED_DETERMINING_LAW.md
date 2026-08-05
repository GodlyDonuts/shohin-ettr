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
