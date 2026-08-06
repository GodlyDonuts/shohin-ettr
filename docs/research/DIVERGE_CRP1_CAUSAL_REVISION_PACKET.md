# DIVERGE-CRP1: Causal Revision Packet

Status: implementation and gate frozen before any neural result.

## Capability hypothesis

VCR1 proved that a small recurrent prefix can turn incomplete SmolLM3 drafts
into concise answers, but it did not revise a complete explicit wrong
derivation. CRP1 tests a materially different mechanism: preserve a bounded
version space over the location of the first invalid step, evaluate every
location with tied recurrent weights, and commit one whole guarded repair
packet before generating a correction and causally replayed suffix.

The packet has one categorical fault-line variable with values `NO_ERROR` and
`STEP_1 ... STEP_N`. Candidate `e` receives separate source masks for the
valid prefix before `e`, the proposed fault at `e`, and every dependent step
after `e`. All candidates share parameters and remain separate through
recurrence. The generator receives only the prefix from one hard-selected
candidate; candidate fields are never averaged.

The exact matched control instantiates the same parameters, candidate
identities, recurrent updates, training rows, targets, and FLOPs, but every
candidate sees the complete trace in all three channels. A gain over this
control is evidence for the causal guard structure rather than candidate
width or recurrent compute.

## Board

The deterministic board contains scalar arithmetic, noncommuting two-register
programs, and symbolic string transformations. Train/development depths are
4--6. Evaluation depths are 7--9 and also hold out the renderer and value/width
band. Every wrong trace is complete and contains exactly one verifier-created
first error. Every later draft step is recomputed from that corrupted state,
so the suffix is locally coherent but globally wrong. The correct answer is
never equal to the wrong draft answer. Every trace also has a correct no-op
twin.

Supervisor program objects and error certificates are stored for independent
assessment but are never rendered to the model beyond the ordinary problem,
complete draft, and final draft answer. The model must emit the first error,
one corrected step, the replayed dependent suffix, and one boxed answer.

## Frozen one-seed experiment

1. Build and hash 4,800 train, 480 development, and 480 OOD evaluation rows.
2. Run two-update guarded and unguarded mechanical smokes from byte-identical
   packet initialization.
3. On mechanical success, train each arm for exactly 200 updates, accumulation
   8, at learning rate `3e-4`. The SmolLM3-3B product generator remains frozen.
4. Evaluate the exact same 480 OOD identities under prompt-only, guarded,
   unguarded, reset, shifted-selection, and packet-swap conditions. Evaluate
   all three trained/plain arms on the 480 correct no-op twins.

## Frozen promotion gate

All conditions are conjunctive:

- guarded wrong-trace exact answers at least 240/480;
- guarded beats prompt-only by at least 48 solves;
- guarded beats equal-compute unguarded by at least 24 solves;
- guarded autonomous packet localization at least 360/480;
- guarded joint localized-and-correct revision at least 192/480;
- guarded correct-twin answer and no-error packet preservation each at least
  432/480, and no more than 12 solves below prompt-only preservation;
- guarded beats unguarded by at least five solves in each family;
- reset, shifted selection, and packet swap each cost at least 24 solves, and
  at least two cost at least 48;
- no arm exhausts the 384-token correction budget.

Failure closes this exact first-error packet without a seed, width, duration,
loss, depth-band, renderer, or threshold repair. A pass qualifies only a
bounded model-owned causal-revision mechanism. It then requires transfer to
natural verified math/code/science traces before any broader DIVERGE claim.
