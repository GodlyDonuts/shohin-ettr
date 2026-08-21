# Sai 4B Benchmark-First Contract

Status: prospective, no GPU execution authorized by this document.

## The result that ended always-revise Shohin

The completed five-benchmark Qwen9 comparison is a falsification, not a mixed
success. The revision arm scored an unweighted `42.806%`, versus `54.022%` for
the original model and `49.911%` for the equal-compute control. Its deltas were
`-11.216` and `-7.105` percentage points. HumanEval+, MBPP+, and IFEval gains do
not offset the `-33.201` point MuSR and `-20.839` point CorrectBench regressions
versus the original model. Mandatory second-pass revision is closed as a route
to general capability.

## Sai is a candidate, not a claimed architecture

Sai targets the best practical model near four billion parameters. The first
candidate parent is the exact Apache-2.0 Qwen3.5-4B post-trained host at
revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. This choice is provisional:
it is the strongest deployable 4B parent for which Shohin already has exact
training and memory evidence. Sai remains a single-pass model. It receives no
hidden draft and is never forced to rewrite an answer.

The first candidate is a conservative behavior-preserving skill update:

1. train only a narrow adapter on verified, benchmark-decontaminated math,
   code, science, logic, and instruction data;
2. mix broad parent-behavior replay into every optimizer window;
3. penalize divergence from frozen-parent logits on replay tokens;
4. compare against both the unchanged parent and an equal-token/equal-update
   ordinary-SFT control; and
5. discard the candidate if a real benchmark veto fires.

Training loss, synthetic mechanism gates, internal development boards, and
historical Shohin scores cannot authorize promotion.

## First real-benchmark gate

The gate uses the complete official HumanEval+, MBPP+, IFEval, MuSR, and
CorrectBench boards with identical prompt bytes and decoding for all three
checkpoints. Every report binds the benchmark version, ordered row identities,
prompt and decoding contracts, and all checkpoint hashes. The candidate must:

- beat both the original and equal-compute macro by at least `1.0` point;
- regress by no more than `1.0` point on every individual benchmark against
  either comparator;
- beat each comparator on at least four of five benchmarks; and
- be nonnegative against both comparators on MuSR and CorrectBench.

`pipeline/analyze_sai4b_public_gate.py` applies this conjunction. A failure
stops that candidate. A pass only authorizes broader confirmation; it does not
lock the architecture.

## Compute order

No GPU is requested until the immutable parent, data union, replay source,
equal-compute control, evaluator, and exact score identities exist. Then the
smallest useful order is mechanics, a low-token training pilot for candidate
and control, and independent one-H100 benchmark shards. The first five-board
result decides whether any larger training or benchmark campaign is allowed.

This reverses the old workflow: public capability evidence selects the
architecture; architecture enthusiasm never selects the evidence.
