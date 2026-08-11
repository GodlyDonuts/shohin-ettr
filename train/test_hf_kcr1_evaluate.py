import hashlib

from hf_kcr1_evaluate import summarize


def test_treatment_mechanics_summary_passes_exact_transactions() -> None:
    rows = []
    results = []
    actions = ("<KEEP>", "<CONTINUE>", "<RESTART>")
    for source in range(20):
        source_identity = hashlib.sha256(f"source-{source}".encode()).hexdigest()
        for action in actions:
            identity = hashlib.sha256(f"{source}-{action}".encode()).hexdigest()
            rows.append(
                {
                    "identity_sha256": identity,
                    "source_identity_sha256": source_identity,
                    "expected_action": action,
                }
            )
            results.append(
                {
                    "identity_sha256": identity,
                    "action_correct": True,
                    "correct": True,
                    "execution_exact": True,
                    "valid_transaction": True,
                    "keep_byte_preserved": action == "<KEEP>",
                    "max_token_exhausted": False,
                }
            )
    summary = summarize(rows, results)
    assert summary["treatment_mechanics_pass"] is True
    assert summary["counterfactual_consistency"] == 1.0
    assert summary["keep_byte_preservation_accuracy"] == 1.0
    assert summary["control_gate_pending"] is True


def test_counterfactual_source_failure_is_visible() -> None:
    actions = ("<KEEP>", "<CONTINUE>", "<RESTART>")
    rows = []
    results = []
    for source in range(10):
        source_identity = str(source)
        for action in actions:
            identity = f"{source}-{action}"
            correct = not (source == 0 and action == "<CONTINUE>")
            rows.append(
                {
                    "identity_sha256": identity,
                    "source_identity_sha256": source_identity,
                    "expected_action": action,
                }
            )
            results.append(
                {
                    "identity_sha256": identity,
                    "action_correct": correct,
                    "correct": correct,
                    "execution_exact": correct,
                    "valid_transaction": correct,
                    "keep_byte_preserved": action == "<KEEP>" and correct,
                    "max_token_exhausted": False,
                }
            )
    summary = summarize(rows, results)
    assert summary["counterfactual_consistency"] == 0.9
    assert summary["branch_accuracy"]["<CONTINUE>"] == 0.9
