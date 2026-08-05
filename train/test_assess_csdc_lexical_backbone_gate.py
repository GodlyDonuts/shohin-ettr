from assess_csdc_lexical_backbone_gate import assess


def result(program: float, answer: float, all_four: int) -> dict[str, object]:
    return {
        "overall": {
            "semantic_program_exact": program,
            "answer_accuracy": answer,
        },
        "group_summary": {"all_four_semantic_program_exact": all_four},
    }


def test_assessment_pass_and_kill_boundaries() -> None:
    shohin_comp = result(0.99, 0.99, 500)
    shohin_lex = result(0.70, 0.80, 200)
    smol_comp = result(0.99, 0.99, 500)
    smol_lex = result(0.91, 0.96, 420)
    shuffled = result(0.01, 0.34, 0)
    passed = assess(
        shohin_comp, shohin_lex, smol_comp, smol_lex, shuffled, shuffled,
    )
    assert passed["all_gates_pass"] is True
    failed = assess(
        shohin_comp,
        shohin_lex,
        smol_comp,
        result(0.84, 0.96, 420),
        shuffled,
        shuffled,
    )
    assert failed["all_gates_pass"] is False
    assert failed["decision"].startswith("close_")
