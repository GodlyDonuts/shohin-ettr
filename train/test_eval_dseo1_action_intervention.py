from eval_dseo1_action_intervention import normalized_trajectory


def test_normalized_trajectory_ignores_spacing_and_case() -> None:
    assert normalized_trajectory("Answer  IS\n3") == normalized_trajectory("answer is 3")
    assert normalized_trajectory("answer is 3") != normalized_trajectory("answer is 4")
