import pytest

from build_diverge_crp1_board import CRP1BoardError, FAMILIES, generate_episode


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("heldout,depth", [(False, 4), (False, 6), (True, 9)])
def test_generated_episode_has_one_complete_causal_error(
    family: str, heldout: bool, depth: int
) -> None:
    row = generate_episode(7301 + depth, family, depth, heldout)
    assert len(row["correct_steps"]) == depth
    assert len(row["wrong_steps"]) == depth
    assert 1 <= row["error_index"] <= depth - 2
    assert row["answer"] != row["wrong_answer"]
    assert row["correct_steps"][: row["error_index"] - 1] == row["wrong_steps"][: row["error_index"] - 1]
    assert row["correct_steps"][row["error_index"] - 1 :] != row["wrong_steps"][row["error_index"] - 1 :]
    assert row["wrong_target"].startswith(f"Error step: {row['error_index']}\n")
    assert row["correct_target"].startswith("Error step: NONE\n")
    assert row["candidate_count"] == depth + 1


def test_generation_is_deterministic_and_split_sensitive() -> None:
    left = generate_episode(1234, "register", 6, False)
    right = generate_episode(1234, "register", 6, False)
    shifted = generate_episode(1234, "register", 6, True)
    assert left == right
    assert left["identity_sha256"] != shifted["identity_sha256"]


def test_bad_episode_contract_fails_closed() -> None:
    with pytest.raises(CRP1BoardError):
        generate_episode(1, "missing", 6, False)
    with pytest.raises(CRP1BoardError):
        generate_episode(1, "scalar", 3, False)
