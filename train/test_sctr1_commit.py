from __future__ import annotations

from sctr1_commit import selective_commit, selective_commit_prompt


def test_keep_commits_original_draft_byte_for_byte() -> None:
    draft = "def f():\n    return 7\n"
    result = selective_commit(draft, "<KEEP>")
    assert result.valid and result.command == "keep"
    assert result.completion == draft


def test_revise_commits_one_complete_replacement() -> None:
    result = selective_commit("wrong", "<REVISE>\ncorrect")
    assert result.valid and result.command == "revise"
    assert result.completion == "correct"


def test_malformed_or_empty_revision_fails_closed() -> None:
    assert not selective_commit("draft", "KEEP").valid
    assert not selective_commit("draft", "<REVISE>\n").valid
    assert not selective_commit("draft", "<KEEP>\nextra").valid


def test_prompt_preserves_source_and_explicit_code_contract() -> None:
    prompt = selective_commit_prompt("write f", "def f(): pass", "mbpp")
    assert prompt.count("Original problem:\nwrite f") == 2
    assert prompt.count("Internal draft:") == 1
    assert "<KEEP>" in prompt and "<REVISE>" in prompt
    assert "only executable Python code" in prompt
