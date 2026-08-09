"""Fail-closed whole-trajectory commitment for selective temporal revision."""

from __future__ import annotations

from dataclasses import dataclass


KEEP_COMMAND = "<KEEP>"
REVISE_COMMAND = "<REVISE>"


@dataclass(frozen=True)
class CommitResult:
    """One coherent committed trajectory and its model-owned decision."""

    command: str
    completion: str
    valid: bool


def selective_commit(draft: str, controller_output: str) -> CommitResult:
    """Commit the entire draft or entire revision; malformed output fails closed."""

    if controller_output == KEEP_COMMAND:
        return CommitResult(command="keep", completion=draft, valid=True)
    prefix = f"{REVISE_COMMAND}\n"
    if controller_output.startswith(prefix):
        revision = controller_output[len(prefix) :]
        if revision.strip():
            return CommitResult(command="revise", completion=revision, valid=True)
    return CommitResult(command="malformed", completion="", valid=False)


def selective_commit_prompt(task_prompt: str, draft: str, task: str) -> str:
    """Request one discrete commit decision followed by an optional replacement."""

    serialized_draft = draft if draft.strip() else "<EMPTY_DRAFT>"
    replacement_format = (
        "The replacement after <REVISE> must be only executable Python code, "
        "without Markdown fences."
        if task == "mbpp"
        else "The replacement after <REVISE> must be a complete corrected solution "
        "with the exact final answer in \\boxed{}."
    )
    return (
        "Check the model's earlier draft against the original problem. If the draft "
        "is already correct and complete, emit exactly <KEEP> and nothing else. If "
        "it is wrong or incomplete, emit <REVISE> on the first line followed by one "
        "complete replacement solution. Do not critique the draft.\n\n"
        f"Original problem:\n{task_prompt}\n\nInternal draft:\n{serialized_draft}\n\n"
        f"Return exactly one commit command. {replacement_format}\n\n"
        f"Original problem:\n{task_prompt}"
    )
