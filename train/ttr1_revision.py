"""Exact prompt and attention mechanics for transferable temporal revision."""

from __future__ import annotations

from typing import Any


DRAFT_MARKER = "Internal draft:\n"
FINAL_PROBLEM_MARKER = "\n\nOriginal problem:"
FORMAT_MARKER = "\n\nReturn "


class TTR1RevisionError(ValueError):
    """A temporal-revision prompt does not satisfy the frozen contract."""


def internal_revision_prompt(task_prompt: str, draft: str, task: str) -> str:
    """Build the frozen source-plus-draft revision prompt."""

    format_instruction = (
        "Return only executable Python code, without Markdown fences."
        if task == "mbpp"
        else "Return a complete corrected solution with the exact final answer in \\boxed{}."
    )
    return (
        "Solve the original problem by checking and revising the model's earlier draft. "
        "The draft may contain useful steps or errors; do not merely critique it.\n\n"
        f"Original problem:\n{task_prompt}\n\nInternal draft:\n{draft}\n\n"
        f"{format_instruction}\n\nOriginal problem:\n{task_prompt}"
    )


def internal_draft_char_span(rendered_prompt: str) -> tuple[int, int]:
    """Locate the sole informative draft span in a rendered chat prompt."""

    marker_start = rendered_prompt.find(DRAFT_MARKER)
    if marker_start < 0:
        raise TTR1RevisionError("rendered prompt has no internal-draft marker")
    draft_start = marker_start + len(DRAFT_MARKER)
    final_problem = rendered_prompt.rfind(FINAL_PROBLEM_MARKER)
    if final_problem <= draft_start:
        raise TTR1RevisionError("rendered prompt lacks the repeated final problem")
    draft_end = rendered_prompt.rfind(FORMAT_MARKER, draft_start, final_problem)
    if draft_end <= draft_start:
        raise TTR1RevisionError("rendered prompt lacks the final format instruction")
    return draft_start, draft_end


def tokenize_with_draft_mask(
    tokenizer: Any,
    rendered_prompt: str,
) -> tuple[list[int], list[int], tuple[int, int]]:
    """Tokenize once and hide only draft-overlapping keys at identical geometry."""

    draft_start, draft_end = internal_draft_char_span(rendered_prompt)
    encoded = tokenizer(
        rendered_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")
    if (
        not isinstance(input_ids, list)
        or not isinstance(offsets, list)
        or len(input_ids) != len(offsets)
        or not input_ids
    ):
        raise TTR1RevisionError("tokenizer lacks exact offset mappings")
    attention_mask: list[int] = []
    masked = 0
    for offset in offsets:
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            raise TTR1RevisionError("token offset geometry differs")
        token_start, token_end = map(int, offset)
        overlaps = token_end > draft_start and token_start < draft_end
        attention_mask.append(0 if overlaps else 1)
        masked += int(overlaps)
    if masked == 0 or sum(attention_mask) == 0:
        raise TTR1RevisionError("draft mask is empty or removes the full prompt")
    return [int(token) for token in input_ids], attention_mask, (draft_start, draft_end)
