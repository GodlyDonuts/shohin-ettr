"""Exact rendering and token admission for causal-revision packets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from hf_product_reasoning_train import PRODUCT_SYSTEM_PROMPT, render_reasoning_messages


PROBLEM_OPEN = "<revision_problem>"
PROBLEM_CLOSE = "</revision_problem>"
TRACE_OPEN = "<complete_draft>"
TRACE_CLOSE = "</complete_draft>"
FINAL_OPEN = "<draft_final>"
FINAL_CLOSE = "</draft_final>"


class CRP1DataError(RuntimeError):
    """A causal-revision row cannot satisfy the exact token contract."""


@dataclass(frozen=True)
class RevisionTokens:
    prompt_ids: list[int]
    response_ids: list[int]
    problem_mask: list[bool]
    step_masks: list[list[bool]]
    final_mask: list[bool]


def _step_open(index: int) -> str:
    return f"<draft_step_{index:02d}>"


def _step_close(index: int) -> str:
    return f"</draft_step_{index:02d}>"


def revision_user_content(
    problem: str,
    trace_steps: Sequence[str],
    draft_final: str,
) -> str:
    problem = problem.strip()
    steps = [step.strip() for step in trace_steps]
    draft_final = draft_final.strip()
    if not problem or not steps or not draft_final or any(not step for step in steps):
        raise CRP1DataError("revision problem, steps, and final must be nonempty")
    reserved = (
        PROBLEM_OPEN,
        PROBLEM_CLOSE,
        TRACE_OPEN,
        TRACE_CLOSE,
        FINAL_OPEN,
        FINAL_CLOSE,
        "<draft_step_",
        "</draft_step_",
    )
    if any(marker in problem or marker in draft_final for marker in reserved) or any(
        marker in step for marker in reserved for step in steps
    ):
        raise CRP1DataError("source text contains a reserved revision marker")
    rendered_steps = "\n".join(
        f"{_step_open(index)}\n{step}\n{_step_close(index)}"
        for index, step in enumerate(steps, start=1)
    )
    return (
        f"{PROBLEM_OPEN}\n{problem}\n{PROBLEM_CLOSE}\n\n"
        f"{TRACE_OPEN}\n{rendered_steps}\n"
        f"{FINAL_OPEN}\n{draft_final}\n{FINAL_CLOSE}\n{TRACE_CLOSE}\n\n"
        "Audit the complete draft. Locate the first invalid step, repair that "
        "step, and replay every dependent later step. If every step is valid, "
        "leave the result unchanged. Return an Error step line, a Correction "
        "line, any replayed Step lines, and exactly one Final answer line."
    )


def render_revision_prompt(
    tokenizer: Any,
    problem: str,
    trace_steps: Sequence[str],
    draft_final: str,
) -> str:
    return render_reasoning_messages(
        tokenizer,
        [
            {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": revision_user_content(problem, trace_steps, draft_final),
            },
        ],
        enable_thinking=False,
    )


def _content_span(rendered: str, opening: str, closing: str) -> tuple[int, int]:
    if rendered.count(opening) != 1 or rendered.count(closing) != 1:
        raise CRP1DataError("rendered revision markers are not unique")
    start = rendered.index(opening) + len(opening)
    if start < len(rendered) and rendered[start] == "\n":
        start += 1
    end = rendered.index(closing, start)
    while end > start and rendered[end - 1] == "\n":
        end -= 1
    if end <= start:
        raise CRP1DataError("rendered revision span is empty")
    return start, end


def _overlaps(offset: tuple[int, int], span: tuple[int, int]) -> bool:
    start, end = offset
    return end > span[0] and start < span[1]


def tokenize_revision_example(
    tokenizer: Any,
    problem: str,
    trace_steps: Sequence[str],
    draft_final: str,
    target: str | None,
    *,
    max_sequence_length: int,
    workspace_slots: int,
) -> RevisionTokens | None:
    if max_sequence_length <= workspace_slots + 8:
        raise CRP1DataError("revision sequence budget is too small")
    rendered = render_revision_prompt(tokenizer, problem, trace_steps, draft_final)
    problem_span = _content_span(rendered, PROBLEM_OPEN, PROBLEM_CLOSE)
    step_spans = [
        _content_span(rendered, _step_open(index), _step_close(index))
        for index in range(1, len(trace_steps) + 1)
    ]
    final_span = _content_span(rendered, FINAL_OPEN, FINAL_CLOSE)
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    prompt_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [tuple(map(int, value)) for value in encoded["offset_mapping"]]
    if len(prompt_ids) != len(offsets) or not prompt_ids:
        raise CRP1DataError("tokenizer offsets differ from revision tokens")
    problem_mask = [_overlaps(offset, problem_span) for offset in offsets]
    step_masks = [
        [_overlaps(offset, span) for offset in offsets] for span in step_spans
    ]
    final_mask = [_overlaps(offset, final_span) for offset in offsets]
    if not any(problem_mask) or not any(final_mask) or any(
        not any(mask) for mask in step_masks
    ):
        raise CRP1DataError("tokenizer lost a revision segment")
    all_masks = [problem_mask, *step_masks, final_mask]
    if any(
        sum(bool(mask[position]) for mask in all_masks) > 1
        for position in range(len(prompt_ids))
    ):
        raise CRP1DataError("tokenizer revision segments overlap")

    response_ids: list[int] = []
    if target is not None:
        target = target.strip()
        if not target:
            raise CRP1DataError("revision target is empty")
        response_ids = [
            int(value) for value in tokenizer.encode(target, add_special_tokens=False)
        ]
        if tokenizer.eos_token_id is None:
            raise CRP1DataError("tokenizer exposes no EOS token")
        response_ids.append(int(tokenizer.eos_token_id))
    total = len(prompt_ids) + workspace_slots + len(response_ids)
    if total > max_sequence_length:
        return None
    return RevisionTokens(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        problem_mask=problem_mask,
        step_masks=step_masks,
        final_mask=final_mask,
    )


__all__ = [
    "CRP1DataError",
    "RevisionTokens",
    "render_revision_prompt",
    "revision_user_content",
    "tokenize_revision_example",
]
