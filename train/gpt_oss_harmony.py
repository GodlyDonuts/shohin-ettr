"""Frozen GPT-OSS Harmony prompt, training, and final-channel projection."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

REASONING_EFFORT = "low"
START_ASSISTANT = "<|start|>assistant"
FINAL_MARKER = "<|channel|>final<|message|>"
ANALYSIS_MARKER = "<|channel|>analysis<|message|>"
RETURN_MARKER = "<|return|>"
END_MARKER = "<|end|>"


class GptOssHarmonyError(RuntimeError):
    """A GPT-OSS prompt or generated Harmony trajectory differed."""


def _encode(tokenizer: Any, text: str) -> list[int]:
    values = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise GptOssHarmonyError("GPT-OSS tokenizer output differs")
    try:
        result = [int(value) for value in values]
    except (TypeError, ValueError) as error:
        raise GptOssHarmonyError("GPT-OSS token ids differ") from error
    if not result or any(value < 0 for value in result):
        raise GptOssHarmonyError("GPT-OSS token ids are empty or negative")
    return result


def render_prompt(tokenizer: Any, question: str) -> str:
    """Render one matched user problem to the exact low-effort generation prefix."""

    if not isinstance(question, str) or not question.strip():
        raise GptOssHarmonyError("GPT-OSS question differs")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True,
        tokenize=False,
        reasoning_effort=REASONING_EFFORT,
    )
    if (
        not isinstance(rendered, str)
        or not rendered.endswith(START_ASSISTANT)
        or FINAL_MARKER in rendered[rendered.rfind("<|start|>user") :]
    ):
        raise GptOssHarmonyError("GPT-OSS generation prompt differs")
    return rendered


def tokenize_training_example(
    tokenizer: Any,
    question: str,
    response: str,
    *,
    max_sequence_length: int,
) -> tuple[list[int], list[int]]:
    """Return exact prompt and supervised final-channel token ids."""

    if (
        not isinstance(response, str)
        or not response.strip()
        or not isinstance(max_sequence_length, int)
        or isinstance(max_sequence_length, bool)
        or max_sequence_length < 1
    ):
        raise GptOssHarmonyError("GPT-OSS training response differs")
    prompt_text = render_prompt(tokenizer, question)
    full_text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
        ],
        add_generation_prompt=False,
        tokenize=False,
        reasoning_effort=REASONING_EFFORT,
    )
    if (
        not isinstance(full_text, str)
        or not full_text.startswith(prompt_text)
        or not full_text[len(prompt_text) :].startswith(FINAL_MARKER)
        or not full_text.endswith(RETURN_MARKER)
    ):
        raise GptOssHarmonyError("GPT-OSS training template differs")
    prompt_ids = _encode(tokenizer, prompt_text)
    full_ids = _encode(tokenizer, full_text)
    if (
        len(full_ids) <= len(prompt_ids)
        or full_ids[: len(prompt_ids)] != prompt_ids
        or len(full_ids) > max_sequence_length
    ):
        raise GptOssHarmonyError("GPT-OSS training token geometry differs")
    return prompt_ids, full_ids[len(prompt_ids) :]


def extract_final_completion(
    tokenizer: Any, generated_ids: Sequence[int]
) -> tuple[str, dict[str, Any]]:
    """Project one generated Harmony trajectory to its final-channel content."""

    if isinstance(generated_ids, (str, bytes)):
        raise GptOssHarmonyError("GPT-OSS generated ids differ")
    try:
        ids = [int(value) for value in generated_ids]
    except (TypeError, ValueError) as error:
        raise GptOssHarmonyError("GPT-OSS generated ids differ") from error
    if any(value < 0 for value in ids):
        raise GptOssHarmonyError("GPT-OSS generated ids are negative")
    raw = tokenizer.decode(
        ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(raw, str):
        raise GptOssHarmonyError("GPT-OSS decoded trajectory differs")
    marker = raw.rfind(FINAL_MARKER)
    final_found = marker >= 0
    body = raw[marker + len(FINAL_MARKER) :] if final_found else ""
    terminators = [
        index
        for marker_text in (RETURN_MARKER, END_MARKER, START_ASSISTANT)
        if (index := body.find(marker_text)) >= 0
    ]
    terminated = bool(terminators)
    if terminators:
        body = body[: min(terminators)]
    completion = body.strip() if final_found else ""
    return completion, {
        "raw_trajectory_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "analysis_channel_present": ANALYSIS_MARKER in raw,
        "final_channel_present": final_found,
        "final_channel_terminated": terminated,
        "empty_final_completion": not completion,
    }
