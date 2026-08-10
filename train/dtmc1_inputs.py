"""Source-custodied source-plus-draft inputs for DTMC1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from hf_product_reasoning_train import PRODUCT_SYSTEM_PROMPT, render_reasoning_messages
from natural_microcode_program import parse_program
from typed_microcode_compiler import MAX_SOURCE_SPANS
from typed_microcode_graph import TypedMicrocodeGraph, compile_typed_graph

SCHEMA = "shohin-dtmc1-model-owned-draft-corpus-v1"
ENVELOPE_PREFIX = "PROBLEM:\n"
ENVELOPE_MIDDLE = "\n\nMODEL-OWNED DRAFT:\n"


class DTMC1InputError(ValueError):
    """DTMC1 input corpus or segment custody differs."""


@dataclass(frozen=True, slots=True)
class DraftExample:
    identity_sha256: str
    graph: TypedMicrocodeGraph
    draft: str
    draft_correct: bool
    exhausted: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def example_from_row(row: dict[str, Any]) -> DraftExample:
    return DraftExample(
        identity_sha256=str(row["identity_sha256"]),
        graph=compile_typed_graph(
            str(row["original_question"]), parse_program(str(row["gold_program"]))
        ),
        draft=str(row["draft"]),
        draft_correct=bool(row.get("draft_correct", False)),
        exhausted=bool(row.get("exhausted", False)),
    )


def load_examples(
    path: Path, expected_sha256: str, expected_rows: int
) -> list[DraftExample]:
    if sha256_file(path) != expected_sha256:
        raise DTMC1InputError("draft corpus SHA-256 differs")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if (
        len(rows) != expected_rows
        or any(row.get("schema") != SCHEMA for row in rows)
        or len({row["identity_sha256"] for row in rows}) != expected_rows
    ):
        raise DTMC1InputError("draft corpus population differs")
    return [example_from_row(row) for row in rows]


def render_draft_source(tokenizer: Any, source: str, draft: str) -> tuple[str, int]:
    content = ENVELOPE_PREFIX + source + ENVELOPE_MIDDLE + draft
    rendered = render_reasoning_messages(
        tokenizer,
        [
            {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        enable_thinking=False,
    )
    content_start = rendered.find(content)
    if content_start < 0 or rendered.find(content, content_start + 1) >= 0:
        raise DTMC1InputError("input envelope is absent or ambiguous")
    source_start = content_start + len(ENVELOPE_PREFIX)
    if rendered[source_start : source_start + len(source)] != source:
        raise DTMC1InputError("source segment boundary differs")
    return rendered, source_start


def tokenize_draft_sources(
    tokenizer: Any,
    examples: Sequence[DraftExample],
    device: torch.device,
    maximum_tokens: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, int]]:
    rendered_and_starts = [
        render_draft_source(tokenizer, example.graph.source, example.draft)
        for example in examples
    ]
    rendered = [item[0] for item in rendered_and_starts]
    source_starts = [item[1] for item in rendered_and_starts]
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        padding=True,
        truncation=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")
    lengths = encoded["attention_mask"].sum(1)
    if int(encoded["input_ids"].shape[1]) > maximum_tokens:
        raise DTMC1InputError("source-plus-draft prompt exceeds frozen context")
    candidate_mask = torch.zeros(
        len(examples),
        MAX_SOURCE_SPANS,
        encoded["input_ids"].shape[1],
        dtype=torch.bool,
    )
    for row, (example, source_start) in enumerate(
        zip(examples, source_starts, strict=True)
    ):
        source_end = source_start + len(example.graph.source)
        for candidate, span in enumerate(example.graph.number_spans):
            absolute_start = source_start + span.start
            absolute_end = source_start + span.end
            for token, (token_start, token_end) in enumerate(offsets[row].tolist()):
                if (
                    token_end > absolute_start
                    and token_start < absolute_end
                    and token_start < source_end
                    and token_end > source_start
                ):
                    candidate_mask[row, candidate, token] = True
            if not candidate_mask[row, candidate].any():
                raise DTMC1InputError("source span lacks a token owner")
        for candidate in range(len(example.graph.number_spans), MAX_SOURCE_SPANS):
            if candidate_mask[row, candidate].any():
                raise DTMC1InputError("draft token entered source candidate mask")
    receipt = {
        "maximum_tokens": int(lengths.max()),
        "charged_source_draft_tokens": int(lengths.sum()),
    }
    return (
        {key: value.to(device) for key, value in encoded.items()},
        candidate_mask.to(device),
        receipt,
    )
