"""Shared tokenization and frozen-host execution for PSET1."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


DATA_SCHEMA = "shohin-pset1-pointer-pair-v1"
DATA_REPORT_SCHEMA = "shohin-pset1-pointer-data-report-v1"
BYTE_EOS = 256


class PSET1RuntimeError(RuntimeError):
    """PSET1 runtime inputs differ from the frozen contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(data: Path, report_path: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = json.loads(report_path.read_text())
    expected = report.get("outputs", {}).get(split, {})
    if (
        report.get("schema") != DATA_REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("holdout_used") is not False
        or report.get("train_diagnostic_source_overlap") != 0
        or Path(str(expected.get("path", ""))).resolve() != data.resolve()
        or expected.get("sha256") != sha256_file(data)
    ):
        raise PSET1RuntimeError("PSET1 data report differs")
    rows = [json.loads(line) for line in data.read_text().splitlines() if line]
    if len(rows) != int(expected.get("sources", -1)) or any(row.get("schema") != DATA_SCHEMA for row in rows):
        raise PSET1RuntimeError("PSET1 row geometry differs")
    return rows, report


def encode_text(tokenizer: Any, text: str) -> tuple[list[int], list[list[int]]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [list(map(int, value)) for value in encoded["offset_mapping"]]
    if not ids or len(ids) != len(offsets):
        raise PSET1RuntimeError("PSET1 tokenization differs")
    return ids, offsets


def character_map(text: str, offsets: list[list[int]]) -> tuple[list[int], list[int]]:
    mapping = [-1] * len(text)
    for token_index, (left, right) in enumerate(offsets):
        for character_index in range(left, right):
            if not 0 <= character_index < len(mapping) or mapping[character_index] != -1:
                raise PSET1RuntimeError("PSET1 character offset overlap")
            mapping[character_index] = token_index
    if not mapping or any(value < 0 for value in mapping):
        raise PSET1RuntimeError("PSET1 character offset gap")
    return mapping, [ord(character) % 256 for character in text]


def tokenize_rows(tokenizer: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        source_ids = [int(value) for value in tokenizer.encode(row["source"], add_special_tokens=False)]
        members = {}
        for name in ("clean", "fault"):
            member = row["members"][name]
            draft_ids, offsets = encode_text(tokenizer, member["draft"])
            mapping, characters = character_map(member["draft"], offsets)
            members[name] = {
                **member,
                "draft_ids": draft_ids,
                "offsets": offsets,
                "character_to_token": mapping,
                "character_ids": characters,
            }
        output.append({**row, "source_ids": source_ids, "members": members})
    return output


def pad_ids(rows: list[list[int]], pad: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(map(len, rows))
    ids = torch.full((len(rows), maximum), pad, device=device, dtype=torch.long)
    mask = torch.zeros((len(rows), maximum), device=device, dtype=torch.bool)
    for index, row in enumerate(rows):
        ids[index, : len(row)] = torch.tensor(row, device=device)
        mask[index, : len(row)] = True
    return ids, mask


def pad_characters(members: list[dict[str, Any]], device: torch.device):
    maximum = max(len(member["character_ids"]) for member in members)
    mapping = torch.zeros((len(members), maximum), device=device, dtype=torch.long)
    ids = torch.zeros((len(members), maximum), device=device, dtype=torch.long)
    mask = torch.zeros((len(members), maximum), device=device, dtype=torch.bool)
    for index, member in enumerate(members):
        length = len(member["character_ids"])
        mapping[index, :length] = torch.tensor(member["character_to_token"], device=device)
        ids[index, :length] = torch.tensor(member["character_ids"], device=device)
        mask[index, :length] = True
    return mapping, ids, mask


@torch.no_grad()
def host_hidden(model: Any, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        embeddings = model.text_model.embed_tokens(ids)
        output = model.text_model(inputs_embeds=embeddings, attention_mask=mask, use_cache=False)
    return output.last_hidden_state.detach()


def replacement_batch(members: list[dict[str, Any]], arm: str, device: torch.device):
    action_key = "permuted_action" if arm == "permuted" else "action"
    replacement_key = "permuted_replacement_byte_ids" if arm == "permuted" else "replacement_byte_ids"
    actions = torch.tensor(
        [0 if member[action_key] == "KEEP" else 1 for member in members], device=device
    )
    pointers = torch.tensor(
        [[member["pointer_start"], member["pointer_end"]] for member in members], device=device
    )
    maximum = max(len(member[replacement_key]) for member in members) + 1
    inputs = torch.full((len(members), maximum), BYTE_EOS, device=device, dtype=torch.long)
    labels = torch.full((len(members), maximum), -100, device=device, dtype=torch.long)
    for index, member in enumerate(members):
        if actions[index].item() == 0:
            continue
        values = list(member[replacement_key])
        inputs[index, : len(values) + 1] = torch.tensor([BYTE_EOS, *values], device=device)
        labels[index, : len(values) + 1] = torch.tensor([*values, BYTE_EOS], device=device)
    return actions, pointers, inputs, labels
