"""Stage-typed composition with the frozen QTE1 entailment query owner."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from diverge_ats1_data import BYTE_OFFSET
from diverge_iem1_runtime import module_state_sha256
from diverge_pqi1_runtime import canonicalize_query


SCHEMA = "shohin-diverge-qte1-composite-v1"
SYSTEM = (
    "Judge whether a claim follows from an instruction. "
    "Answer exactly YES or NO without explanation."
)


class QTE1CompositeError(RuntimeError):
    """A QTE1 owner, source, or isolation contract differs."""


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _storage_pointers(parameters: Iterable[nn.Parameter]) -> set[int]:
    return {parameter.untyped_storage().data_ptr() for parameter in parameters}


def _ids(tokenizer: Any, text: str) -> list[int]:
    values = list(tokenizer.encode(text, add_special_tokens=False))
    if not values:
        raise QTE1CompositeError("QTE1 fixed text tokenized empty")
    return values


def _prompt(tokenizer: Any, query: str, candidate: int) -> str:
    if candidate not in (0, 1):
        raise QTE1CompositeError("QTE1 candidate differs")
    target = "alpha" if candidate == 0 else "beta"
    distractor = "beta" if candidate == 0 else "alpha"
    user = (
        f"Instruction: {query}\n"
        f"Claim: {target} is the requested answer source and {distractor} is the distractor."
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return f"System: {SYSTEM}\nUser: {user}\nAssistant:"


class QwenEntailmentQueryOwner(nn.Module):
    """Return complete-candidate YES/NO entailment log odds."""

    def __init__(
        self,
        backbone: nn.Module,
        tokenizer: Any,
        *,
        model_receipt: dict[str, str],
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        if batch_size != 32 or not model_receipt:
            raise QTE1CompositeError("QTE1 frozen query geometry differs")
        self.backbone = backbone.requires_grad_(False).eval()
        self.tokenizer = tokenizer
        self.model_receipt = dict(model_receipt)
        self.batch_size = batch_size
        self.answer_ids = tuple(tuple(_ids(tokenizer, value)) for value in ("YES", "NO"))

    def receipt_sha256(self) -> str:
        return _canonical_sha256(
            {
                "model_receipt": self.model_receipt,
                "system_sha256": hashlib.sha256(SYSTEM.encode("ascii")).hexdigest(),
                "answer_ids": self.answer_ids,
                "batch_size": self.batch_size,
            }
        )

    def _canonical_queries(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        symbol_masks: torch.Tensor,
    ) -> list[str]:
        if (
            byte_ids.ndim != 2
            or byte_ids.shape != attention_mask.shape
            or symbol_masks.shape[:2] != (byte_ids.shape[0], 2)
            or symbol_masks.shape[2] != byte_ids.shape[1]
        ):
            raise QTE1CompositeError("QTE1 byte interface differs")
        ids = byte_ids.detach().cpu()
        attention = attention_mask.detach().cpu()
        symbols = symbol_masks.detach().cpu()
        queries = []
        for row in range(ids.shape[0]):
            length = int(attention[row].sum())
            values = ids[row, 1:length] - BYTE_OFFSET
            if torch.any((values < 0) | (values > 127)):
                raise QTE1CompositeError("QTE1 source is not ASCII")
            text = bytes(int(value) for value in values.tolist()).decode("ascii")
            masks = [symbols[row, group, 1:length].tolist() for group in range(2)]
            queries.append(canonicalize_query(text, masks).text)
        return queries

    @torch.no_grad()
    def forward(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        symbol_masks: torch.Tensor,
    ) -> torch.Tensor:
        queries = self._canonical_queries(byte_ids, attention_mask, symbol_masks)
        rows: list[tuple[int, int, int, list[int], list[int]]] = []
        for row, query in enumerate(queries):
            for candidate in (0, 1):
                prompt = _ids(self.tokenizer, _prompt(self.tokenizer, query, candidate))
                for answer, suffix in enumerate(self.answer_ids):
                    rows.append((row, candidate, answer, prompt, list(suffix)))
        device = next(self.backbone.parameters()).device
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise QTE1CompositeError("QTE1 tokenizer has no padding token")
        likelihoods = torch.empty((len(queries), 2, 2), device=device)
        for start in range(0, len(rows), self.batch_size):
            batch = rows[start : start + self.batch_size]
            maximum = max(
                len(prompt) + len(suffix) - 1 for *_, prompt, suffix in batch
            )
            inputs = torch.full(
                (len(batch), maximum), int(pad_id), dtype=torch.long, device=device
            )
            mask = torch.zeros_like(inputs)
            for index, (*_, prompt, suffix) in enumerate(batch):
                sequence = prompt + suffix
                inputs[index, : len(sequence) - 1] = torch.tensor(
                    sequence[:-1], dtype=torch.long, device=device
                )
                mask[index, : len(sequence) - 1] = 1
            output = self.backbone(input_ids=inputs, attention_mask=mask, use_cache=False)
            probabilities = F.log_softmax(output.logits.float(), dim=-1)
            for index, (row, candidate, answer, prompt, suffix) in enumerate(batch):
                positions = torch.arange(
                    len(prompt) - 1,
                    len(prompt) + len(suffix) - 1,
                    device=device,
                )
                target = torch.tensor(suffix, dtype=torch.long, device=device)
                likelihoods[row, candidate, answer] = probabilities[
                    index, positions, target
                ].sum()
        return likelihoods[:, :, 0] - likelihoods[:, :, 1]


class QTE1StageTypedMachine(nn.Module):
    """Route protected stage owners and frozen QTE1 QUERY entailment."""

    def __init__(
        self,
        source_owner: nn.Module,
        evidence_owner: nn.Module,
        query_owner: QwenEntailmentQueryOwner,
    ) -> None:
        super().__init__()
        self.source_owner = source_owner
        self.numeric_evidence_owner = evidence_owner
        self.query_owner = query_owner
        self.requires_grad_(False)
        self._initial_hashes = self.owner_hashes()
        validate_composite(self)

    @property
    def evidence_owner(self) -> nn.Module:
        return self.numeric_evidence_owner

    def forward_query(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        symbol_masks: torch.Tensor,
    ) -> torch.Tensor:
        return self.query_owner(byte_ids, attention_mask, symbol_masks)

    def owner_hashes(self) -> dict[str, str]:
        return {
            "WORLD": module_state_sha256(self.source_owner),
            "EVIDENCE": module_state_sha256(self.numeric_evidence_owner),
            "QUERY": self.query_owner.receipt_sha256(),
        }

    def owner_manifest(self) -> dict[str, object]:
        validate_composite(self)
        return {
            "schema": SCHEMA,
            "owner_hashes": self.owner_hashes(),
            "initial_owner_hashes": dict(self._initial_hashes),
            "transaction_order": ["WORLD", "EVIDENCE", "EXECUTE", "QUERY"],
            "stage_routes": {
                "WORLD": "TOL3",
                "EVIDENCE": "NVE1_NUMERIC_AND_SYMBOL",
                "QUERY": "QTE1_QWEN_CANDIDATE_ENTAILMENT",
            },
            "cross_stage_parameter_sharing": False,
            "trainable_parameters": 0,
        }


def validate_composite(model: QTE1StageTypedMachine) -> None:
    owners = {
        "WORLD": _storage_pointers(model.source_owner.parameters()),
        "EVIDENCE": _storage_pointers(model.numeric_evidence_owner.parameters()),
        "QUERY": _storage_pointers(model.query_owner.parameters()),
    }
    for left_name, left in owners.items():
        for right_name, right in owners.items():
            if left_name < right_name and left & right:
                raise QTE1CompositeError(
                    f"QTE1 owners {left_name}/{right_name} alias storage"
                )
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise QTE1CompositeError("QTE1 composite contains a plastic parameter")


__all__ = [
    "QTE1CompositeError",
    "QTE1StageTypedMachine",
    "QwenEntailmentQueryOwner",
    "SCHEMA",
    "SYSTEM",
    "validate_composite",
]
