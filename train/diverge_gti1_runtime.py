"""Autoregressive typed-transaction interface for DIVERGE-GTI1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Literal, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from diverge_ats1_data import PAD_ID
from diverge_nve1_data import symbol_occurrence_groups
from diverge_pqi1_runtime import canonicalize_query
from hf_product_reasoning_train import install_lora


SCHEMA = "shohin-diverge-gti1-runtime-v1"
PROMPT_PREFIX = (
    "Read the instruction and emit exactly one typed transaction.\n"
    "Instruction: "
)
PROMPT_SUFFIX = "\nTransaction:"
TRANSACTIONS = (" READ(alpha)", " READ(beta)")
GTI1Control = Literal["normal", "scrub_context", "swap_mentions"]


class GTI1RuntimeError(RuntimeError):
    """A GTI1 source, transaction, or model contract differs."""


@dataclass(frozen=True, slots=True)
class GTI1Config:
    lora_layers: int = 4
    lora_rank: int = 8
    lora_alpha: float = 16.0

    def validate(self) -> None:
        if (
            self.lora_layers != 4
            or self.lora_rank != 8
            or self.lora_alpha != 16.0
        ):
            raise GTI1RuntimeError("GTI1 frozen LoRA geometry differs")


def _token_ids(tokenizer: Tokenizer, text: str) -> list[int]:
    ids = list(tokenizer.encode(text, add_special_tokens=False).ids)
    if not ids:
        raise GTI1RuntimeError("GTI1 fixed text tokenized empty")
    return ids


def transaction_token_ids(tokenizer: Tokenizer) -> tuple[tuple[int, ...], ...]:
    encoded = tuple(tuple(_token_ids(tokenizer, value)) for value in TRANSACTIONS)
    if len(encoded[0]) != len(encoded[1]):
        raise GTI1RuntimeError("GTI1 transaction token lengths differ")
    return encoded


def canonical_query_text(
    record: Mapping[str, Any], *, control: GTI1Control = "normal"
) -> str:
    if control not in ("normal", "scrub_context", "swap_mentions"):
        raise GTI1RuntimeError(f"unknown GTI1 control: {control}")
    text = str(record["source_text"])
    symbols = tuple(str(value) for value in record["symbols"])
    groups = symbol_occurrence_groups(text, symbols)
    if len(groups) != 2:
        raise GTI1RuntimeError("GTI1 query does not expose two mention groups")
    masks = []
    for _, spans in groups:
        mask = [False] * len(text)
        for left, right in spans:
            mask[left:right] = [True] * (right - left)
        masks.append(mask)
    canonical = canonicalize_query(
        text, masks, scrub_context=control == "scrub_context"
    ).text
    if control == "swap_mentions":
        canonical = canonical.replace("alpha", "__gti1_swap__")
        canonical = canonical.replace("beta", "alpha")
        canonical = canonical.replace("__gti1_swap__", "beta")
    return canonical


def render_prompt(
    record: Mapping[str, Any], *, control: GTI1Control = "normal"
) -> str:
    return PROMPT_PREFIX + canonical_query_text(record, control=control) + PROMPT_SUFFIX


def expected_transaction(record: Mapping[str, Any]) -> int:
    roles = tuple(int(value) for value in record["symbol_role_ids"])
    if sorted(roles) != [0, 1]:
        raise GTI1RuntimeError("GTI1 role permutation differs")
    return roles.index(0)


def adapter_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
        if ".lora_a." in name or ".lora_b." in name
    }


def _tensor_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def adapter_state_sha256(module: nn.Module) -> str:
    return _tensor_digest(adapter_state_dict(module))


def frozen_backbone_state_sha256(module: nn.Module) -> str:
    state = {}
    for name, tensor in module.state_dict().items():
        if ".lora_a." in name or ".lora_b." in name:
            continue
        canonical = name.replace(".base.", ".")
        if canonical in state:
            raise GTI1RuntimeError("GTI1 canonical frozen tensor name collides")
        state[canonical] = tensor
    return _tensor_digest(state)


def load_adapter_state(module: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    own = module.state_dict()
    expected = {
        name for name in own if ".lora_a." in name or ".lora_b." in name
    }
    if set(state) != expected:
        raise GTI1RuntimeError("GTI1 adapter tensor names differ")
    with torch.no_grad():
        for name in expected:
            if own[name].shape != state[name].shape:
                raise GTI1RuntimeError(f"GTI1 adapter tensor shape differs: {name}")
            own[name].copy_(state[name])


class GenerativeTransactionInterpreter(nn.Module):
    """Frozen causal backbone with final-block LoRA and legal transaction decoding."""

    def __init__(
        self,
        backbone: nn.Module,
        tokenizer: Tokenizer,
        config: GTI1Config,
    ) -> None:
        super().__init__()
        config.validate()
        if config.lora_layers > len(backbone.blocks):
            raise GTI1RuntimeError("GTI1 LoRA depth exceeds backbone")
        backbone.requires_grad_(False)
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.config = config
        self.frozen_state_before = frozen_backbone_state_sha256(backbone)
        self.lora_projection_count = 0
        for block in backbone.blocks[-config.lora_layers :]:
            self.lora_projection_count += install_lora(
                block, config.lora_rank, config.lora_alpha
            )
        if self.lora_projection_count == 0:
            raise GTI1RuntimeError("GTI1 installed no LoRA projections")
        self.transaction_ids = transaction_token_ids(tokenizer)
        if frozen_backbone_state_sha256(backbone) != self.frozen_state_before:
            raise GTI1RuntimeError("GTI1 LoRA installation changed a frozen tensor")

    def adapter_parameters(self) -> Iterable[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if ".lora_a." in name or ".lora_b." in name:
                yield parameter

    def encode_example(
        self,
        record: Mapping[str, Any],
        *,
        target: int | None,
        control: GTI1Control = "normal",
    ) -> tuple[list[int], list[int]]:
        prompt = _token_ids(self.tokenizer, render_prompt(record, control=control))
        if target is None:
            return prompt, []
        if target not in (0, 1):
            raise GTI1RuntimeError("GTI1 transaction target differs")
        return prompt, list(self.transaction_ids[target])

    def supervised_batch(
        self,
        records: Sequence[Mapping[str, Any]],
        targets: Sequence[int],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(records) != len(targets) or not records:
            raise GTI1RuntimeError("GTI1 supervised batch geometry differs")
        encoded = [
            self.encode_example(record, target=int(target))
            for record, target in zip(records, targets, strict=True)
        ]
        maximum = max(len(prompt) + len(suffix) - 1 for prompt, suffix in encoded)
        inputs = torch.full(
            (len(encoded), maximum), PAD_ID, dtype=torch.long, device=device
        )
        labels = torch.full_like(inputs, -100)
        for row, (prompt, suffix) in enumerate(encoded):
            sequence = prompt + suffix
            inputs[row, : len(sequence) - 1] = torch.tensor(
                sequence[:-1], dtype=torch.long, device=device
            )
            labels[row, len(prompt) - 1 : len(sequence) - 1] = torch.tensor(
                suffix, dtype=torch.long, device=device
            )
        return inputs, labels

    def supervised_loss(
        self,
        records: Sequence[Mapping[str, Any]],
        targets: Sequence[int],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        inputs, labels = self.supervised_batch(records, targets, device=device)
        logits, _ = self.backbone(inputs)
        return F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )

    @torch.no_grad()
    def candidate_scores(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        device: torch.device,
        batch_size: int,
        control: GTI1Control = "normal",
    ) -> torch.Tensor:
        rows: list[tuple[int, int, list[int], list[int]]] = []
        for row, record in enumerate(records):
            prompt, _ = self.encode_example(record, target=None, control=control)
            for candidate, suffix in enumerate(self.transaction_ids):
                if len(prompt) + len(suffix) > int(self.backbone.cfg.seq_len):
                    raise GTI1RuntimeError("GTI1 sequence exceeds backbone context")
                rows.append((row, candidate, prompt, list(suffix)))

        scores = torch.empty((len(records), 2), dtype=torch.float32)
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            maximum = max(
                len(prompt) + len(suffix) - 1
                for _, _, prompt, suffix in batch
            )
            inputs = torch.full(
                (len(batch), maximum), PAD_ID, dtype=torch.long, device=device
            )
            for index, (_, _, prompt, suffix) in enumerate(batch):
                sequence = prompt + suffix
                inputs[index, : len(sequence) - 1] = torch.tensor(
                    sequence[:-1], dtype=torch.long, device=device
                )
            logits, _ = self.backbone(inputs)
            log_probabilities = F.log_softmax(logits.float(), dim=-1)
            for index, (row, candidate, prompt, suffix) in enumerate(batch):
                positions = torch.arange(
                    len(prompt) - 1,
                    len(prompt) + len(suffix) - 1,
                    device=device,
                )
                target = torch.tensor(suffix, dtype=torch.long, device=device)
                scores[row, candidate] = log_probabilities[
                    index, positions, target
                ].sum().cpu()
        return scores


__all__ = [
    "GTI1Config",
    "GTI1Control",
    "GTI1RuntimeError",
    "GenerativeTransactionInterpreter",
    "PROMPT_PREFIX",
    "PROMPT_SUFFIX",
    "SCHEMA",
    "TRANSACTIONS",
    "adapter_state_dict",
    "adapter_state_sha256",
    "canonical_query_text",
    "expected_transaction",
    "frozen_backbone_state_sha256",
    "load_adapter_state",
    "render_prompt",
    "transaction_token_ids",
]
