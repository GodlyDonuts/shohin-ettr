"""Outcome-grounded candidate entailment for DIVERGE-CGL1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Literal, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from diverge_ats1_data import PAD_ID
from diverge_gti1_runtime import canonical_query_text
from hf_product_reasoning_train import install_lora


SCHEMA = "shohin-diverge-cgl1-runtime-v1"
SYSTEM = (
    "Judge whether a claim follows from an instruction. "
    "Answer exactly YES or NO without explanation."
)
ANSWERS = (" YES", " NO")
CGL1Control = Literal["normal", "scrub_context", "swap_mentions"]


class CGL1RuntimeError(RuntimeError):
    """A CGL1 model or candidate-entailment transaction differs."""


@dataclass(frozen=True, slots=True)
class CGL1Config:
    lora_layers: int = 8
    lora_rank: int = 16
    lora_alpha: float = 32.0

    def validate(self) -> None:
        if (
            self.lora_layers != 8
            or self.lora_rank != 16
            or self.lora_alpha != 32.0
        ):
            raise CGL1RuntimeError("CGL1 frozen LoRA geometry differs")


def _token_ids(tokenizer: Tokenizer, text: str) -> list[int]:
    values = list(tokenizer.encode(text, add_special_tokens=False).ids)
    if not values:
        raise CGL1RuntimeError("CGL1 fixed text tokenized empty")
    return values


def render_claim_prompt(
    record: Mapping[str, Any],
    candidate: int,
    *,
    control: CGL1Control = "normal",
) -> str:
    if candidate not in (0, 1):
        raise CGL1RuntimeError("CGL1 candidate differs")
    query = canonical_query_text(record, control=control)
    target = "alpha" if candidate == 0 else "beta"
    distractor = "beta" if candidate == 0 else "alpha"
    return (
        f"Instruction: {SYSTEM}\n"
        f"Source: {query}\n"
        f"Claim: {target} is the requested answer source and "
        f"{distractor} is the distractor.\nAnswer:"
    )


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
        digest.update(
            tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
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
            raise CGL1RuntimeError("CGL1 canonical frozen tensor name collides")
        state[canonical] = tensor
    return _tensor_digest(state)


def load_adapter_state(module: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    own = module.state_dict()
    expected = {
        name for name in own if ".lora_a." in name or ".lora_b." in name
    }
    if set(state) != expected:
        raise CGL1RuntimeError("CGL1 adapter tensor names differ")
    with torch.no_grad():
        for name in expected:
            if own[name].shape != state[name].shape:
                raise CGL1RuntimeError(f"CGL1 adapter shape differs: {name}")
            own[name].copy_(state[name])


class CausalGroundingInterpreter(nn.Module):
    """Frozen causal backbone plus LoRA-scored complete transaction claims."""

    def __init__(
        self,
        backbone: nn.Module,
        tokenizer: Tokenizer,
        config: CGL1Config,
    ) -> None:
        super().__init__()
        config.validate()
        if config.lora_layers > len(backbone.blocks):
            raise CGL1RuntimeError("CGL1 LoRA depth exceeds backbone")
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
            raise CGL1RuntimeError("CGL1 installed no LoRA projections")
        self.answer_ids = tuple(
            tuple(_token_ids(tokenizer, answer)) for answer in ANSWERS
        )
        if frozen_backbone_state_sha256(backbone) != self.frozen_state_before:
            raise CGL1RuntimeError("CGL1 LoRA installation changed a frozen tensor")

    def adapter_parameters(self) -> Iterable[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if ".lora_a." in name or ".lora_b." in name:
                yield parameter

    def _candidate_rows(
        self,
        records: Sequence[Mapping[str, Any]],
        control: CGL1Control,
    ) -> list[tuple[list[int], list[int]]]:
        rows = []
        for record in records:
            for candidate in (0, 1):
                prompt = _token_ids(
                    self.tokenizer,
                    render_claim_prompt(record, candidate, control=control),
                )
                for suffix in self.answer_ids:
                    if len(prompt) + len(suffix) > int(self.backbone.cfg.seq_len):
                        raise CGL1RuntimeError("CGL1 sequence exceeds backbone context")
                    rows.append((prompt, list(suffix)))
        return rows

    def _score_rows(
        self,
        rows: Sequence[tuple[list[int], list[int]]],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        maximum = max(len(prompt) + len(suffix) - 1 for prompt, suffix in rows)
        inputs = torch.full(
            (len(rows), maximum), PAD_ID, dtype=torch.long, device=device
        )
        for index, (prompt, suffix) in enumerate(rows):
            sequence = prompt + suffix
            inputs[index, : len(sequence) - 1] = torch.tensor(
                sequence[:-1], dtype=torch.long, device=device
            )
        logits, _ = self.backbone(inputs)
        log_probabilities = F.log_softmax(logits.float(), dim=-1)
        values = []
        for index, (prompt, suffix) in enumerate(rows):
            positions = torch.arange(
                len(prompt) - 1,
                len(prompt) + len(suffix) - 1,
                device=device,
            )
            target = torch.tensor(suffix, dtype=torch.long, device=device)
            values.append(log_probabilities[index, positions, target].sum())
        return torch.stack(values)

    def training_scores(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if not records:
            raise CGL1RuntimeError("CGL1 training batch is empty")
        likelihoods = self._score_rows(
            self._candidate_rows(records, "normal"), device=device
        ).reshape(len(records), 2, 2)
        return likelihoods[:, :, 0] - likelihoods[:, :, 1]

    @torch.no_grad()
    def candidate_scores(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        device: torch.device,
        batch_size: int,
        control: CGL1Control = "normal",
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise CGL1RuntimeError("CGL1 evaluation batch size differs")
        outputs = []
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            likelihoods = self._score_rows(
                self._candidate_rows(batch, control), device=device
            ).reshape(len(batch), 2, 2)
            outputs.append((likelihoods[:, :, 0] - likelihoods[:, :, 1]).cpu())
        return torch.cat(outputs)


__all__ = [
    "ANSWERS",
    "CGL1Config",
    "CGL1Control",
    "CGL1RuntimeError",
    "CausalGroundingInterpreter",
    "SCHEMA",
    "SYSTEM",
    "adapter_state_dict",
    "adapter_state_sha256",
    "frozen_backbone_state_sha256",
    "load_adapter_state",
    "render_claim_prompt",
]
