"""Evaluate a Hugging Face reasoning backbone on frozen local math boards."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time
from typing import Any


class ProductEvalError(RuntimeError):
    """The product reasoning evaluator contract was violated."""


def _clean_number(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().replace(",", "").replace("$", "").rstrip(".")
    match = re.search(r"-?\d+(?:/\d+)?(?:\.\d+)?", cleaned)
    return match.group(0) if match else None


def extract_gsm8k(text: str) -> str | None:
    boxed_numbers = [
        _clean_number(value)
        for value in _boxed_values(text)
        if _clean_number(value) is not None
    ]
    if boxed_numbers:
        return boxed_numbers[-1]
    currency = re.findall(
        r"(?:answer|final answer)\s*(?:is|:)\s*\$?\s*"
        r"(-?[\d,]+)\s+dollars?(?:\s+and)?\s+([\d,]+)\s+cents?",
        text,
        flags=re.IGNORECASE,
    )
    if currency:
        dollars, cents = currency[-1]
        sign = -1 if dollars.startswith("-") else 1
        amount = Decimal(dollars.replace(",", "")) + sign * (
            Decimal(cents.replace(",", "")) / Decimal(100)
        )
        return format(amount, "f")
    explicit = re.findall(
        r"(?:answer|final answer)\s*(?:is|:)\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if explicit:
        return _clean_number(explicit[-1])
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return _clean_number(numbers[-1]) if numbers else None


def _normalize_aime_integer(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().replace("$", "").replace(",", "").rstrip(".")
    return normalized if re.fullmatch(r"\d{1,3}", normalized) else None


def extract_aime(text: str) -> str | None:
    """Extract only an explicit final AIME integer, never an intermediate number."""

    for value in reversed(_boxed_values(text)):
        normalized = _normalize_aime_integer(value)
        if normalized is not None:
            return normalized
    explicit = re.findall(
        r"(?:answer|final answer)\s*(?:is|:)\s*([^\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    for value in reversed(explicit):
        normalized = _normalize_aime_integer(value)
        if normalized is not None:
            return normalized
    return None


def _boxed_values(text: str) -> list[str]:
    values: list[str] = []
    cursor = 0
    while True:
        start = text.find(r"\boxed", cursor)
        if start < 0:
            break
        opening = text.find("{", start)
        if opening >= 0:
            depth = 0
            for index in range(opening, len(text)):
                if text[index] == "{":
                    depth += 1
                elif text[index] == "}":
                    depth -= 1
                    if depth == 0:
                        value = text[opening + 1 : index].strip()
                        if value:
                            values.append(value)
                        cursor = index + 1
                        break
            else:
                cursor = opening + 1
        else:
            cursor = start + len(r"\boxed")
    return values


def extract_boxed(text: str) -> str | None:
    boxed = _boxed_values(text)
    if boxed:
        return boxed[-1]
    fallback = re.findall(
        r"(?:answer|final answer)\s*(?:is|:)\s*\$?\s*([^\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    return _trim_explicit_answer(fallback[-1]) if fallback else None


def _trim_explicit_answer(value: str) -> str:
    """Stop an explicit answer before a new prose verification sentence."""

    answer = re.split(r"</think>|<\|", value, maxsplit=1)[0].strip()
    answer = re.split(
        r"\.\s+(?=(?:but|however|wait|hmm|let|i|this|that|so|therefore)\b)",
        answer,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    answer = re.split(
        r",\s+(?=(?:which|but|although|however)\b)",
        answer,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return answer.strip().rstrip(".")


def gold_gsm8k(row: dict[str, Any]) -> str | None:
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", str(row.get("answer", "")))
    return _clean_number(match.group(1)) if match else None


def gold_numeric_answer(row: dict[str, Any]) -> str | None:
    return _clean_number(str(row.get("answer", "")))


def _normalize_short_answer(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"\s+", " ", value.strip()).strip("$ ")
    while True:
        wrapped = re.fullmatch(
            r"\\(?:text|textrm|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}",
            normalized,
        )
        if wrapped is None:
            break
        normalized = wrapped.group(1).strip()
    normalized = normalized.strip(" .,:;\"'")
    if re.fullmatch(r"\(?[A-Za-z]\)?", normalized):
        normalized = normalized.strip("()").upper()
    return normalized.casefold()


def extract_short_answer(text: str) -> str | None:
    boxed = _boxed_values(text)
    if boxed:
        return boxed[-1]
    explicit = re.findall(
        r"(?:answer|final answer)\s*(?:is|:)\s*([^\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    if explicit:
        return _trim_explicit_answer(explicit[-1])
    labels = re.findall(r"(?<![A-Za-z])\(([A-Z])\)(?![A-Za-z])", text)
    if labels:
        return labels[-1]
    booleans = re.findall(r"\b(?:true|false|yes|no)\b", text, flags=re.IGNORECASE)
    if booleans:
        return booleans[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else None


def has_explicit_final_answer(completion: str) -> bool:
    """Return whether a trace deliberately emitted a final-answer marker."""

    return r"\boxed" in completion or bool(
        re.search(
            r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)",
            completion,
            flags=re.IGNORECASE,
        )
    )


def gold_short_answer(row: dict[str, Any]) -> str | None:
    target = row.get("target") or row.get("answer")
    return str(target) if target is not None else None


def match_short_answer(prediction: str | None, gold: str | None) -> bool:
    return (
        prediction is not None
        and gold is not None
        and _normalize_short_answer(prediction) == _normalize_short_answer(gold)
    )


def gold_math(row: dict[str, Any]) -> str | None:
    for key in ("answer", "solution", "expected_answer"):
        value = row.get(key)
        if value:
            boxed = extract_boxed(str(value))
            if boxed is not None:
                return boxed.replace(" ", "")
    return None


def match_gsm8k(prediction: str | None, gold: str | None) -> bool:
    if prediction is None or gold is None:
        return False
    predicted_number = _clean_number(prediction)
    gold_number = _clean_number(gold)
    if predicted_number is None or gold_number is None:
        return False

    def numeric_fraction(value: str) -> Fraction | None:
        try:
            if "/" in value:
                numerator, denominator = value.split("/", 1)
                return Fraction(int(numerator), int(denominator))
            return Fraction(Decimal(value))
        except (InvalidOperation, ValueError, ZeroDivisionError):
            return None

    predicted_fraction = numeric_fraction(predicted_number)
    gold_fraction = numeric_fraction(gold_number)
    return predicted_fraction is not None and predicted_fraction == gold_fraction


def match_aime(prediction: str | None, gold: str | None) -> bool:
    predicted = _normalize_aime_integer(prediction)
    expected = _normalize_aime_integer(gold)
    return (
        predicted is not None
        and expected is not None
        and int(predicted) == int(expected)
    )


def _normalize_math(value: str | None) -> str | None:
    if value is None:
        return None
    return (
        value.replace(" ", "")
        .replace(r"\dfrac", r"\frac")
        .replace(r"\tfrac", r"\frac")
        .replace("$", "")
        .rstrip(".")
    )


def _simple_math_fraction(value: str | None) -> Fraction | None:
    """Parse plain numeric answers when the symbolic backend declines them."""

    normalized = _normalize_math(value)
    if normalized is None:
        return None
    compact = normalized.replace(r"\left", "").replace(r"\right", "").replace(",", "")
    latex_fraction = re.fullmatch(
        r"\\frac(?:\{([+-]?\d+)\}|([+-]?\d+))\{([+-]?\d+)\}",
        compact,
    )
    if latex_fraction is not None:
        numerator = latex_fraction.group(1) or latex_fraction.group(2)
        try:
            return Fraction(int(numerator), int(latex_fraction.group(3)))
        except ZeroDivisionError:
            return None
    plain_fraction = re.fullmatch(r"([+-]?\d+)/([+-]?\d+)", compact)
    if plain_fraction is not None:
        try:
            return Fraction(int(plain_fraction.group(1)), int(plain_fraction.group(2)))
        except ZeroDivisionError:
            return None
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", compact):
        try:
            return Fraction(Decimal(compact))
        except InvalidOperation:
            return None
    return None


def _normalize_math_choice(value: str | None) -> str | None:
    """Normalize a single MATH multiple-choice label without touching expressions."""

    normalized = _normalize_short_answer(value)
    return normalized if normalized in {"a", "b", "c", "d", "e"} else None


def match_math(prediction: str | None, gold: str | None) -> bool:
    if prediction is None or gold is None:
        return False
    predicted_choice = _normalize_math_choice(prediction)
    gold_choice = _normalize_math_choice(gold)
    if predicted_choice is not None and gold_choice is not None:
        return predicted_choice == gold_choice
    predicted_number = _simple_math_fraction(prediction)
    gold_number = _simple_math_fraction(gold)
    if predicted_number is not None and gold_number is not None:
        return predicted_number == gold_number
    try:
        from math_verify import LatexExtractionConfig, parse, verify

        extraction = [LatexExtractionConfig()]
        parsed_gold = parse(f"${gold}$", extraction_config=extraction)
        parsed_prediction = parse(f"${prediction}$", extraction_config=extraction)
        if parsed_gold and parsed_prediction and verify(parsed_gold, parsed_prediction):
            return True
    except (ImportError, RuntimeError, TypeError, ValueError):
        pass
    return _normalize_math(prediction) == _normalize_math(gold)


TASKS: dict[str, dict[str, Any]] = {
    "aime": {
        "kind": "answer",
        "extract": extract_aime,
        "gold": gold_numeric_answer,
        "match": match_aime,
    },
    "bbh_logic": {
        "kind": "answer",
        "extract": extract_short_answer,
        "gold": gold_short_answer,
        "match": match_short_answer,
    },
    "gsm8k": {
        "kind": "answer",
        "extract": extract_gsm8k,
        "gold": gold_gsm8k,
        "match": match_gsm8k,
    },
    "gpqa": {
        "kind": "answer",
        "extract": extract_short_answer,
        "gold": gold_short_answer,
        "match": match_short_answer,
    },
    "math500": {
        "kind": "answer",
        "extract": extract_boxed,
        "gold": gold_math,
        "match": match_math,
    },
    "preformatted_short_answer": {
        "kind": "answer",
        "extract": extract_short_answer,
        "gold": gold_short_answer,
        "match": match_short_answer,
    },
    "humaneval": {"kind": "code"},
    "mbpp": {"kind": "code"},
}


def _question(row: dict[str, Any]) -> str:
    for key in ("question", "problem", "prompt", "text", "input"):
        value = row.get(key)
        if value:
            return str(value)
    raise ProductEvalError("evaluation row has no question")


def _task_prompt(task: str, row: dict[str, Any]) -> str:
    if task == "humaneval":
        return (
            "Complete the Python function below. Return only executable Python "
            "code containing the complete function, without Markdown fences.\n\n"
            f"{row['prompt']}"
        )
    if task == "mbpp":
        tests = "\n".join(str(item) for item in row.get("test_list", ()))
        return (
            "Write Python code that solves the task and passes every test. Return "
            "only executable Python code, without Markdown fences.\n\nTask:\n"
            f"{row['text']}\n\nTests:\n{tests}"
        )
    if task == "bbh_logic":
        return (
            f"{_question(row)}\n\nReason carefully, then put only the exact requested "
            "answer or option label inside \\boxed{}."
        )
    if task == "gpqa":
        choices = "\n".join(
            f"({choice['label']}) {choice['text']}" for choice in row["choices"]
        )
        return (
            f"{row['question']}\n\n{choices}\n\nReason carefully, then put only "
            "the correct option label inside \\boxed{}."
        )
    return _question(row)


def _strip_reasoning_and_fences(completion: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL).strip()
    fenced = re.findall(
        r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE
    )
    if fenced:
        text = fenced[-1].strip()
    text = re.sub(r"^\s*(?:answer|final answer)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def _truncate_code(completion: str, stops: tuple[str, ...]) -> str:
    code = _strip_reasoning_and_fences(completion)
    locations = [location for stop in stops if (location := code.find(stop)) >= 0]
    return code[: min(locations)].rstrip() if locations else code.rstrip()


def _humaneval_candidate_source(row: dict[str, Any], completion: str) -> str:
    code = _truncate_code(
        completion,
        ("\nif __name__", "\nprint(", "\nassert ", "\nQuestion:"),
    )
    prompt = str(row["prompt"])
    if re.search(r"(?m)^\s*(?:async\s+)?def\s+", code):
        definition = re.search(r"(?m)^\s*(?:async\s+)?def\s+", prompt)
        preamble = prompt[: definition.start()] if definition is not None else ""
        return preamble + code
    indented = "\n".join(f"    {line}" if line else line for line in code.splitlines())
    return prompt + indented


def _humaneval_program(row: dict[str, Any], completion: str) -> str:
    candidate = _humaneval_candidate_source(row, completion)
    return candidate + "\n\n" + str(row["test"]) + f"\ncheck({row['entry_point']})\n"


def _mbpp_program(row: dict[str, Any], completion: str) -> str:
    code = _truncate_code(
        completion,
        ("\n[DONE]", "\nQuestion:", "\nif __name__", "\n>>>"),
    )
    setup = str(row.get("test_setup_code", "") or "")
    tests = "\n".join(str(item) for item in row.get("test_list", ()))
    return code + "\n" + setup + "\n" + tests + "\n"


def _diagnostic_text(value: str | bytes | None, limit: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-limit:]


def _bounded_program_result(program: str, timeout_seconds: float) -> dict[str, Any]:
    """Execute generated code with wall/CPU/memory limits and bounded diagnostics."""

    if timeout_seconds <= 0:
        raise ProductEvalError("code timeout must be positive")
    with tempfile.TemporaryDirectory(prefix="shohin-product-code-") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(program, encoding="utf-8")

        def limits() -> None:
            try:
                import resource

                cpu = max(1, int(timeout_seconds))
                resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
                resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
                resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
                resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
            except (ImportError, OSError, ValueError):
                pass

        try:
            result = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                preexec_fn=limits,
                check=False,
            )
            return {
                "passed": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": _diagnostic_text(result.stdout, 2000),
                "stderr": _diagnostic_text(result.stderr, 4000),
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "passed": False,
                "returncode": None,
                "stdout": _diagnostic_text(exc.stdout, 2000),
                "stderr": "execution timed out",
            }


def _row_identity(task: str, row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{task}\0{_question(row)}".encode()).hexdigest()


def select_rows(
    task: str,
    rows: list[dict[str, Any]],
    count: int,
    subset_seed: int,
) -> list[dict[str, Any]]:
    if count <= 0 or count > len(rows):
        raise ProductEvalError("requested row count is outside the board")
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{subset_seed}\0{_row_identity(task, row)}".encode()
        ).hexdigest(),
    )
    selected = ranked[:count]
    identities = [_row_identity(task, row) for row in selected]
    if len(set(identities)) != len(identities):
        raise ProductEvalError("selected board contains duplicate prompts")
    return selected


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise ProductEvalError(f"refusing to replace evaluation report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _load_model(
    model_root: Path,
    adapter_checkpoint: Path | None,
    model_loader: str,
    quantization: str = "none",
):
    import torch

    from hf_product_reasoning_train import (
        ProductReasoningModel,
        load_product_backbone,
        load_trainable_checkpoint,
    )

    metadata = None
    if adapter_checkpoint is not None:
        payload = torch.load(adapter_checkpoint, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ProductEvalError("adapter checkpoint metadata is missing")
        checkpoint_loader = str(metadata.get("model_loader", model_loader))
        if model_loader != "auto" and checkpoint_loader != model_loader:
            raise ProductEvalError("adapter checkpoint model loader differs")
        model_loader = checkpoint_loader
    checkpoint_quantization = (
        str(metadata.get("quantization", "none"))
        if metadata is not None
        else quantization
    )
    if metadata is not None and quantization != "none" and (
        quantization != checkpoint_quantization
    ):
        raise ProductEvalError("adapter checkpoint quantization differs")
    backbone, resolved_model_loader = load_product_backbone(
        model_root,
        model_loader,
        dtype=torch.bfloat16,
        device_map={"": 0},
        quantization=checkpoint_quantization,
    )
    if adapter_checkpoint is None:
        return backbone.eval(), None, resolved_model_loader

    assert metadata is not None
    if metadata.get("model_loader") not in (None, resolved_model_loader):
        raise ProductEvalError("resolved model loader differs from checkpoint")
    workspace = metadata.get("workspace_config") or {}
    if metadata.get("architecture") == "diverge-ltm1":
        from diverge_ltm1_product import LTM1ProductModel

        model = LTM1ProductModel(
            backbone,
            lora_layers=int(metadata["lora_layers"]),
            lora_rank=int(metadata["lora_rank"]),
            lora_alpha=float(metadata["lora_alpha"]),
            latent_width=int(workspace["latent_width"]),
            trajectory_slots=int(workspace["trajectory_slots"]),
            recurrent_steps=int(workspace["recurrent_steps"]),
            fault_bits=int(workspace["fault_bits"]),
            attention_heads=int(workspace["attention_heads"]),
            ff_multiplier=int(workspace["ff_multiplier"]),
            trace_weight=float(metadata["trace_weight"]),
            balance_weight=float(metadata["balance_weight"]),
            halting_weight=float(metadata["halting_weight"]),
        ).to("cuda:0")
    elif metadata.get("architecture") == "diverge-qst1":
        from diverge_qst1_product import QST1ProductModel

        model = QST1ProductModel(
            backbone,
            lora_layers=int(metadata["lora_layers"]),
            lora_rank=int(metadata["lora_rank"]),
            lora_alpha=float(metadata["lora_alpha"]),
            workspace_width=int(workspace["workspace_width"]),
            source_slots=int(workspace["source_slots"]),
            state_slots=int(workspace["state_slots"]),
            query_slots=int(workspace["query_slots"]),
            recurrent_steps=int(workspace["recurrent_steps"]),
            attention_heads=int(workspace["attention_heads"]),
            ff_multiplier=int(workspace["ff_multiplier"]),
            binding_temperature=float(workspace["binding_temperature"]),
            binding_weight=float(metadata["binding_weight"]),
            reset_weight=float(metadata["reset_weight"]),
            halting_weight=float(metadata["halting_weight"]),
        ).to("cuda:0")
    elif metadata.get("architecture") == "diverge-qpt1":
        from diverge_qpt1_product import QPT1ProductModel

        model = QPT1ProductModel(
            backbone,
            lora_layers=int(metadata["lora_layers"]),
            lora_rank=int(metadata["lora_rank"]),
            lora_alpha=float(metadata["lora_alpha"]),
            workspace_width=int(workspace["workspace_width"]),
            source_slots=int(workspace["source_slots"]),
            query_slots=int(workspace["query_slots"]),
            recurrent_steps=int(workspace["recurrent_steps"]),
            attention_heads=int(workspace["attention_heads"]),
            ff_multiplier=int(workspace["ff_multiplier"]),
            pointer_temperature=float(workspace["pointer_temperature"]),
            binding_weight=float(metadata["binding_weight"]),
            coverage_weight=float(metadata["coverage_weight"]),
            reset_weight=float(metadata["reset_weight"]),
        ).to("cuda:0")
    elif metadata.get("architecture") == "diverge-sag1":
        from diverge_sag1_product import SAG1ProductModel

        model = SAG1ProductModel(
            backbone,
            base_checkpoint=Path(metadata["base_checkpoint"]),
            base_checkpoint_sha256=str(metadata["base_checkpoint_sha256"]),
            model_revision=str(metadata["model_revision"]),
            lora_layers=int(metadata["lora_layers"]),
            lora_rank=int(metadata["lora_rank"]),
            lora_alpha=float(metadata["lora_alpha"]),
            workspace_width=int(workspace["workspace_width"]),
            source_slots=int(workspace["source_slots"]),
            query_slots=int(workspace["query_slots"]),
            recurrent_steps=int(workspace["recurrent_steps"]),
            attention_heads=int(workspace["attention_heads"]),
            ff_multiplier=int(workspace["ff_multiplier"]),
            pointer_temperature=float(workspace["pointer_temperature"]),
            binding_weight=float(metadata["binding_weight"]),
            coverage_weight=float(metadata["coverage_weight"]),
            reset_weight=float(metadata["reset_weight"]),
            router_hidden=int(metadata["router_hidden"]),
            advantage_margin=float(metadata["advantage_margin"]),
            router_threshold=float(metadata["router_threshold"]),
            router_weight=float(metadata["router_weight"]),
            risk_weight=float(metadata["risk_weight"]),
            sparsity_weight=float(metadata["sparsity_weight"]),
        ).to("cuda:0")
    elif metadata.get("architecture") == "shohin-drem1-moe-revision-v1":
        from drem1_moe_revision import DREM1Config, DREM1ProductModel

        drem = metadata.get("drem1_config")
        if not isinstance(drem, dict):
            raise ProductEvalError("DREM1 checkpoint config is missing")
        model = DREM1ProductModel(
            backbone,
            DREM1Config(**drem),
            mode=str(metadata["drem1_mode"]),
            collapse_weight=float(metadata["collapse_weight"]),
            context_control=str(metadata.get("drem1_context_control", "normal")),
        ).to("cuda:0")
    elif metadata.get("architecture") == "shohin-ecr1-moe-revision-v1":
        from ecr1_moe_revision import ECR1Config, ECR1ProductModel

        ecr = metadata.get("ecr1_config")
        if not isinstance(ecr, dict):
            raise ProductEvalError("ECR1 checkpoint config is missing")
        model = ECR1ProductModel(
            backbone,
            ECR1Config(**ecr),
            draft_control=str(metadata.get("ecr1_draft_control", "normal")),
        ).to("cuda:0")
    else:
        model = ProductReasoningModel(
            backbone=backbone,
            arm=str(metadata["arm"]),
            lora_layers=int(metadata["lora_layers"]),
            lora_rank=int(metadata["lora_rank"]),
            lora_alpha=float(metadata["lora_alpha"]),
            workspace_width=int(workspace.get("workspace_width", 512)),
            workspace_slots=int(workspace.get("workspace_slots", 16)),
            recurrent_steps=int(workspace.get("recurrent_steps", 8)),
            dense_width=(
                int(workspace.get("workspace_width", 192))
                if str(metadata["arm"]).startswith("dense")
                else 192
            ),
            unfreeze_layers=int(metadata.get("unfreeze_layers", 0)),
            lora_scope=str(metadata.get("lora_scope", "all")),
        )
        if metadata.get("quantization", "none") == "none":
            model.to("cuda:0")
    update, restored_metadata = load_trainable_checkpoint(adapter_checkpoint, model)
    model.eval()
    return (
        model,
        {
            "update": update,
            "model_loader": resolved_model_loader,
            **restored_metadata,
        },
        resolved_model_loader,
    )


def _generation_arguments(mode: str, max_new_tokens: int) -> dict[str, Any]:
    common: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if mode == "greedy":
        return {**common, "do_sample": False}
    if mode == "qwen-thinking":
        return {
            **common,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.0,
        }
    raise ProductEvalError("unsupported generation mode")


def _set_inference_control(
    model: Any,
    adapter_metadata: dict[str, Any] | None,
    control: str,
) -> None:
    """Apply a frozen QPT1 causal intervention or fail closed."""

    if control == "normal":
        if (
            adapter_metadata is not None
            and adapter_metadata.get("architecture") == "diverge-qpt1"
        ):
            model.set_control(control)
        return
    if (
        adapter_metadata is None
        or adapter_metadata.get("architecture") != "diverge-qpt1"
        or not hasattr(model, "set_control")
    ):
        raise ProductEvalError("QPT1 control requires a QPT1 checkpoint")
    model.set_control(control)


def _completion_usage(
    token_ids: list[int],
    stop_token_ids: list[int],
    max_new_tokens: int,
) -> tuple[int, bool]:
    stop_positions = [
        token_ids.index(token_id)
        for token_id in stop_token_ids
        if token_id in token_ids
    ]
    if stop_positions:
        return min(stop_positions) + 1, False
    return len(token_ids), len(token_ids) >= max_new_tokens


def _generation_stop_token_ids(tokenizer: Any) -> list[int]:
    """Stop at EOS or any new chat turn instead of decoding a fake dialogue."""

    stop_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        stop_ids.append(int(tokenizer.eos_token_id))
    for token in ("<|im_start|>", "<|user|>", "<|assistant|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if (
            isinstance(token_id, int)
            and token_id >= 0
            and token_id != tokenizer.unk_token_id
            and tokenizer.convert_ids_to_tokens(token_id) == token
            and token_id not in stop_ids
        ):
            stop_ids.append(token_id)
    if not stop_ids:
        raise ProductEvalError("tokenizer exposes no generation stop token")
    return stop_ids


def _make_prompt(question: str) -> str:
    return (
        "Solve the problem carefully. Show the reasoning needed to verify the "
        "result, then put only the final answer inside \\boxed{}.\n\nProblem:\n"
        f"{question}"
    )


def _render_prompt(
    tokenizer: Any,
    question: str,
    adapter: bool,
    enable_thinking: bool,
) -> str:
    from hf_product_reasoning_train import (
        PRODUCT_SYSTEM_PROMPT,
        render_reasoning_messages,
    )

    if adapter:
        messages = [
            {
                "role": "system",
                "content": PRODUCT_SYSTEM_PROMPT,
            },
            {"role": "user", "content": question},
        ]
        thinking = False
    else:
        messages = [{"role": "user", "content": _make_prompt(question)}]
        thinking = enable_thinking
    return render_reasoning_messages(
        tokenizer,
        messages,
        enable_thinking=thinking,
    )


def _finalization_question(question: str, completion: str) -> str:
    return (
        "The draft below reached its generation limit before reliably emitting "
        "a final answer. Use only the original problem and draft. Do not redo or "
        "extend the reasoning. Return only one explicit final answer inside "
        "\\boxed{}.\n\nOriginal problem:\n"
        f"{question}\n\nDraft reasoning:\n{completion}\n\nEnd of draft. Do not "
        "continue the reasoning. Respond now with exactly one line in this "
        "form: \\boxed{answer}"
    )


def _completion_for_scoring(
    completion: str,
    exhausted: bool,
    finalization: str | None,
) -> str | None:
    if not exhausted or has_explicit_final_answer(completion):
        return completion
    if finalization is not None and has_explicit_final_answer(finalization):
        return finalization
    return None


def _generate_adapter(
    model: Any,
    encoded: dict[str, Any],
    generation_arguments: dict[str, Any],
    pad_token_id: int,
):
    import torch

    # LoRA and workspace parameters remain FP32 for optimizer stability while
    # the frozen backbone is BF16. Training already runs under autocast; the
    # same mixed-precision contract must hold during autonomous generation.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if hasattr(model, "generation_embeddings"):
            embeddings, attention = model.generation_embeddings(
                encoded["input_ids"], encoded["attention_mask"]
            )
        else:
            from hf_product_reasoning_train import product_generation_embeddings

            embeddings, attention = product_generation_embeddings(
                model,
                encoded["input_ids"],
                encoded["attention_mask"],
            )
        extra_arguments = {}
        if hasattr(model, "generation_position_ids"):
            extra_arguments["position_ids"] = model.generation_position_ids()
        return model.backbone.generate(
            inputs_embeds=embeddings,
            attention_mask=attention,
            pad_token_id=pad_token_id,
            **extra_arguments,
            **generation_arguments,
        )


def _generate_completions(
    model: Any,
    tokenizer: Any,
    rendered: list[str],
    adapter: bool,
    generation_mode: str,
    max_new_tokens: int,
    stop_token_ids: list[int],
) -> tuple[list[str], list[tuple[int, bool]]]:
    import torch

    encoded = tokenizer(rendered, padding=True, return_tensors="pt")
    encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
    if adapter and hasattr(model, "prepare_generation_draft_indicator"):
        model.prepare_generation_draft_indicator(
            tokenizer,
            rendered,
            encoded["input_ids"],
            encoded["attention_mask"],
        )
    if adapter and hasattr(model, "prepare_generation_draft_attention"):
        model.prepare_generation_draft_attention(
            tokenizer,
            rendered,
            encoded["input_ids"],
            encoded["attention_mask"],
        )
    prompt_width = int(encoded["input_ids"].shape[1])
    with torch.inference_mode():
        generation_arguments = _generation_arguments(generation_mode, max_new_tokens)
        generation_arguments["eos_token_id"] = (
            stop_token_ids[0] if len(stop_token_ids) == 1 else stop_token_ids
        )
        if not adapter:
            output = model.generate(
                **encoded,
                pad_token_id=tokenizer.pad_token_id,
                **generation_arguments,
            )
            completion_ids = output[:, prompt_width:]
        else:
            completion_ids = _generate_adapter(
                model,
                encoded,
                generation_arguments,
                tokenizer.pad_token_id,
            )
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    usage = [
        _completion_usage(token_row.tolist(), stop_token_ids, max_new_tokens)
        for token_row in completion_ids
    ]
    return completions, usage


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    task = TASKS[args.task]
    data_bytes = args.data.read_bytes()
    rows = [json.loads(line) for line in data_bytes.splitlines() if line.strip()]
    selected = select_rows(args.task, rows, args.count, args.subset_seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, adapter_metadata, resolved_model_loader = _load_model(
        args.model_root,
        args.adapter_checkpoint,
        args.model_loader,
    )
    if hasattr(model, "set_selection_strategy"):
        model.set_selection_strategy(args.ltm_selection)
    if hasattr(model, "set_code_intervention"):
        model.set_code_intervention(args.ecr_code_intervention)
    _set_inference_control(model, adapter_metadata, args.qpt_control)
    if hasattr(model, "reset_routing_receipt"):
        model.reset_routing_receipt()
    stop_token_ids = _generation_stop_token_ids(tokenizer)

    random.seed(args.generation_seed)
    torch.manual_seed(args.generation_seed)
    torch.cuda.manual_seed_all(args.generation_seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    correct = 0
    generated_tokens = 0
    max_token_exhausted = 0
    finalization_attempts = 0
    finalization_generated_tokens = 0
    finalization_max_token_exhausted = 0
    results: list[dict[str, Any]] = []

    for offset in range(0, len(selected), args.batch_size):
        batch = selected[offset : offset + args.batch_size]
        rendered = [
            _render_prompt(
                tokenizer,
                _task_prompt(args.task, row),
                args.adapter_checkpoint is not None,
                args.enable_thinking,
            )
            for row in batch
        ]
        completions, completion_usage = _generate_completions(
            model,
            tokenizer,
            rendered,
            args.adapter_checkpoint is not None,
            args.generation_mode,
            args.max_new_tokens,
            stop_token_ids,
        )
        finalizations: list[str | None] = [None] * len(batch)
        finalization_usage: list[tuple[int, bool]] = [(0, False)] * len(batch)
        if args.finalize_exhausted and task["kind"] != "code":
            finalize_indices = [
                index
                for index, (_, exhausted) in enumerate(completion_usage)
                if exhausted and not has_explicit_final_answer(completions[index])
            ]
            if finalize_indices:
                finalize_rendered = [
                    _render_prompt(
                        tokenizer,
                        _finalization_question(
                            _task_prompt(args.task, batch[index]),
                            completions[index],
                        ),
                        args.adapter_checkpoint is not None,
                        False,
                    )
                    for index in finalize_indices
                ]
                recovered, recovered_usage = _generate_completions(
                    model,
                    tokenizer,
                    finalize_rendered,
                    args.adapter_checkpoint is not None,
                    "greedy",
                    args.finalize_max_new_tokens,
                    stop_token_ids,
                )
                for index, recovered_text, recovered_count in zip(
                    finalize_indices, recovered, recovered_usage, strict=True
                ):
                    finalizations[index] = recovered_text
                    finalization_usage[index] = recovered_count
        for row, completion, (token_count, exhausted), finalization, (
            finalize_token_count,
            finalize_exhausted,
        ) in zip(
            batch,
            completions,
            completion_usage,
            finalizations,
            finalization_usage,
            strict=True,
        ):
            execution = None
            program = None
            if task["kind"] == "code":
                program = (
                    _humaneval_program(row, completion)
                    if args.task == "humaneval"
                    else _mbpp_program(row, completion)
                )
                execution = _bounded_program_result(program, args.code_timeout)
                prediction = "pass" if execution["passed"] else "fail"
                gold = "pass"
                is_correct = bool(execution["passed"])
            else:
                scoring_completion = _completion_for_scoring(
                    completion,
                    exhausted,
                    finalization,
                )
                prediction = (
                    task["extract"](scoring_completion)
                    if scoring_completion is not None
                    else None
                )
                gold = task["gold"](row)
                is_correct = bool(task["match"](prediction, gold))
            correct += int(is_correct)
            generated_tokens += token_count + finalize_token_count
            max_token_exhausted += int(exhausted)
            finalization_attempts += int(finalization is not None)
            finalization_generated_tokens += finalize_token_count
            finalization_max_token_exhausted += int(finalize_exhausted)
            identity = _row_identity(args.task, row)
            results.append(
                {
                    "identity_sha256": identity,
                    "question": _question(row),
                    "gold": gold,
                    "prediction": prediction,
                    "correct": is_correct,
                    "generated_tokens": token_count,
                    "max_token_exhausted": exhausted,
                    "completion": completion,
                    "finalization_completion": finalization,
                    "finalization_generated_tokens": finalize_token_count,
                    "finalization_max_token_exhausted": finalize_exhausted,
                    "program": program,
                    "execution": execution,
                }
            )
        print(
            f"[product-eval] {min(offset + len(batch), len(selected))}/"
            f"{len(selected)} correct={correct}",
            flush=True,
        )

    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    return {
        "schema": "shohin-hf-product-reasoning-eval-v3",
        "status": "complete",
        "model_root": str((args.model_source_root or args.model_root).resolve()),
        "loaded_model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": resolved_model_loader,
        "adapter_checkpoint": (
            str(args.adapter_checkpoint.resolve()) if args.adapter_checkpoint else None
        ),
        "adapter_metadata": adapter_metadata,
        "routing_receipt": (
            model.routing_receipt() if hasattr(model, "routing_receipt") else None
        ),
        "task": args.task,
        "data": str(args.data.resolve()),
        "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "subset_seed": args.subset_seed,
        "generation_seed": args.generation_seed,
        "generation_mode": args.generation_mode,
        "enable_thinking": args.enable_thinking,
        "ltm_selection": args.ltm_selection,
        "qpt_control": args.qpt_control,
        "ecr_code_intervention": args.ecr_code_intervention,
        "effective_enable_thinking": (
            args.enable_thinking if args.adapter_checkpoint is None else False
        ),
        "max_new_tokens": args.max_new_tokens,
        "generation_stop_token_ids": stop_token_ids,
        "batch_size": args.batch_size,
        "code_timeout_seconds": (args.code_timeout if task["kind"] == "code" else None),
        "correct": correct,
        "total": len(selected),
        "accuracy": correct / len(selected),
        "elapsed_seconds": elapsed,
        "generated_tokens": generated_tokens,
        "max_token_exhausted": max_token_exhausted,
        "cap_exhausted_explicit_answer_required": True,
        "finalize_exhausted": args.finalize_exhausted,
        "finalize_max_new_tokens": args.finalize_max_new_tokens,
        "finalization_attempts": finalization_attempts,
        "finalization_generated_tokens": finalization_generated_tokens,
        "finalization_max_token_exhausted": finalization_max_token_exhausted,
        "generated_tokens_per_second": generated_tokens / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "selection_sha256": hashlib.sha256(
            "\n".join(result["identity_sha256"] for result in results).encode()
        ).hexdigest(),
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--model-loader",
        choices=("auto", "causal", "multimodal"),
        default="auto",
    )
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--subset-seed", type=int, default=20260802)
    parser.add_argument("--generation-seed", type=int, default=31)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--finalize-exhausted", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--finalize-max-new-tokens", type=int, default=64)
    parser.add_argument("--code-timeout", type=float, default=8.0)
    parser.add_argument(
        "--generation-mode", choices=("greedy", "qwen-thinking"), default="greedy"
    )
    parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--ltm-selection",
        choices=("highest_prior", "lowest_prior", "reset"),
        default="highest_prior",
    )
    parser.add_argument(
        "--qpt-control",
        choices=("normal", "packet_swap", "state_reset", "release_off"),
        default="normal",
    )
    parser.add_argument(
        "--ecr-code-intervention",
        choices=("normal", "zero", "mean", "permutation"),
        default="normal",
    )
    args = parser.parse_args()
    if (
        args.batch_size <= 0
        or args.max_new_tokens <= 0
        or args.finalize_max_new_tokens <= 0
        or args.code_timeout <= 0
    ):
        parser.error("batch size and generation limits must be positive")
    return args


def main() -> int:
    args = parse_args()
    report = run(args)
    _atomic_json(args.output, report)
    print(
        f"[product-eval] {args.task} {report['correct']}/{report['total']} "
        f"= {report['accuracy']:.2%}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
