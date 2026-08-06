from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from build_diverge_vcr1_board import BOARD_SCHEMA, build


class CharacterTokenizer:
    chat_template = None
    is_fast = True
    eos_token_id = 0
    pad_token_id = 0

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(character) + 1 for character in text]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ):
        assert not add_special_tokens and return_offsets_mapping
        return {
            "input_ids": self.encode(text),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _write_source(path: Path) -> None:
    rows = []
    for group in ("math", "science"):
        for index in range(60):
            identity = hashlib.sha256(f"{group}-{index}".encode()).hexdigest()
            rows.append(
                {
                    "schema": "shohin-product-verifier-preference-pairs-v1",
                    "identity_sha256": identity,
                    "pair_rank_sha256": hashlib.sha256(
                        f"pair-{group}-{index}".encode()
                    ).hexdigest(),
                    "training_group": group,
                    "question": f"{group} question {index}",
                    "chosen": f"The final answer is \\boxed{{{index}}}.",
                    "rejected": f"wrong response {index}",
                }
            )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_builder_makes_disjoint_balanced_exact_boards(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "pairs.jsonl"
    _write_source(source)
    excluded = tmp_path / "eval.jsonl"
    excluded.write_text(json.dumps({"question": "math question 0"}) + "\n")
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: CharacterTokenizer()
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    train = tmp_path / "train.jsonl"
    development = tmp_path / "development.jsonl"
    report_path = tmp_path / "report.json"
    args = argparse.Namespace(
        model_root=tmp_path,
        source=source,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        train_output=train,
        development_output=development,
        report=report_path,
        exclude_eval=[excluded],
        max_sequence_length=2000,
        workspace_slots=8,
        train_per_group=20,
        development_per_group=10,
        minimum_train_per_group=5,
        minimum_development_per_group=5,
        development_fraction=0.30,
        seed=2026080603,
    )
    report = build(args)
    train_rows = _rows(train)
    development_rows = _rows(development)
    assert len(train_rows) == 40
    assert len(development_rows) == 20
    assert {row["schema"] for row in train_rows} == {BOARD_SCHEMA}
    train_ids = {row["identity_sha256"] for row in train_rows}
    development_ids = {row["identity_sha256"] for row in development_rows}
    assert not train_ids & development_ids
    assert all(row["question"] != "math question 0" for row in train_rows)
    assert report["selected_truncations"] == 0
    assert report["identity_overlap"] == 0
    assert report["train_group_counts"] == {"math": 20, "science": 20}
    assert report["development_group_counts"] == {"math": 10, "science": 10}
