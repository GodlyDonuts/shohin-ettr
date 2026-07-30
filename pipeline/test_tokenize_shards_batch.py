"""End-to-end parity gates for batched deterministic shard tokenization."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "pipeline" / "tokenize_shards.py"
TOKENIZER = ROOT / "artifacts" / "shohin-tok-32k.json"
FIRST_TEXT = (
    "The first retained document explains a technical concept clearly.\n"
    "It gives a separate supporting detail in a second line.\n"
    "A final line states the conclusion without boilerplate."
)
SECOND_TEXT = (
    "A second retained document expands the earlier technical discussion.\n"
    "Its middle line supplies an independent concrete example.\n"
    "The final line is different from every preceding line."
)
THIRD_TEXT = (
    "A separate domain gives the tokenizer another valid record.\n"
    "This line provides a second piece of useful context.\n"
    "The final line keeps the record distinct and well formed."
)


def _run(
    source: Path,
    output: Path,
    *,
    batch_size: int,
    max_tokens: int = 0,
    finepdf_core_only: bool = False,
) -> None:
    command = [
        sys.executable,
        str(SCRIPT),
        "--tokenizer",
        str(TOKENIZER),
        "--dataset",
        "local-test",
        "--revision",
        "pinned-test-revision",
        "--input-files",
        str(source),
        "--text-col",
        "text",
        "--out-dir",
        str(output),
        "--min-chars",
        "8",
        "--exact-dedup",
        "--max-line-repeat-fraction",
        "0.80",
        "--required-min-number",
        "int_score=4",
        "--required-min-number",
        "metadata.reasoning_depth.primary.code=3",
        "--required-allowed-value",
        "metadata.document_type.primary.code=3",
        "--required-allowed-value",
        "metadata.document_type.primary.code=8",
        "--domain-field",
        "url",
        "--max-tokens-per-domain",
        "1000",
        "--tokenizer-batch-size",
        str(batch_size),
    ]
    if max_tokens:
        command.extend(("--max-tokens", str(max_tokens)))
    if finepdf_core_only:
        command.extend(
            (
                "--config",
                "eng_Latn",
                "--document-policy",
                "finepdf_core_v1",
                "--document-policy-allowed-tier",
                "core",
            )
        )
        command[command.index("local-test")] = "HuggingFaceFW/finepdfs-edu"
    environment = dict(os.environ)
    environment["TOKENIZERS_PARALLELISM"] = "true"
    subprocess.run(command, check=True, cwd=ROOT, env=environment)


def _artifact_bytes(path: Path) -> dict[str, bytes]:
    result = {}
    for file_path in sorted(path.iterdir()):
        if file_path.name == "manifest.json":
            manifest = json.loads(file_path.read_text())
            manifest["tokenizer"]["batch_size"] = "normalized"
            manifest["payload_sha256"] = "normalized"
            result[file_path.name] = json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        else:
            result[file_path.name] = file_path.read_bytes()
    return result


def _write_rows(path: Path) -> None:
    repeated = ("same repeat line\n" * 6) + "one distinct line"
    rows = [
        {
            "text": "tiny",
            "int_score": 5,
            "metadata": {
                "reasoning_depth": {"primary": {"code": 4}},
                "document_type": {"primary": {"code": 3}},
            },
            "url": "https://short.example",
        },
        {
            "text": FIRST_TEXT,
            "int_score": 5,
            "metadata": {
                "reasoning_depth": {"primary": {"code": 4}},
                "document_type": {"primary": {"code": 3}},
            },
            "url": "https://one.example/path",
        },
        {
            "text": FIRST_TEXT,
            "int_score": 5,
            "metadata": {
                "reasoning_depth": {"primary": {"code": 4}},
                "document_type": {"primary": {"code": 8}},
            },
            "url": "https://two.example/path",
        },
        {
            "text": repeated,
            "int_score": 5,
            "metadata": {
                "reasoning_depth": {"primary": {"code": 4}},
                "document_type": {"primary": {"code": 3}},
            },
            "url": "https://repeat.example",
        },
        {
            "text": (
                "A low score record must never enter the corpus.\n"
                "Its second line makes the quality gate the deciding filter.\n"
                "A third distinct line prevents a repetition rejection."
            ),
            "int_score": 3,
            "metadata": {
                "reasoning_depth": {"primary": {"code": 4}},
                "document_type": {"primary": {"code": 8}},
            },
            "url": "https://low.example",
        },
        {
            "text": SECOND_TEXT,
            "int_score": 5,
            "metadata": {
                "reasoning_depth": {"primary": {"code": 2}},
                "document_type": {"primary": {"code": 3}},
            },
            "url": "https://one.example/second",
        },
        {
            "text": THIRD_TEXT,
            "int_score": 5,
            "metadata": {
                "reasoning_depth": {"primary": {"code": 5}},
                "document_type": {"primary": {"code": 8}},
            },
            "url": "https://three.example/path",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_batched_tokenization_matches_single_record_artifacts(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_rows(source)
    sequential = tmp_path / "sequential"
    batched = tmp_path / "batched"

    _run(source, sequential, batch_size=1)
    _run(source, batched, batch_size=8)

    assert _artifact_bytes(sequential) == _artifact_bytes(batched)
    manifest = json.loads((sequential / "manifest.json").read_text())
    assert manifest["kept"] == 2
    assert manifest["tokens"] > 0
    assert manifest["dropped_short"] == 1
    assert manifest["dropped_duplicate"] == 1
    assert manifest["dropped_repetition"] == 1
    assert manifest["dropped_quality"] == 2
    assert manifest["filters"]["required_minimum_numbers"] == {
        "int_score": 4.0,
        "metadata.reasoning_depth.primary.code": 3.0,
    }
    assert manifest["filters"]["required_allowed_values"] == {
        "metadata.document_type.primary.code": ["3", "8"],
    }


def test_batched_tokenization_preserves_early_max_token_stop(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_rows(source)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    first_tokens = len(tokenizer.encode(FIRST_TEXT).ids) + 1
    sequential = tmp_path / "sequential"
    batched = tmp_path / "batched"

    _run(source, sequential, batch_size=1, max_tokens=first_tokens)
    _run(source, batched, batch_size=8, max_tokens=first_tokens)

    assert _artifact_bytes(sequential) == _artifact_bytes(batched)


def test_finepdf_policy_is_batched_deterministic_and_manifest_bound(tmp_path):
    source = tmp_path / "finepdf.jsonl"
    rows = [
        {
            "text": "\n".join(
                (
                    "A rigorous technical discussion of computation and systems.",
                    "The method identifies assumptions and derives a conclusion.",
                    "A concrete example checks the result against the premises.",
                )
                * 20
            ),
            "fw_edu_scores": [2.8, 2.7],
            "url": "https://repository.example.edu/core",
            "int_score": 5,
            "metadata": {
                "reasoning_depth": {"primary": {"code": 4}},
                "document_type": {"primary": {"code": 3}},
            },
        },
        {
            "text": "\n".join(
                (
                    "A coherent specialized discussion introduces a local subject.",
                    "Its evidence is useful but lacks a formal research structure.",
                    "The concluding paragraph summarizes the narrow application.",
                )
                * 20
            ),
            "fw_edu_scores": [1.8],
            "url": "https://example.org/residual",
            "int_score": 5,
            "metadata": {
                "reasoning_depth": {"primary": {"code": 4}},
                "document_type": {"primary": {"code": 3}},
            },
        },
        {
            "text": "\n".join(
                (
                    "Weekly newsletter issue 14. Parent reminders.",
                    "The school calendar lists dates for the community.",
                    "Contact the office for the next monthly notice.",
                )
                * 20
            ),
            "fw_edu_scores": [3.5],
            "url": "https://school.example.edu/newsletter",
            "int_score": 5,
            "metadata": {
                "reasoning_depth": {"primary": {"code": 4}},
                "document_type": {"primary": {"code": 3}},
            },
        },
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    sequential = tmp_path / "finepdf-sequential"
    batched = tmp_path / "finepdf-batched"

    _run(source, sequential, batch_size=1, finepdf_core_only=True)
    _run(source, batched, batch_size=8, finepdf_core_only=True)

    assert _artifact_bytes(sequential) == _artifact_bytes(batched)
    manifest = json.loads((sequential / "manifest.json").read_text())
    assert manifest["kept"] == 1
    assert manifest["dropped_document_policy"] == 2
    assert manifest["filters"]["document_policy"]["allowed_tiers"] == ["core"]
    assert manifest["filters"]["document_policy"]["seen_tiers"] == {
        "core": 1,
        "reject": 1,
        "residual": 1,
    }
    assert manifest["filters"]["document_policy"]["retained_tiers"] == {
        "core": 1
    }
    assert len(
        manifest["filters"]["document_policy"]["source"]["sha256"]
    ) == 64
