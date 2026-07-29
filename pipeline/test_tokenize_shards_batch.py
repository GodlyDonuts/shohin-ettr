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


def _run(source: Path, output: Path, *, batch_size: int, max_tokens: int = 0) -> None:
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
        "--min-number-field",
        "int_score",
        "--min-number",
        "4",
        "--domain-field",
        "url",
        "--max-tokens-per-domain",
        "1000",
        "--tokenizer-batch-size",
        str(batch_size),
    ]
    if max_tokens:
        command.extend(("--max-tokens", str(max_tokens)))
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
        {"text": "tiny", "int_score": 5, "url": "https://short.example"},
        {
            "text": FIRST_TEXT,
            "int_score": 5,
            "url": "https://one.example/path",
        },
        {
            "text": FIRST_TEXT,
            "int_score": 5,
            "url": "https://two.example/path",
        },
        {"text": repeated, "int_score": 5, "url": "https://repeat.example"},
        {
            "text": (
                "A low score record must never enter the corpus.\n"
                "Its second line makes the quality gate the deciding filter.\n"
                "A third distinct line prevents a repetition rejection."
            ),
            "int_score": 3,
            "url": "https://low.example",
        },
        {
            "text": SECOND_TEXT,
            "int_score": 5,
            "url": "https://one.example/second",
        },
        {
            "text": THIRD_TEXT,
            "int_score": 5,
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
    assert manifest["kept"] == 3
    assert manifest["tokens"] > 0
    assert manifest["dropped_short"] == 1
    assert manifest["dropped_duplicate"] == 1
    assert manifest["dropped_repetition"] == 1
    assert manifest["dropped_quality"] == 1


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
