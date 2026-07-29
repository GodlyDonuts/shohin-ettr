import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
import zstandard as zstd

from pipeline.audit_v3_sensitive_content import (
    SensitiveContentAuditError,
    audit_sensitive_content,
    classify_sensitive_text,
)
from pipeline.tokenize_shards import (
    DOCUMENT_LEDGER_NAME,
    DOCUMENT_LEDGER_SCHEMA,
    canonical_payload_sha256,
    sha256_file,
)


SELECTION_CODE = Path(__file__).with_name("audit_v3_sensitive_content.py")
JOB = (
    Path(__file__).parent / "jobs" / "audit_v3_sensitive_content.sbatch"
).read_text()


def _corpus(tmp_path: Path) -> tuple[Path, Path]:
    texts = (
        "A normal paper. Correspondence: author@example.org.",
        'Configuration: api_key = "AbcdEFGH12345678ijklMNOP".',
        "Temporary cloud identifier AKIAIOSFODNN7EXAMPLE appears here.",
    )
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.train_from_iterator(
        texts,
        BpeTrainer(
            vocab_size=320,
            initial_alphabet=ByteLevel.alphabet(),
            special_tokens=["[UNK]"],
        ),
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    source_selection = tmp_path / "source_selection.py"
    source_selection.write_text("print('source selection')\n")
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text('{"question":"held out"}\n')
    eval_pickle = tmp_path / "evalgrams.pkl"
    eval_pickle.write_bytes(b"evalgrams")
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    payload = bytearray()
    rows = []
    offset = 0
    for index, text in enumerate(texts):
        token_ids = tokenizer.encode(text).ids
        encoded = b"".join(
            int(token).to_bytes(2, "little") for token in token_ids
        )
        payload.extend(encoded)
        rows.append(
            {
                "allowed_value": "CCBY",
                "chars": len(text),
                "document_sha256": hashlib.sha256(
                    text.encode()
                ).hexdigest(),
                "domain": f"domain-{index}.example",
                "schema": DOCUMENT_LEDGER_SCHEMA,
                "shard": "shard_00000.u16.zst",
                "source_row_index": index,
                "stable_identity_sha256": hashlib.sha256(
                    f"identity-{index}".encode()
                ).hexdigest(),
                "token_end": offset + len(token_ids),
                "token_sha256": hashlib.sha256(encoded).hexdigest(),
                "token_start": offset,
                "tokens": len(token_ids),
            }
        )
        offset += len(token_ids)

    shard = corpus / "shard_00000.u16.zst"
    shard.write_bytes(zstd.ZstdCompressor(level=3).compress(bytes(payload)))
    ledger_payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("ascii")
    ledger = corpus / DOCUMENT_LEDGER_NAME
    ledger.write_bytes(
        zstd.ZstdCompressor(level=3).compress(ledger_payload)
    )
    manifest = {
        "schema": "shohin-tokenized-shards-v3",
        "selection_code_sha256": sha256_file(source_selection),
        "source_files": [],
        "tokenizer": {
            "path": str(tokenizer_path),
            "bytes": tokenizer_path.stat().st_size,
            "sha256": sha256_file(tokenizer_path),
        },
        "tokens": offset,
        "shards": 1,
        "kept": len(rows),
        "filters": {"exact_dedup": True},
        "shard_files": [
            {
                "path": shard.name,
                "bytes": shard.stat().st_size,
                "tokens": offset,
                "sha256": sha256_file(shard),
            }
        ],
        "document_ledger": {
            "path": DOCUMENT_LEDGER_NAME,
            "bytes": ledger.stat().st_size,
            "sha256": sha256_file(ledger),
            "rows": len(rows),
            "tokens": offset,
            "contains_document_text": False,
            "schema": DOCUMENT_LEDGER_SCHEMA,
        },
        "decontamination": {
            "pickle_path": str(eval_pickle),
            "pickle_bytes": eval_pickle.stat().st_size,
            "pickle_sha256": sha256_file(eval_pickle),
            "eval_files": [
                {
                    "path": str(eval_file),
                    "bytes": eval_file.stat().st_size,
                    "sha256": sha256_file(eval_file),
                }
            ],
        },
    }
    manifest["payload_sha256"] = canonical_payload_sha256(manifest)
    (corpus / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return corpus, source_selection


def test_sensitive_audit_separates_contact_flags_from_credentials(
    tmp_path: Path,
) -> None:
    corpus, source_selection = _corpus(tmp_path)
    output = tmp_path / "audit"
    report = audit_sensitive_content(
        corpus_dir=corpus,
        selection_code=source_selection,
        output_dir=output,
    )
    summary = report["summary"]
    assert report["status"] == "pass"
    assert summary["flagged_documents"] == 3
    assert summary["automatic_exclusion_documents"] == 2
    assert summary["category_documents"] == {
        "aws_access_key": 1,
        "credential_assignment": 1,
        "email": 1,
    }
    with (output / "sensitive_findings.jsonl.zst").open("rb") as source:
        with zstd.ZstdDecompressor().stream_reader(source) as reader:
            findings = [
                json.loads(line)
                for line in io.TextIOWrapper(reader, encoding="ascii")
            ]
    assert len(findings) == 3
    assert sum(
        bool(row["automatic_exclusion_categories"]) for row in findings
    ) == 2
    assert all("text" not in json.dumps(row).lower() for row in findings)


def test_placeholder_credentials_do_not_trigger_automatic_exclusion() -> None:
    assert classify_sensitive_text(
        "api_key = your_api_key_placeholder"
    ) == {}
    assert classify_sensitive_text(
        'api_key = "AbcdEFGH12345678ijklMNOP"'
    ) == {"credential_assignment": 1}


def test_source_substitution_fails_before_output(tmp_path: Path) -> None:
    corpus, source_selection = _corpus(tmp_path)
    (corpus / "shard_00000.u16.zst").write_bytes(b"substituted")
    output = tmp_path / "audit"
    with pytest.raises(Exception, match="byte count|SHA-256"):
        audit_sensitive_content(
            corpus_dir=corpus,
            selection_code=source_selection,
            output_dir=output,
        )
    assert not output.exists()


def test_bad_audit_arguments_fail_closed(tmp_path: Path) -> None:
    corpus, source_selection = _corpus(tmp_path)
    output = tmp_path / "audit"
    output.mkdir()
    with pytest.raises(SensitiveContentAuditError, match="arguments"):
        audit_sensitive_content(
            corpus_dir=corpus,
            selection_code=source_selection,
            output_dir=output,
        )


def test_cli_emits_compact_receipt_not_full_verification(tmp_path: Path) -> None:
    corpus, source_selection = _corpus(tmp_path)
    output = tmp_path / "audit"
    result = subprocess.run(
        [
            sys.executable,
            str(SELECTION_CODE),
            "--corpus-dir",
            str(corpus),
            "--selection-code",
            str(source_selection),
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "pass"
    assert receipt["corpus"]["documents"] == 3
    assert receipt["summary"]["automatic_exclusion_documents"] == 2
    assert "verification" not in result.stdout
    assert "source_files" not in result.stdout
    assert len(result.stdout) < 2_000


def test_sensitive_audit_job_is_hash_bound_and_cpu_only() -> None:
    assert "SHA256SUMS" in JOB
    assert "--corpus-dir" in JOB
    assert "--selection-code" in JOB
    assert "--output-dir" in JOB
    assert "--gres" not in JOB
    assert "CUDA" not in JOB
