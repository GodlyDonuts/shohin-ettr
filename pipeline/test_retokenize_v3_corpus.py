import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
import zstandard as zstd

from pipeline.retokenize_v3_corpus import retokenize_corpus
from pipeline.tokenize_shards import (
    DOCUMENT_LEDGER_NAME,
    DOCUMENT_LEDGER_SCHEMA,
    DocumentLedgerWriter,
    canonical_payload_sha256,
    file_receipt,
    sha256_file,
)
from pipeline.verify_tokenized_shards import verify_manifest


def _tokenizer(path: Path, vocabulary: dict[str, int]) -> Path:
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path))
    return path


def _source_corpus(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_code = tmp_path / "source_code.py"
    source_code.write_text("# fixture\n", encoding="ascii")
    tokenizer_path = _tokenizer(
        tmp_path / "source_tokenizer.json",
        {"[UNK]": 0, "[EOS]": 1, "alpha": 2, "beta": 3, "gamma": 4},
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    texts = ["alpha beta", "gamma alpha"]
    token_documents = [
        [*tokenizer.encode(text, add_special_tokens=False).ids, 1]
        for text in texts
    ]
    all_ids = [token for document in token_documents for token in document]
    payload = np.asarray(all_ids, dtype="<u2").tobytes()
    compressed = zstd.ZstdCompressor(level=3).compress(payload)
    shard_path = source_dir / "shard_00000.u16.zst"
    shard_path.write_bytes(compressed)
    ledger = DocumentLedgerWriter(source_dir / DOCUMENT_LEDGER_NAME)
    offset = 0
    for index, (text, ids) in enumerate(zip(texts, token_documents)):
        document_payload = np.asarray(ids, dtype="<u2").tobytes()
        ledger.write(
            {
                "schema": DOCUMENT_LEDGER_SCHEMA,
                "source_row_index": index,
                "stable_identity_sha256": hashlib.sha256(
                    f"identity-{index}".encode("ascii")
                ).hexdigest(),
                "document_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "domain": "fixture.example",
                "allowed_value": None,
                "chars": len(text),
                "tokens": len(ids),
                "shard": shard_path.name,
                "token_start": offset,
                "token_end": offset + len(ids),
                "token_sha256": hashlib.sha256(document_payload).hexdigest(),
            }
        )
        offset += len(ids)
    ledger_receipt = ledger.close()
    raw_source = tmp_path / "raw.jsonl"
    raw_source.write_text("{}\n", encoding="ascii")
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text("{}\n", encoding="ascii")
    manifest = {
        "schema": "shohin-tokenized-shards-v3",
        "dataset": "fixture",
        "config": None,
        "split": "train",
        "requested_revision": "fixture",
        "resolved_revision": "fixture",
        "local_input_format": "json",
        "source_files": [file_receipt(raw_source)],
        "selection_code_sha256": sha256_file(source_code),
        "tokenizer": {
            **file_receipt(tokenizer_path),
            "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
            "eos_token": "[EOS]",
            "eos_id": 1,
            "batch_size": 2,
        },
        "tokens": len(all_ids),
        "shards": 1,
        "shard_files": [
            {
                "path": shard_path.name,
                "bytes": len(compressed),
                "tokens": len(all_ids),
                "sha256": hashlib.sha256(compressed).hexdigest(),
            }
        ],
        "document_ledger": ledger_receipt,
        "seen": 2,
        "kept": 2,
        "decontamination": {
            "pickle_path": None,
            "eval_files": [file_receipt(eval_file)],
        },
        "filters": {"exact_dedup": True, "document_policy": None},
    }
    manifest["payload_sha256"] = canonical_payload_sha256(manifest)
    (source_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return source_dir, source_code, tokenizer_path


def test_retokenization_preserves_documents_and_verifies_output(tmp_path: Path):
    source_dir, source_code, _source_tokenizer = _source_corpus(tmp_path)
    target_tokenizer = _tokenizer(
        tmp_path / "target_tokenizer.json",
        {"[UNK]": 0, "[EOS]": 1, "alpha": 2, "beta": 3, "gamma": 4},
    )
    output_dir = tmp_path / "output"
    selection_code = Path(__file__).with_name("retokenize_v3_corpus.py")
    report = retokenize_corpus(
        source_dir=source_dir,
        source_selection_code=source_code,
        target_tokenizer_path=target_tokenizer,
        target_eos_token="[EOS]",
        selection_code=selection_code,
        output_dir=output_dir,
        shard_tokens=3,
        batch_size=2,
    )
    assert report["documents"] == 2
    assert report["source_tokens"] == report["target_tokens"] == 6
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["retokenization"]["all_source_text_sha256_verified"] is True
    assert manifest["retokenization"]["all_target_double_encodings_verified"] is True
    assert manifest["retokenization"]["target_decoder_inversion_required"] is False
    assert manifest["retokenization"]["contains_document_text"] is False
    assert manifest["tokenizer"]["sha256"] == sha256_file(target_tokenizer)
    verification = verify_manifest(
        output_dir,
        selection_code=selection_code,
        require_external_inputs=True,
    )
    assert verification["document_rows"] == 2
