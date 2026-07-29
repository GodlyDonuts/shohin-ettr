import io
import json
from pathlib import Path

import pytest
import zstandard as zstd

from pipeline.audit_cross_source_exact_dedup import CorpusSpec
from pipeline.audit_cross_source_near_dedup import audit_near_duplicates
from pipeline.materialize_cross_source_near_residual import (
    NearResidualError,
    materialize_near_residual,
)
from pipeline.test_audit_cross_source_near_dedup import (
    _fixture,
)
from pipeline.tokenize_shards import DOCUMENT_LEDGER_NAME
from pipeline.verify_tokenized_shards import verify_manifest


SELECTION_CODE = Path(__file__).with_name(
    "materialize_cross_source_near_residual.py"
)


def _prepare(tmp_path: Path) -> tuple[Path, Path, Path]:
    first, second, source_selection = _fixture(tmp_path)
    near = tmp_path / "near"
    audit_near_duplicates(
        (
            CorpusSpec("incumbent", first),
            CorpusSpec("challenger", second),
        ),
        selection_code=source_selection,
        output_dir=near,
        minimum_tokens=16,
        batch_documents=1,
        require_external_inputs=True,
    )
    return second, near, source_selection


def test_near_removal_is_applied_and_residual_reverifies(tmp_path):
    source, near, source_selection = _prepare(tmp_path)
    output = tmp_path / "residual"
    result = materialize_near_residual(
        source_dir=source,
        near_dir=near,
        corpus_name="challenger",
        source_selection_code=source_selection,
        selection_code=SELECTION_CODE,
        output_dir=output,
        shard_tokens=32,
    )
    assert result["documents"] == 1
    assert result["tokens"] == 90
    assert result["dropped_documents"] == 1
    assert result["dropped_tokens"] == 128
    verification = verify_manifest(
        output,
        selection_code=SELECTION_CODE,
        require_external_inputs=True,
    )
    assert verification["document_rows"] == 1
    with (output / DOCUMENT_LEDGER_NAME).open("rb") as source_file:
        with zstd.ZstdDecompressor().stream_reader(source_file) as reader:
            row = json.loads(
                io.TextIOWrapper(reader, encoding="ascii").read()
            )
    assert row["stable_identity_sha256"] == "4" * 64
    assert row["token_start"] == 0
    assert row["token_end"] == 90


def test_tampered_near_removal_fails_without_output(tmp_path):
    source, near, source_selection = _prepare(tmp_path)
    removal_path = near / "near_duplicate_removals.jsonl.zst"
    removal_path.write_bytes(b"tampered")
    output = tmp_path / "residual"
    with pytest.raises(NearResidualError, match="removal artifact differs"):
        materialize_near_residual(
            source_dir=source,
            near_dir=near,
            corpus_name="challenger",
            source_selection_code=source_selection,
            selection_code=SELECTION_CODE,
            output_dir=output,
        )
    assert not output.exists()


def test_wrong_source_selection_code_fails_closed(tmp_path):
    source, near, _source_selection = _prepare(tmp_path)
    wrong = tmp_path / "wrong.py"
    wrong.write_text("print('wrong')\n")
    with pytest.raises(Exception, match="selection code"):
        materialize_near_residual(
            source_dir=source,
            near_dir=near,
            corpus_name="challenger",
            source_selection_code=wrong,
            selection_code=SELECTION_CODE,
            output_dir=tmp_path / "residual",
        )
