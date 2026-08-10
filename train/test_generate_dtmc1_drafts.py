import json

import pytest

from generate_dtmc1_drafts import DTMC1DraftError, load_rows


def test_load_rows_uses_physical_jsonl_lines(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [
        {
            "identity_sha256": f"{index:064x}",
            "original_question": "line separator \u2028 remains inside JSON",
        }
        for index in range(6333)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(load_rows(path, digest)) == 6333
    with pytest.raises(DTMC1DraftError, match="SHA-256"):
        load_rows(path, "0" * 64)
