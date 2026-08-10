import hashlib
import json

from merge_dtmc1_drafts import load_jsonl, sha256_file


def test_load_jsonl_preserves_physical_unicode_separator(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [{"identity_sha256": "a" * 64, "draft": "x\u2028y"}]
    path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    assert load_jsonl(path) == rows
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()
