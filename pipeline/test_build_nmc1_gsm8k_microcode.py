import json

from build_nmc1_gsm8k_microcode import _rows


def test_physical_jsonl_reader_preserves_unicode_line_separator(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    expected = {"question": "left\u2028right", "answer": "x"}
    path.write_text(json.dumps(expected, ensure_ascii=False) + "\n")
    assert _rows(path) == [expected]
