from pathlib import Path

from merge_vfr1_trace_shards import _request_identities


def test_request_identity_order_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "requests.jsonl"
    path.write_text(
        '{"schema":"shohin-vfr1-teacher-request-v1","identity_sha256":"'
        + "a" * 64
        + '"}\n'
        + '{"schema":"shohin-vfr1-teacher-request-v1","identity_sha256":"'
        + "b" * 64
        + '"}\n',
        encoding="utf-8",
    )
    assert _request_identities(path) == ["a" * 64, "b" * 64]
