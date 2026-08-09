import pytest

from hf_vfr1_generate_traces import VFR1GenerationError, parse_trace, shard_bounds


def test_parse_trace_extracts_complete_blocks() -> None:
    fault, revision = parse_trace(
        "<FAULT>Arithmetic sign error.</FAULT>\n"
        "<REVISION>Recompute. \\boxed{2}</REVISION>"
    )
    assert fault == "Arithmetic sign error."
    assert revision.endswith("\\boxed{2}")


@pytest.mark.parametrize(
    "text",
    [
        "<FAULT>x</FAULT><REVISION>y",
        "<REVISION>y</REVISION><FAULT>x</FAULT>",
        "<FAULT>\\boxed{1}</FAULT><REVISION>y</REVISION>",
        "<FAULT>x</FAULT><REVISION>y</REVISION>trailing",
    ],
)
def test_parse_trace_rejects_ambiguous_or_leaky_outputs(text: str) -> None:
    with pytest.raises(VFR1GenerationError):
        parse_trace(text)


def test_shard_bounds_cover_without_overlap() -> None:
    bounds = [shard_bounds(11, index, 3) for index in range(3)]
    assert bounds == [(0, 3), (3, 7), (7, 11)]
