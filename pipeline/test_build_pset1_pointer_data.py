from build_pset1_pointer_data import character_coverage, exact_surface_bytes, extract_source


class Tokenizer:
    def encode(self, value, add_special_tokens=False):
        return [ord(character) for character in value]

    def decode(self, ids, **kwargs):
        return "".join(chr(value) for value in ids)


def test_extract_source_from_revision_prompt() -> None:
    question = (
        "prefix Original problem:\n2+2?\n\nInternal draft:\nwrong\n\n"
        "Original problem:\n2+2? suffix 2+2?"
    )
    assert extract_source(question) == "2+2?"


def test_character_coverage_requires_exact_partition() -> None:
    offsets = [[0, 1], [1, 3], [3, 4]]
    assert character_coverage(offsets, 4)
    assert not character_coverage([[0, 2], [1, 4]], 4)


def test_surface_bytes_round_trip() -> None:
    assert exact_surface_bytes("42", 2) == [52, 50]
    assert exact_surface_bytes("42", 1) is None
