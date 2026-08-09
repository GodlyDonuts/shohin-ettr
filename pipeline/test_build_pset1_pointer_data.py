from build_pset1_pointer_data import exact_surface_tokens, extract_source, token_span


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


def test_token_span_requires_exact_boundaries() -> None:
    offsets = [[0, 1], [1, 3], [3, 4]]
    assert token_span(offsets, 1, 3) == (1, 1)
    assert token_span(offsets, 2, 3) is None


def test_surface_tokens_round_trip() -> None:
    assert exact_surface_tokens(Tokenizer(), "42", 2) == [52, 50]
    assert exact_surface_tokens(Tokenizer(), "42", 1) is None
