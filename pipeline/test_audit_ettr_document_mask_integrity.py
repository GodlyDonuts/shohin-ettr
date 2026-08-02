from types import SimpleNamespace

from audit_ettr_document_mask_integrity import legacy_public_document_end
from audit_ettr_public_opcode_identifiability import (
    _CALL_STRIDE,
    _FRAME_B,
    _INTEGER_BASE,
    _REIFY_END,
    _deterministic_cover,
    public_document_indices,
)


CODEBOOK_SIZE = _REIFY_END + 256


class _FakeCodec:
    def __init__(self):
        self.codebook = SimpleNamespace(token_ids=tuple(range(CODEBOOK_SIZE)))

    @staticmethod
    def _render_logical(indices):
        return (" " + " ".join(str(value) for value in indices)).encode("ascii")

    @staticmethod
    def _payload_indices(payload):
        return tuple(int(value) for value in payload.decode("ascii").split())


def _call(head, children):
    return ("call", head, tuple(children))


def _render_postfix(node, *, reverse):
    if node[0] == "integer":
        return [_INTEGER_BASE + node[1]]
    children = list(node[2])
    if reverse:
        children.reverse()
    body = [
        value
        for child in children
        for value in _render_postfix(child, reverse=reverse)
    ]
    return [*body, node[1] * _CALL_STRIDE + len(children)]


def test_legacy_mask_truncates_a_nested_reverse_postfix_root():
    codec = _FakeCodec()
    inner = _call(
        14,
        (
            ("integer", 3),
            _call(1, (("integer", 0), ("integer", 1))),
            _call(7, (("integer", 1),)),
        ),
    )
    tree = _call(15, (("integer", 0), inner))
    document = (_FRAME_B, _FRAME_B, *_render_postfix(tree, reverse=True))
    payload = codec._render_logical(document)
    width = len(document) + 64
    cover = _deterministic_cover(
        payload,
        width=width,
        count=width - len(document),
        codebook_size=CODEBOOK_SIZE,
    )
    physical = (*document, *cover)
    source = codec._render_logical(physical).decode("ascii")

    legacy = legacy_public_document_end(physical, codebook_size=CODEBOOK_SIZE)
    exact = len(public_document_indices(codec, source))

    assert legacy < exact
    assert exact == len(document)
