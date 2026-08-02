from collections import Counter
from types import SimpleNamespace

import pytest

import audit_ettr_public_opcode_identifiability as audit
from audit_ettr_public_opcode_identifiability import (
    _CALL_STRIDE,
    _FRAME_A,
    _FRAME_B,
    _INTEGER_BASE,
    _REIFY_BASE,
    _REIFY_END,
    _conditional_summary,
    _deterministic_cover,
    _train_to_development,
    canonical_public_tree,
    parse_public_transport,
    public_document_indices,
)


CODEBOOK_SIZE = _REIFY_END + 256
IDENTIFIER_A = CODEBOOK_SIZE - 2
IDENTIFIER_B = CODEBOOK_SIZE - 1


def _call(head, children):
    return ("call", head, tuple(children))


TREE = _call(
    14,
    (
        ("symbol", IDENTIFIER_A),
        ("integer", 5),
        _call(1, (("symbol", IDENTIFIER_A), ("symbol", IDENTIFIER_B))),
    ),
)


def _render(node, *, prefix, reverse):
    kind = node[0]
    if kind == "symbol":
        return [node[1]]
    if kind == "integer":
        return [_INTEGER_BASE + node[1]]
    if kind == "reify":
        descendants = [node[1], *node[2]]
        if reverse:
            descendants[1:] = reversed(descendants[1:])
        code = _REIFY_BASE + len(node[2])
        body = [
            item
            for child in descendants
            for item in _render(child, prefix=prefix, reverse=reverse)
        ]
        return [code, *body] if prefix else [*body, code]
    assert kind == "call"
    children = list(node[2])
    if reverse:
        children.reverse()
    head = node[1] * _CALL_STRIDE + len(children)
    body = [item for child in children for item in _render(child, prefix=prefix, reverse=reverse)]
    return [head, *body] if prefix else [*body, head]


@pytest.mark.parametrize(
    ("prefix", "reverse", "preamble"),
    (
        (True, False, (_FRAME_A, _FRAME_A)),
        (True, True, (_FRAME_A, _FRAME_B)),
        (False, False, (_FRAME_B, _FRAME_A)),
        (False, True, (_FRAME_B, _FRAME_B)),
    ),
)
def test_public_parser_is_renderer_and_cover_invariant(prefix, reverse, preamble):
    physical = (*preamble, *_render(TREE, prefix=prefix, reverse=reverse))
    parsed = parse_public_transport(physical, codebook_size=CODEBOOK_SIZE)
    assert canonical_public_tree(parsed, mode="alpha_exact") == canonical_public_tree(
        TREE, mode="alpha_exact"
    )


@pytest.mark.parametrize(
    ("prefix", "reverse", "preamble"),
    (
        (True, False, (_FRAME_A, _FRAME_A)),
        (True, True, (_FRAME_A, _FRAME_B)),
        (False, False, (_FRAME_B, _FRAME_A)),
        (False, True, (_FRAME_B, _FRAME_B)),
    ),
)
def test_public_parser_preserves_relation_and_reverses_only_reified_endpoints(
    prefix,
    reverse,
    preamble,
):
    tree = _call(
        14,
        (
            ("symbol", IDENTIFIER_A),
            ("integer", 5),
            (
                "reify",
                ("symbol", IDENTIFIER_B),
                (("symbol", IDENTIFIER_A), ("integer", 7)),
            ),
        ),
    )
    physical = (*preamble, *_render(tree, prefix=prefix, reverse=reverse))

    parsed = parse_public_transport(physical, codebook_size=CODEBOOK_SIZE)

    assert canonical_public_tree(parsed, mode="alpha_exact") == canonical_public_tree(
        tree, mode="alpha_exact"
    )


def test_reverse_postfix_does_not_stop_at_nested_root_shaped_child():
    tree = _call(15, (("integer", 0), TREE))
    physical = (
        _FRAME_B,
        _FRAME_B,
        *_render(tree, prefix=False, reverse=True),
    )

    parsed = parse_public_transport(physical, codebook_size=CODEBOOK_SIZE)

    assert canonical_public_tree(parsed, mode="alpha_exact") == canonical_public_tree(
        tree, mode="alpha_exact"
    )


class _FakeCodec:
    def __init__(self):
        self.codebook = SimpleNamespace(token_ids=tuple(range(CODEBOOK_SIZE)))

    @staticmethod
    def _render_logical(indices):
        return (" " + " ".join(str(value) for value in indices)).encode("ascii")

    @staticmethod
    def _payload_indices(payload):
        return tuple(int(value) for value in payload.decode("ascii").split())


def test_public_document_boundary_uses_cover_not_first_root_candidate():
    codec = _FakeCodec()
    tree = _call(15, (("integer", 0), TREE))
    document = (
        _FRAME_B,
        _FRAME_B,
        *_render(tree, prefix=False, reverse=True),
    )
    document_payload = codec._render_logical(document)
    width = len(document) + 64
    cover = _deterministic_cover(
        document_payload,
        width=width,
        count=width - len(document),
        codebook_size=CODEBOOK_SIZE,
    )
    source = codec._render_logical((*document, *cover)).decode("ascii")

    recovered = public_document_indices(codec, source)

    assert recovered == document


def test_alpha_signature_ignores_identifier_codes_but_preserves_equality():
    renamed = _call(
        14,
        (
            ("symbol", IDENTIFIER_B),
            ("integer", 5),
            _call(1, (("symbol", IDENTIFIER_B), ("symbol", IDENTIFIER_A))),
        ),
    )
    broken = _call(
        14,
        (
            ("symbol", IDENTIFIER_B),
            ("integer", 5),
            _call(1, (("symbol", IDENTIFIER_B), ("symbol", IDENTIFIER_B))),
        ),
    )
    expected = canonical_public_tree(TREE, mode="alpha_exact")
    assert canonical_public_tree(renamed, mode="alpha_exact") == expected
    assert canonical_public_tree(broken, mode="alpha_exact") != expected


def test_signature_modes_form_explicit_abstraction_ladder():
    changed_integer = _call(
        14,
        (
            ("symbol", IDENTIFIER_A),
            ("integer", 19),
            _call(1, (("symbol", IDENTIFIER_A), ("symbol", IDENTIFIER_B))),
        ),
    )
    assert canonical_public_tree(TREE, mode="alpha_exact") != canonical_public_tree(
        changed_integer, mode="alpha_exact"
    )
    assert canonical_public_tree(TREE, mode="alpha_operator") == canonical_public_tree(
        changed_integer, mode="alpha_operator"
    )


def test_record_labels_use_all_four_corner_conditioned_sources(monkeypatch):
    views = tuple(
        SimpleNamespace(
            renderer=renderer,
            world_sources=tuple(f"world-{index}" for index in range(4)),
            command_sources=tuple(f"command-{index}" for index in range(4)),
        )
        for renderer in range(4)
    )
    record = SimpleNamespace(source_visible=SimpleNamespace(views=views))
    monkeypatch.setattr(
        audit,
        "public_source_signatures",
        lambda _codec, source: {mode: source for mode in audit._SIGNATURE_MODES},
    )
    monkeypatch.setattr(
        audit,
        "_record_programs",
        lambda _record: tuple({"opcode": f"program-{index}"} for index in range(4)),
    )

    labels = tuple(audit._record_labels(record, object()))

    assert len(labels) == 24
    for corner in range(4):
        for mode in audit._SIGNATURE_MODES:
            command_key = audit._digest(
                {"command": f"command-{corner}", "mode": mode}
            )
            joint_key = audit._digest(
                {
                    "command": f"command-{corner}",
                    "mode": mode,
                    "world": f"world-{corner}",
                }
            )
            assert (mode, "command", command_key, f"program-{corner}") in labels
            assert (
                mode,
                "world_command",
                joint_key,
                f"program-{corner}",
            ) in labels


def test_conditional_and_cross_split_statistics_are_exact():
    train = {
        "a": Counter({"x": 3, "y": 1}),
        "b": Counter({"z": 2}),
    }
    development = {
        "a": Counter({"x": 2, "y": 1}),
        "c": Counter({"q": 2}),
    }
    summary = _conditional_summary(train)
    assert summary["instances"] == 6
    assert summary["bayes_correct"] == 5
    assert summary["bayes_rate"] == pytest.approx(5 / 6)
    cross = _train_to_development(train, development)
    assert cross == {
        "accuracy_all": pytest.approx(2 / 5),
        "accuracy_seen": pytest.approx(2 / 3),
        "correct": 2,
        "development_instances": 5,
        "seen_instances": 3,
        "seen_rate": pytest.approx(3 / 5),
        "training_modal_ties": 0,
        "unseen_instances": 2,
    }
