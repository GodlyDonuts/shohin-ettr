from collections import Counter

import pytest

from audit_ettr_public_opcode_identifiability import (
    _CALL_STRIDE,
    _FRAME_A,
    _FRAME_B,
    _INTEGER_BASE,
    _REIFY_END,
    _conditional_summary,
    _train_to_development,
    canonical_public_tree,
    parse_public_transport,
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
    physical = (*preamble, *_render(TREE, prefix=prefix, reverse=reverse), 0, 1, 2)
    parsed = parse_public_transport(physical, codebook_size=CODEBOOK_SIZE)
    assert canonical_public_tree(parsed, mode="alpha_exact") == canonical_public_tree(
        TREE, mode="alpha_exact"
    )


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
            _call(1, (("symbol", IDENTIFIER_A), ("symbol", IDENTIFIER_B))),
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
