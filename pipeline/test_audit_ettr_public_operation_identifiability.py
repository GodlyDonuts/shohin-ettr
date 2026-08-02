from __future__ import annotations

from audit_ettr_public_operation_identifiability import resolved_operations


def _call(head: int, *children: tuple[object, ...]) -> tuple[object, ...]:
    return ("call", head, tuple(children))


def test_resolved_operations_replace_opaque_names_with_declarations() -> None:
    operator = ("symbol", 701)
    object_symbol = ("symbol", 702)
    root = _call(
        14,
        ("integer", 2),
        _call(
            1,
            _call(3, operator, ("integer", 4), _call(0)),
            _call(3, object_symbol, ("integer", 7), _call(0)),
        ),
        _call(13, _call(4, operator, object_symbol)),
    )

    assert resolved_operations(root) == (
        [
            "call",
            4,
            [
                ["declared-symbol", 4, ["call", 0, []]],
                ["declared-symbol", 7, ["call", 0, []]],
            ],
        ],
    )


def test_resolved_operations_accept_outer_transport_wrapper() -> None:
    operator = ("symbol", 703)
    semantic = _call(
        14,
        ("integer", 2),
        _call(1, _call(3, operator, ("integer", 1), _call(0))),
        _call(13, _call(4, operator), _call(4, operator)),
    )

    assert len(resolved_operations(_call(15, ("integer", 0), semantic))) == 2


def test_resolved_operations_accept_direct_local_command() -> None:
    semantic = _call(
        13,
        _call(4, ("integer", 2), ("integer", 7), ("integer", 0)),
        _call(4, ("integer", 1), ("integer", 3), ("integer", 1)),
    )

    assert resolved_operations(_call(15, ("integer", 0), semantic)) == (
        [
            "call",
            4,
            [["integer", 2], ["integer", 7], ["integer", 0]],
        ],
        [
            "call",
            4,
            [["integer", 1], ["integer", 3], ["integer", 1]],
        ],
    )
