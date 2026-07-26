from __future__ import annotations

from ettr_il_v2_surface import call, integer, symbol
from audit_ettr_il_v2_surface_capacity import (
    WIDTHS,
    bounded_node_report,
    bounded_presentation_report,
    count_nodes,
)
from ettr_il_v2_token_native_surface import (
    DEFAULT_TOKENIZER_PATH,
    TokenNativeSurfaceCodec,
)


def test_node_counter_is_exact() -> None:
    result = count_nodes(
        call(
            14,
            integer(2),
            call(4, symbol("x0000000000000001")),
        )
    )
    assert result.as_dict() == {
        "calls": 2,
        "integers": 1,
        "symbols": 1,
        "total": 4,
    }


def test_full_bounded_factor_universe_fits_at_one_token_per_node() -> None:
    report = bounded_node_report()
    assert report["horn_world"]["maximum"]["total"] == 138
    assert report["rewrite_world"]["maximum"]["total"] == 114
    assert report["resource_world"]["maximum"]["total"] == 125
    assert report["horn_command"]["maximum"]["total"] == 87
    assert report["rewrite_command"]["maximum"]["total"] == 81
    assert report["resource_command"]["maximum"]["total"] == 60
    assert all(
        entry["maximum"]["total"]
        <= WIDTHS[
            "world" if name.endswith("_world") else "command"
        ]
        for name, entry in report.items()
    )


def test_all_six_presentations_fit_after_lossless_transport_fusion() -> None:
    report = bounded_presentation_report(
        TokenNativeSurfaceCodec(DEFAULT_TOKENIZER_PATH)
    )
    assert set(report) == {
        "horn_world",
        "rewrite_world",
        "resource_world",
        "horn_command",
        "rewrite_command",
        "resource_command",
    }
    assert all(
        presentation["within_width"]
        and presentation["maximum_token_count"] <= factor["width"]
        for factor in report.values()
        for presentation in factor["presentations"].values()
    )
    assert (
        report["horn_world"]["presentations"]["relation_reification"][
            "maximum_ast_nodes"
        ]
        > report["horn_world"]["width"]
    )
