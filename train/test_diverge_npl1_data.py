#!/usr/bin/env python3
"""Contract tests for conditional NPL1 natural development surfaces."""

from __future__ import annotations

from diverge_npl1_data import (
    DEVELOPMENT_SEED,
    episode_names,
    natural_public_record,
    parse_program_surface,
    render_feedback,
    validate_natural_public_record,
)
from diverge_pl1_data import build_split


def main() -> None:
    episode = build_split(
        split="npl1_development", seed=DEVELOPMENT_SEED, count=1
    )[0]
    record = natural_public_record(episode)
    validate_natural_public_record(record)
    _, registers = episode_names(episode)
    for surface, program in zip(
        (*record["acquisition"], *record["transfer"]),
        (*episode.acquisition, *episode.transfer),
        strict=True,
    ):
        initial, symbols = parse_program_surface(surface, episode.aliases, registers)
        assert initial == program.initial_state
        assert symbols == program.symbols
    for plan in record["feedback_plan"]:
        text = render_feedback(plan, certificate_code=7)
        assert str(plan["target_branch"]) in text
        assert str(plan["distractor_branch"]) in text
    assert "symbol_to_operation" not in record
    assert all("symbols" not in value for value in record["acquisition"])
    assert all("terminal_state" not in value for value in record["transfer"])
    print("DIVERGE-NPL1 data tests passed")


if __name__ == "__main__":
    main()
