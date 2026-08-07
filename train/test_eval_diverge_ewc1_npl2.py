from __future__ import annotations

from eval_diverge_ewc1_npl2 import build_typed_cache


def test_build_typed_cache_preserves_program_and_episode_order():
    public = [
        {
            "episode_id": "episode-a",
            "branch_names": [f"branch{chr(97 + index)}" for index in range(8)],
            "acquisition": [{"source_text": "a"}, {"source_text": "b"}],
            "transfer": [{"source_text": "c"}],
        }
    ]
    predictions = [((1, 2), (0, 1)), ((3, 4), (2,)), ((5, 6), (3, 4, 5))]
    cache = build_typed_cache(public, predictions)
    episode = cache["episode-a"]
    assert [program.initial_state for program in episode.acquisition] == [(1, 2), (3, 4)]
    assert episode.transfer[0].symbols == (3, 4, 5)
