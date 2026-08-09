from build_vfr1_capability_data import format_trace, shuffled_faults


def test_format_trace_round_trips_parser() -> None:
    from hf_vfr1_generate_traces import parse_trace

    rendered = format_trace("Wrong sign.", "Recompute. \\boxed{2}")
    assert parse_trace(rendered) == ("Wrong sign.", "Recompute. \\boxed{2}")


def test_fault_shuffle_preserves_multiset_and_changes_owners() -> None:
    faults = {
        f"{index:064x}": ("math500", f"fault {index}") for index in range(5)
    }
    shuffled = shuffled_faults(faults, block_size=3)
    assert sorted(shuffled.values()) == sorted(fault for _, fault in faults.values())
    assert all(shuffled[identity] != fault for identity, (_, fault) in faults.items())
