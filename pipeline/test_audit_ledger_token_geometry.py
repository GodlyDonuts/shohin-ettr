from audit_ledger_token_geometry import quantiles


def test_quantiles_are_nearest_rank_and_monotonic():
    result = quantiles(list(range(1, 1001)))
    assert result == {
        "p50": 500,
        "p90": 900,
        "p95": 950,
        "p99": 990,
        "p999": 999,
        "max": 1000,
    }


def test_quantiles_singleton():
    assert set(quantiles([17]).values()) == {17}
