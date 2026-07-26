"""Exact deterministic bootstrap primitives for R12-ETTR-IL-v2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Iterable


PROTOCOL = "R12-ETTR-IL-v2"
BOOTSTRAP_REPLICATES = 100_000
MODEL_SEED_COUNT = 5
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_UINT64_SPACE = 1 << 64


class StatisticsError(ValueError):
    """The frozen statistical protocol cannot be evaluated exactly."""


def replicate_root(split_plaintext_sha256: str, replicate_index: int) -> bytes:
    if _HEX64.fullmatch(split_plaintext_sha256) is None:
        raise StatisticsError("split plaintext SHA-256 differs")
    if (
        type(replicate_index) is not int
        or not 0 <= replicate_index < BOOTSTRAP_REPLICATES
    ):
        raise StatisticsError("bootstrap replicate index differs")
    return hashlib.sha256(
        (
            f"{PROTOCOL}|bootstrap|{split_plaintext_sha256}|"
            f"{replicate_index}"
        ).encode("ascii")
    ).digest()


@dataclass(slots=True)
class CounterStream:
    """Domain-separated SHA-256 counter stream consumed as uint64 words."""

    root: bytes
    domain: str
    _counter: int = 0
    _block: bytes = b""
    _offset: int = 0

    def __post_init__(self) -> None:
        if type(self.root) is not bytes or len(self.root) != 32:
            raise StatisticsError("replicate root must be exactly 32 bytes")
        if type(self.domain) is not str:
            raise StatisticsError("draw domain must be ASCII text")
        try:
            encoded = self.domain.encode("ascii")
        except UnicodeEncodeError as exc:
            raise StatisticsError("draw domain must be ASCII text") from exc
        if not encoded or len(encoded) > 65_535:
            raise StatisticsError("draw domain length differs")

    @property
    def words_consumed(self) -> int:
        return self._counter * 4 - (len(self._block) - self._offset) // 8

    def _refill(self) -> None:
        domain = self.domain.encode("ascii")
        self._block = hashlib.sha256(
            self.root
            + b"\x00"
            + len(domain).to_bytes(2, "big")
            + domain
            + self._counter.to_bytes(8, "big")
        ).digest()
        self._counter += 1
        self._offset = 0

    def uint64(self) -> int:
        if self._offset == len(self._block):
            self._refill()
        value = int.from_bytes(
            self._block[self._offset : self._offset + 8],
            "big",
        )
        self._offset += 8
        return value

    def draw_below(self, n: int) -> int:
        if type(n) is not int or not 1 <= n <= _UINT64_SPACE:
            raise StatisticsError("draw upper bound differs")
        cutoff = (_UINT64_SPACE // n) * n
        while True:
            value = self.uint64()
            if value < cutoff:
                return value % n

    def draws(self, *, n: int, count: int) -> tuple[int, ...]:
        if type(count) is not int or count < 0:
            raise StatisticsError("draw count differs")
        return tuple(self.draw_below(n) for _ in range(count))


@dataclass(frozen=True, order=True, slots=True)
class BootstrapCell:
    fold: int
    ontology: str
    stratum: str
    core_count: int

    def validate(self) -> None:
        if self.fold not in (0, 1, 2):
            raise StatisticsError("cell fold differs")
        if self.ontology not in {"horn", "rewrite", "resource"}:
            raise StatisticsError("cell ontology differs")
        if not self.stratum or not self.stratum.isascii():
            raise StatisticsError("cell stratum differs")
        expected = 24 if self.stratum == "all_axes" else 32
        if self.core_count != expected:
            raise StatisticsError("cell semantic-core count differs")

    @property
    def domain(self) -> str:
        return f"cell|{self.fold}|{self.ontology}|{self.stratum}"


@dataclass(frozen=True, slots=True)
class BootstrapPlan:
    replicate_index: int
    model_seed_indices: tuple[int, ...]
    cell_indices: tuple[tuple[BootstrapCell, tuple[int, ...]], ...]


def build_bootstrap_plan(
    split_plaintext_sha256: str,
    replicate_index: int,
    cells: Iterable[BootstrapCell],
) -> BootstrapPlan:
    root = replicate_root(split_plaintext_sha256, replicate_index)
    seed_stream = CounterStream(root, "model-seeds")
    seed_indices = seed_stream.draws(
        n=MODEL_SEED_COUNT,
        count=MODEL_SEED_COUNT,
    )

    values = tuple(sorted(cells))
    if not values:
        raise StatisticsError("bootstrap cell population is empty")
    if len(set(values)) != len(values):
        raise StatisticsError("bootstrap cells are not unique")
    sampled: list[tuple[BootstrapCell, tuple[int, ...]]] = []
    for cell in values:
        cell.validate()
        stream = CounterStream(root, cell.domain)
        sampled.append(
            (
                cell,
                stream.draws(n=cell.core_count, count=cell.core_count),
            )
        )
    return BootstrapPlan(
        replicate_index=replicate_index,
        model_seed_indices=seed_indices,
        cell_indices=tuple(sampled),
    )


def simultaneous_lower_bounds(
    observed: tuple[float, ...],
    bootstrap_effects: Iterable[tuple[float, ...]],
) -> tuple[float, ...]:
    if not observed:
        raise StatisticsError("endpoint population is empty")
    maxima: list[float] = []
    for replicate in bootstrap_effects:
        if len(replicate) != len(observed):
            raise StatisticsError("bootstrap endpoint count differs")
        maxima.append(
            max(
                estimate - resample
                for estimate, resample in zip(
                    observed,
                    replicate,
                    strict=True,
                )
            )
        )
    if len(maxima) != BOOTSTRAP_REPLICATES:
        raise StatisticsError("bootstrap replicate count differs")
    maxima.sort()
    quantile = maxima[94_999]
    return tuple(estimate - quantile for estimate in observed)


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BootstrapCell",
    "BootstrapPlan",
    "CounterStream",
    "MODEL_SEED_COUNT",
    "PROTOCOL",
    "StatisticsError",
    "build_bootstrap_plan",
    "replicate_root",
    "simultaneous_lower_bounds",
]
