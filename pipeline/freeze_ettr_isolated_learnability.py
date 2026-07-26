"""Deterministic, fail-closed freeze gate for R12-ETTR-IL-v1.

This module is deliberately CPU-only.  It does not import torch, tokenizers,
training code, or checkpoint code.  It has two responsibilities:

1. prove that the repository and preregistration provide enough exact
   semantics to materialize the isolated-learnability board; and
2. strictly audit canonical JSONL supplied by a future production generator.

The current preregistration fixes several hashes without committing their
literal byte preimages and describes several generators without an exact
machine-readable contract.  Those omissions are reported as blockers.  The
``materialize`` command therefore fails before creating an output directory
instead of choosing an interpretation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
from typing import Any


PROTOCOL = "R12-ETTR-IL-v1"
AUDIT_SCHEMA = "r12-ettr-il-freeze-audit-v1"
ROW_SCHEMA = "r12-ettr-il-assessor-row-v1"
MANIFEST_SCHEMA = "r12-ettr-il-dataset-manifest-v1"
PREREG_NAME = "R12_ETTR_ISOLATED_LEARNABILITY_PREREG.md"

MASTER_MATERIAL = "R12_ETTR_ISOLATED_LEARNABILITY_V1|2026-07-26|source-deleted"
MASTER_COMMITMENT = (
    "3b796eef284f523a125a18f5c94ae01b1d8305723751c9710086d670d35867aa"
)
SPLIT_SPEC_SHA256 = (
    "d98a6895ec11a52bce5625b8c7aa85c7be755d52d4d5082ab7d3a34b2dedff10"
)
FOLD_SPEC_SHA256 = {
    0: "66e1039afba221c9e591ec171c89107c1243c20ba00e85641164cd0be08e83c9",
    1: "abff905152652119039f83cac199288b59a8ef4b744db9e73e496a41bef3113b",
    2: "f82c82c0475c2d10b4c45c30bd07f1cb3282951fd825b6f1f0a77d029baacfeb",
}

ONTOLOGIES = ("horn", "rewrite", "resource")
SPLITS = ("train", "development", "confirmation")
STRATA = (
    "seen_id",
    "rule",
    "composition",
    "renderer",
    "rule_composition",
    "rule_renderer",
    "composition_renderer",
    "all_axes",
)
PRESENTATIONS = (
    "base",
    "alpha_reorder",
    "alias_split",
    "relation_reification",
    "type_twin",
    "execution_semantics_twin",
    "ambiguity_deleted_twin",
)
FIT_PRESENTATIONS = frozenset(PRESENTATIONS[:3])
SCORE_ONLY_PRESENTATIONS = frozenset(PRESENTATIONS[3:])

FIT_THEORIES = {
    "horn": (1, 3, 4, 6, 8, 9, 11, 13, 14, 16, 17, 18),
    "rewrite": (0, 4, 5, 7, 9, 12, 14),
    "resource": tuple(
        index
        for index in range(60)
        if index not in {0, 7, 14, 21, 30, 39, 48, 59}
    ),
}
SCORE_THEORIES = {
    "horn": (0, 2, 5, 7, 10, 12, 15, 19),
    "rewrite": (1, 2, 3, 6, 8, 10, 11, 13),
    "resource": (0, 7, 14, 21, 30, 39, 48, 59),
}
FOLDS = {
    0: {"fit": ("rewrite", "resource"), "withheld": "horn"},
    1: {"fit": ("horn", "resource"), "withheld": "rewrite"},
    2: {"fit": ("horn", "rewrite"), "withheld": "resource"},
}

EXPECTED_SOURCE_HASHES = {
    "train/endogenous_typed_theory_reactor.py": (
        "7b8a1f98267268240766775c558d9f3e98cd62c680a181993fd5be477ea9cd0a"
    ),
    "train/ettr_episode.py": (
        "daf47408eb7db53c4a2e2e50d8490d4900ab7014b0dc9bb6721c9e73a058a7d3"
    ),
    "train/ettr_objectives.py": (
        "94c4112bb6861fa7e4e89889c09bdb55730866ae0c4a73a39bbd5a1e02975bb8"
    ),
    "train/ettr_qualification.py": (
        "00ad5f07f80bd14008fa68dc85c3646c49b44e4df95cf7b06131e71b47b25920"
    ),
    "train/ettr_stage_supervisor.py": (
        "4a2511a37be5f24501e26d0aa976e1c2d9f92cbf02bff87c1c7bc65685b63207"
    ),
}

# These files are the production finite ontologies, structural variants, and
# existing score boards present at the preregistered repository commit.
EXPECTED_BOARD_SOURCE_HASHES = {
    "pipeline/cross_ontology_horn_board.py": (
        "24d5d31b2116cf58cbda41c4a25d0a880bf541ceab425acc4afb8687e88475d7"
    ),
    "pipeline/cross_ontology_rewrite_board.py": (
        "2d6231aff0fc2479d62bc660241de265232f8c06089c47ce230540ced024a198"
    ),
    "pipeline/cross_ontology_resource_board.py": (
        "b6ea173148ad0f35989160727438a93e7186a48c7822fd35c9fa887546b45d47"
    ),
    "pipeline/cross_ontology_horn_variants.py": (
        "25707fe12ccb3e588da9b666e738e377179e749428aed48b74aa264a0bd42b2b"
    ),
    "pipeline/cross_ontology_rewrite_variants.py": (
        "44aa07314d273a8b01e87d9b3143a4a3434aebb6a28531944e03f821da160383"
    ),
    "pipeline/cross_ontology_resource_variants.py": (
        "10e1c50a6023cc6be35f37e68cb620c6596453ae17e86312be75bd7e6dc39b99"
    ),
    "pipeline/ettr_factorial_qualification_board.py": (
        "74189d36e3cf77c4af890dd133fb5b0c8453821882e71a2081528f05f497f286"
    ),
    "pipeline/cross_ontology_qualification_matrix.py": (
        "767e15deb1da59fd030357962029482c7e228263314300417c9342d275a509d5"
    ),
}

SCORE_BOARD_COMMITMENTS = {
    "cross_ontology_matrix": (
        2688,
        "d1904b54a0fab8e59cfcb0b0dd464f5c8778e5b828907028ec8614aeae76d5d5",
    ),
    "factorial": (
        48,
        "18686ff7f0476b5a4432830f2a301f693833cf867656d3997a010cf17bb0149a",
    ),
    "hybrid": (
        48,
        "d155f868494f9379b214028c8d7475cc2cde08192c9b3a5bbdea5a73b29f98e2",
    ),
}

REQUIRED_ROW_KEYS = frozenset(
    {
        "schema",
        "row_id",
        "fold",
        "split",
        "ontology",
        "stratum",
        "rectangle_id",
        "world_index",
        "command_index",
        "packet_id",
        "semantic_index",
        "paraphrase_index",
        "depth",
        "renderer",
        "presentation",
        "theory_index",
        "theory_sha256",
        "semantic_world_sha256",
        "command_sha256",
        "opaque_names",
        "graph_sha256",
        "token_ids",
        "candidate",
        "terminal_sha256",
        "target",
        "disposition",
        "wrong_world_target_changed",
        "wrong_command_target_changed",
        "shuffled_state_target_changed",
    }
)
CANDIDATE_KEYS = frozenset({"world_hex", "command_hex", "query_hex"})
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
OPAQUE_RE = re.compile(r"[A-Za-z0-9_]+\Z")
NORMALIZED_WORD_RE = re.compile(r"[a-z0-9]+")
FORBIDDEN_CANDIDATE_WORDS = frozenset(
    {
        "answer",
        "confirmation",
        "development",
        "expected",
        "family",
        "horn",
        "label",
        "ontology",
        "oracle",
        "renderer",
        "resource",
        "rewrite",
        "split",
        "target",
        "theory",
        "variant",
    }
)
FORBIDDEN_DATA_PATH_FRAGMENTS = (
    "checkpoint",
    "ckpt",
    "flagship",
    "optimizer",
    "shard",
)


class FreezeError(ValueError):
    """Raised when the freeze contract cannot be established exactly."""


@dataclass(frozen=True, order=True, slots=True)
class Issue:
    code: str
    clause: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"clause": self.clause, "code": self.code, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class Geometry:
    """Expected row geometry; tests may use a smaller exact geometry."""

    train_rectangles_per_fit_ontology: int = 1152
    scored_rectangles_per_cell: int = 96
    rows_per_rectangle: int = 16


def canonical_json_bytes(value: object) -> bytes:
    """Return the only admitted canonical JSON representation."""

    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    """Serialize records literally, with one canonical object per line."""

    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FreezeError(f"{label} is not an integer")
    return value


def _plain_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise FreezeError(f"{label} is not a string")
    return value


def _hash(value: object, label: str) -> str:
    text = _plain_str(value, label)
    if HASH_RE.fullmatch(text) is None:
        raise FreezeError(f"{label} is not a lowercase SHA-256")
    return text


def _canonical_row_id(row: Mapping[str, Any]) -> str:
    material = {key: value for key, value in row.items() if key != "row_id"}
    return sha256_bytes(canonical_json_bytes(material))


def _candidate_bytes(row: Mapping[str, Any]) -> tuple[bytes, bytes, bytes]:
    candidate = row["candidate"]
    if not isinstance(candidate, dict) or set(candidate) != CANDIDATE_KEYS:
        raise FreezeError("candidate stage payload keys differ")
    decoded = []
    for key in ("world_hex", "command_hex", "query_hex"):
        value = _plain_str(candidate[key], f"candidate.{key}")
        if len(value) % 2 or re.fullmatch(r"[0-9a-f]*", value) is None:
            raise FreezeError(f"candidate.{key} is not canonical lowercase hex")
        decoded.append(bytes.fromhex(value))
    return tuple(decoded)  # type: ignore[return-value]


def _normalized_ngrams(payload: bytes, width: int = 13) -> frozenset[tuple[str, ...]]:
    try:
        text = payload.decode("ascii").lower()
    except UnicodeDecodeError as exc:
        raise FreezeError("candidate bytes are not ASCII") from exc
    words = NORMALIZED_WORD_RE.findall(text)
    return frozenset(
        tuple(words[index : index + width])
        for index in range(max(0, len(words) - width + 1))
    )


def _validate_row_shape(row: Mapping[str, Any]) -> None:
    if set(row) != REQUIRED_ROW_KEYS:
        missing = sorted(REQUIRED_ROW_KEYS - set(row))
        extra = sorted(set(row) - REQUIRED_ROW_KEYS)
        raise FreezeError(f"row keys differ: missing={missing}, extra={extra}")
    if row["schema"] != ROW_SCHEMA:
        raise FreezeError("row schema differs")
    if _canonical_row_id(row) != _hash(row["row_id"], "row_id"):
        raise FreezeError("row_id does not bind the canonical row")

    fold = _plain_int(row["fold"], "fold")
    split = _plain_str(row["split"], "split")
    ontology = _plain_str(row["ontology"], "ontology")
    stratum = _plain_str(row["stratum"], "stratum")
    depth = _plain_int(row["depth"], "depth")
    renderer = _plain_int(row["renderer"], "renderer")
    presentation = _plain_str(row["presentation"], "presentation")
    theory_index = _plain_int(row["theory_index"], "theory_index")
    if fold not in FOLDS or split not in SPLITS or ontology not in ONTOLOGIES:
        raise FreezeError("fold, split, or ontology leaves the frozen contract")
    if stratum not in STRATA:
        raise FreezeError("stratum leaves the frozen contract")
    if depth not in range(1, 7) or renderer not in range(4):
        raise FreezeError("depth or renderer leaves the frozen contract")
    if presentation not in PRESENTATIONS:
        raise FreezeError("presentation leaves the frozen contract")
    if theory_index not in range(60):
        raise FreezeError("theory index leaves bounded support")
    for key in (
        "theory_sha256",
        "semantic_world_sha256",
        "command_sha256",
        "graph_sha256",
        "terminal_sha256",
        "packet_id",
        "rectangle_id",
    ):
        _hash(row[key], key)
    for key, upper in (
        ("world_index", 2),
        ("command_index", 2),
        ("semantic_index", 2),
        ("paraphrase_index", 2),
    ):
        value = _plain_int(row[key], key)
        if value not in range(upper):
            raise FreezeError(f"{key} leaves the 2x2 rectangle")
    names = row["opaque_names"]
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or OPAQUE_RE.fullmatch(name) is None for name in names)
        or len(names) != len(set(names))
    ):
        raise FreezeError("opaque_names differ")
    tokens = row["token_ids"]
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in tokens)
    ):
        raise FreezeError("token_ids differ")
    if row["disposition"] not in {"answer", "abstain", "reject"}:
        raise FreezeError("disposition differs")
    if (
        row["disposition"] == "answer"
        and not isinstance(row["target"], bool)
    ) or (
        row["disposition"] != "answer"
        and row["target"] is not None
    ):
        raise FreezeError("target does not match its disposition")
    for key in (
        "wrong_world_target_changed",
        "wrong_command_target_changed",
        "shuffled_state_target_changed",
    ):
        if row[key] is not True:
            raise FreezeError(f"{key} is not proven")

    world, command, query = _candidate_bytes(row)
    words = set(NORMALIZED_WORD_RE.findall((world + command + query).decode("ascii").lower()))
    leaked = sorted(words & FORBIDDEN_CANDIDATE_WORDS)
    if leaked:
        raise FreezeError(f"candidate payload leaks assessor words: {leaked}")


def _validate_split_membership(row: Mapping[str, Any]) -> None:
    fold = int(row["fold"])
    split = str(row["split"])
    ontology = str(row["ontology"])
    theory = int(row["theory_index"])
    renderer = int(row["renderer"])
    depth = int(row["depth"])
    presentation = str(row["presentation"])
    stratum = str(row["stratum"])

    if split == "train":
        if ontology not in FOLDS[fold]["fit"]:
            raise FreezeError("withheld ontology entered fitting")
        if stratum != "seen_id":
            raise FreezeError("training stratum is not seen_id")
        if theory not in FIT_THEORIES[ontology]:
            raise FreezeError("score-only theory entered fitting")
        if renderer not in {0, 1}:
            raise FreezeError("score-only renderer entered fitting")
        if depth not in {1, 2, 3}:
            raise FreezeError("held-out composition entered fitting")
        if presentation not in FIT_PRESENTATIONS:
            raise FreezeError("score-only presentation entered fitting")
        return

    rule_shift = "rule" in stratum or stratum == "all_axes"
    composition_shift = "composition" in stratum or stratum == "all_axes"
    renderer_shift = "renderer" in stratum or stratum == "all_axes"
    expected_theories = SCORE_THEORIES[ontology] if rule_shift else FIT_THEORIES[ontology]
    if theory not in expected_theories:
        raise FreezeError("theory does not match scored stratum")
    if (depth in {4, 5, 6}) != composition_shift:
        raise FreezeError("composition depth does not match scored stratum")
    if renderer_shift:
        expected_renderers = {2, 3} if split == "development" else {3}
    else:
        expected_renderers = {0, 1}
    if renderer not in expected_renderers:
        raise FreezeError("renderer does not match scored stratum")
    if stratum == "all_axes":
        if presentation not in SCORE_ONLY_PRESENTATIONS:
            raise FreezeError("all_axes lacks a score-only presentation")
    elif presentation not in FIT_PRESENTATIONS:
        raise FreezeError("score-only presentation escaped all_axes")


def _audit_rectangles(rows: Sequence[Mapping[str, Any]]) -> None:
    rectangles: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rectangles[str(row["rectangle_id"])].append(row)
    for rectangle_id, group in rectangles.items():
        if len(group) != 16:
            raise FreezeError(f"rectangle {rectangle_id} does not contain 16 rows")
        coordinates = {
            (
                int(row["world_index"]),
                int(row["command_index"]),
                int(row["semantic_index"]),
                int(row["paraphrase_index"]),
            )
            for row in group
        }
        if coordinates != {
            (world, command, semantic, paraphrase)
            for world in range(2)
            for command in range(2)
            for semantic in range(2)
            for paraphrase in range(2)
        }:
            raise FreezeError(f"rectangle {rectangle_id} coordinates differ")
        packets: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
        for row in group:
            packets[(int(row["world_index"]), int(row["command_index"]))].append(row)
        terminals = {str(packet[0]["terminal_sha256"]) for packet in packets.values()}
        if len(terminals) != 4:
            raise FreezeError(f"rectangle {rectangle_id} terminal packets collide")
        for coordinate, packet in packets.items():
            if (
                len({row["packet_id"] for row in packet}) != 1
                or len({row["terminal_sha256"] for row in packet}) != 1
            ):
                raise FreezeError(
                    f"rectangle {rectangle_id} packet {coordinate} is inconsistent"
                )
            for semantic in range(2):
                targets = {
                    row["target"]
                    for row in packet
                    if int(row["semantic_index"]) == semantic
                }
                if len(targets) != 1:
                    raise FreezeError("paraphrases change their semantic target")
        outcome = {
            (
                int(row["world_index"]),
                int(row["command_index"]),
                int(row["semantic_index"]),
            ): (row["disposition"], row["target"])
            for row in group
        }
        for command in range(2):
            if not any(
                outcome[(0, command, semantic)]
                != outcome[(1, command, semantic)]
                for semantic in range(2)
            ):
                raise FreezeError("WORLD edge changes no answer")
        for world in range(2):
            if not any(
                outcome[(world, 0, semantic)]
                != outcome[(world, 1, semantic)]
                for semantic in range(2)
            ):
                raise FreezeError("COMMAND edge changes no answer")


def _audit_overlaps(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    indexes: dict[str, dict[object, str]] = {
        key: {}
        for key in (
            "raw_row",
            "semantic_world_sha256",
            "theory_sha256",
            "command_sha256",
            "opaque_name",
            "graph_sha256",
            "token_sequence",
            "normalized_13gram",
        )
    }
    for row in rows:
        split = str(row["split"])
        world, command, query = _candidate_bytes(row)
        values: dict[str, Iterable[object]] = {
            "raw_row": (
                sha256_bytes(
                    canonical_json_bytes(
                        {
                            "command_hex": command.hex(),
                            "query_hex": query.hex(),
                            "world_hex": world.hex(),
                        }
                    )
                ),
            ),
            "semantic_world_sha256": (row["semantic_world_sha256"],),
            "theory_sha256": (row["theory_sha256"],),
            "command_sha256": (row["command_sha256"],),
            "opaque_name": tuple(row["opaque_names"]),
            "graph_sha256": (row["graph_sha256"],),
            "token_sequence": (tuple(row["token_ids"]),),
            "normalized_13gram": _normalized_ngrams(world + b" " + command + b" " + query),
        }
        for kind, items in values.items():
            for item in items:
                previous = indexes[kind].setdefault(item, split)
                if previous != split:
                    raise FreezeError(
                        f"{kind} overlaps across {previous} and {split}"
                    )
    return {kind: len(values) for kind, values in sorted(indexes.items())}


def _audit_balance(rows: Sequence[Mapping[str, Any]]) -> None:
    groups: dict[tuple[object, ...], Counter[bool]] = defaultdict(Counter)
    for row in rows:
        if row["disposition"] != "answer":
            continue
        key = (
            row["fold"],
            row["split"],
            row["ontology"],
            row["stratum"],
            row["semantic_index"],
            row["paraphrase_index"],
        )
        groups[key][bool(row["target"])] += 1
    for key, counts in groups.items():
        if counts[False] != counts[True]:
            raise FreezeError(f"binary target balance differs for {key}")


def _audit_counts(
    rows: Sequence[Mapping[str, Any]],
    geometry: Geometry,
) -> dict[str, int]:
    rectangles = {
        (row["fold"], row["split"], row["ontology"], row["stratum"], row["rectangle_id"])
        for row in rows
    }
    by_cell: Counter[tuple[object, ...]] = Counter(
        (fold, split, ontology, stratum)
        for fold, split, ontology, stratum, _ in rectangles
    )
    for fold in FOLDS:
        for ontology in FOLDS[fold]["fit"]:
            key = (fold, "train", ontology, "seen_id")
            if by_cell[key] != geometry.train_rectangles_per_fit_ontology:
                raise FreezeError(f"training rectangle count differs for {key}")
        for split in ("development", "confirmation"):
            for ontology in ONTOLOGIES:
                for stratum in STRATA:
                    key = (fold, split, ontology, stratum)
                    if by_cell[key] != geometry.scored_rectangles_per_cell:
                        raise FreezeError(f"scored rectangle count differs for {key}")
    expected_rectangles = sum(by_cell.values())
    if len(rows) != expected_rectangles * geometry.rows_per_rectangle:
        raise FreezeError("dataset row count differs from rectangle geometry")
    return {
        "rectangles": expected_rectangles,
        "rows": len(rows),
    }


def audit_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    geometry: Geometry = Geometry(),
    require_full_counts: bool = True,
) -> dict[str, Any]:
    """Strictly audit assessor rows without trusting their embedded hashes."""

    if not rows:
        raise FreezeError("dataset is empty")
    for row in rows:
        if not isinstance(row, dict):
            raise FreezeError("dataset row is not an object")
        _validate_row_shape(row)
        _validate_split_membership(row)
    if len({row["row_id"] for row in rows}) != len(rows):
        raise FreezeError("row_id repeats")
    _audit_rectangles(rows)
    overlap_counts = _audit_overlaps(rows)
    _audit_balance(rows)
    counts = (
        _audit_counts(rows, geometry)
        if require_full_counts
        else {
            "rectangles": len({row["rectangle_id"] for row in rows}),
            "rows": len(rows),
        }
    )
    return {
        "all_contracts_pass": True,
        "counts": counts,
        "overlap_index_sizes": overlap_counts,
        "schema": "r12-ettr-il-row-audit-v1",
    }


def read_canonical_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read canonical JSONL and reject byte-equivalent-looking substitutes."""

    _require_safe_data_path(path)
    if not path.is_file() or path.is_symlink():
        raise FreezeError("JSONL is absent or not a regular non-symlink file")
    payload = path.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise FreezeError("JSONL is empty or lacks its final newline")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(payload.splitlines(keepends=True), start=1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FreezeError(f"JSONL row {number} is invalid") from exc
        try:
            canonical = canonical_json_bytes(value)
        except (TypeError, ValueError, UnicodeError) as exc:
            raise FreezeError(f"JSONL row {number} is not canonical") from exc
        if not isinstance(value, dict) or canonical != line:
            raise FreezeError(f"JSONL row {number} is not canonical")
        rows.append(value)
    return rows


def _require_safe_data_path(path: Path) -> None:
    parts = tuple(part.lower() for part in path.parts)
    for part in parts:
        if any(fragment in part for fragment in FORBIDDEN_DATA_PATH_FRAGMENTS):
            raise FreezeError("data path resembles a prohibited training asset")


def dataset_file_record(path: str, payload: bytes, row_count: int) -> dict[str, Any]:
    if not path or path.startswith("/") or ".." in Path(path).parts:
        raise FreezeError("dataset path is not a safe relative path")
    return {
        "bytes": len(payload),
        "path": path,
        "row_count": row_count,
        "sha256": sha256_bytes(payload),
    }


def dataset_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted((dict(record) for record in records), key=lambda item: item["path"])
    return sha256_bytes(canonical_json_bytes(ordered))


def _anchor_issues(prereg_text: str) -> list[Issue]:
    required = {
        "protocol": f"**Protocol:** `{PROTOCOL}`",
        "master_material": f"`{MASTER_MATERIAL}`",
        "master_commitment": f"`{MASTER_COMMITMENT}`",
        "split_spec": f"`{SPLIT_SPEC_SHA256}`",
        **{
            f"fold_{fold}": f"`{digest}`"
            for fold, digest in FOLD_SPEC_SHA256.items()
        },
    }
    return [
        Issue(
            "prereg_anchor_missing",
            "2/5",
            f"preregistration lacks exact {label} anchor",
        )
        for label, anchor in sorted(required.items())
        if prereg_text.count(anchor) != 1
    ]


def _source_issues(repo_root: Path) -> tuple[list[Issue], dict[str, str]]:
    issues: list[Issue] = []
    observed: dict[str, str] = {}
    for relative, expected in sorted(
        {**EXPECTED_SOURCE_HASHES, **EXPECTED_BOARD_SOURCE_HASHES}.items()
    ):
        path = repo_root / relative
        if not path.is_file() or path.is_symlink():
            issues.append(
                Issue(
                    "source_missing",
                    "2/5",
                    f"{relative} is absent or not a regular file",
                )
            )
            continue
        actual = sha256_file(path)
        observed[relative] = actual
        if actual != expected:
            issues.append(
                Issue(
                    "source_hash_mismatch",
                    "2/5",
                    f"{relative}: expected {expected}, observed {actual}",
                )
            )
    return issues, observed


def _production_capability_issues(repo_root: Path) -> tuple[list[Issue], dict[str, Any]]:
    """Import only the six finite CPU board modules and inspect public pools."""

    issues: list[Issue] = []
    pipeline = str((repo_root / "pipeline").resolve())
    previous = list(sys.path)
    module_names = (
        "cross_ontology_schema",
        "cross_ontology_horn_board",
        "cross_ontology_horn_variants",
        "cross_ontology_resource_board",
        "cross_ontology_resource_variants",
        "cross_ontology_rewrite_board",
        "cross_ontology_rewrite_variants",
    )
    displaced = {
        name: sys.modules.pop(name)
        for name in module_names
        if name in sys.modules
    }
    try:
        sys.path.insert(0, pipeline)
        imported = {
            name: importlib.import_module(name)
            for name in module_names
        }
        expected_root = Path(pipeline)
        for name, module in imported.items():
            raw_origin = getattr(module, "__file__", None)
            expected_origin = (expected_root / f"{name}.py").resolve()
            if (
                not isinstance(raw_origin, str)
                or Path(raw_origin).resolve() != expected_origin
            ):
                raise FreezeError(f"{name} imported from an unpinned origin")
        horn = imported["cross_ontology_horn_board"]
        horn_variants = imported["cross_ontology_horn_variants"]
        resource = imported["cross_ontology_resource_board"]
        resource_variants = imported["cross_ontology_resource_variants"]
        rewrite = imported["cross_ontology_rewrite_board"]
        rewrite_variants = imported["cross_ontology_rewrite_variants"]

        observed_pools = {
            "horn": {
                "fit": tuple(
                    index
                    for index in range(len(horn.THEORIES))
                    if index not in SCORE_THEORIES["horn"]
                ),
                "score": tuple(SCORE_THEORIES["horn"]),
                "total": len(horn.THEORIES),
            },
            "rewrite": {
                "fit": tuple(rewrite.TRAIN_THEORY_INDICES),
                "score": tuple(rewrite.HELDOUT_THEORY_INDICES),
                "total": len(rewrite.THEORIES),
            },
            "resource": {
                "fit": tuple(
                    index
                    for index in range(len(resource.THEORIES))
                    if index not in resource_variants.QUALIFICATION_THEORY_INDICES
                ),
                "score": tuple(resource_variants.QUALIFICATION_THEORY_INDICES),
                "total": len(resource.THEORIES),
            },
        }
        for ontology in ONTOLOGIES:
            if (
                observed_pools[ontology]["fit"] != FIT_THEORIES[ontology]
                or observed_pools[ontology]["score"] != SCORE_THEORIES[ontology]
            ):
                issues.append(
                    Issue(
                        "theory_pool_mismatch",
                        "4.1",
                        f"{ontology} production theory pools differ",
                    )
                )
        variant_counts = {
            "horn": len(horn_variants.VARIANT_ORDER),
            "rewrite": len(rewrite_variants.VARIANT_ORDER),
            "resource": len(resource_variants.VARIANT_ORDER),
        }
        if any(count != 7 for count in variant_counts.values()):
            issues.append(
                Issue(
                    "variant_api_mismatch",
                    "4.4",
                    f"production variant counts differ: {variant_counts}",
                )
            )
        resource_max_depth = int(resource.MAX_SEQUENCE_LENGTH)
        if resource_max_depth < 6:
            issues.append(
                Issue(
                    "production_depth_support_insufficient",
                    "4.3",
                    "resource executor supports depth "
                    f"{resource_max_depth}, but held-out depth 6 is required",
                )
            )
        capabilities = {
            "resource_max_composition_depth": resource_max_depth,
            "theory_pools": observed_pools,
            "variant_counts": variant_counts,
            "production_ontology_apis_available": True,
        }
    except Exception as exc:
        issues.append(
            Issue(
                "production_api_import_failed",
                "4",
                f"{type(exc).__name__}: {exc}",
            )
        )
        capabilities = {"production_ontology_apis_available": False}
    finally:
        sys.path[:] = previous
        for name in module_names:
            sys.modules.pop(name, None)
        sys.modules.update(displaced)
    return issues, capabilities


def _unresolved_prereg_issues(
    *,
    split_spec: Path | None,
    fold_specs: Mapping[int, Path],
    tokenizer: Path | None,
) -> list[Issue]:
    issues: list[Issue] = []

    def require_preimage(
        path: Path | None,
        expected: str,
        code: str,
        detail: str,
    ) -> None:
        if path is None:
            issues.append(Issue(code, "5", detail))
            return
        try:
            _require_safe_data_path(path)
        except FreezeError as exc:
            issues.append(Issue(code, "5", str(exc)))
            return
        if not path.is_file() or path.is_symlink():
            issues.append(Issue(code, "5", f"{path} is absent or not a regular file"))
            return
        payload = path.read_bytes()
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError):
            value = None
        if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
            issues.append(Issue(code, "5", f"{path} is not canonical JSON"))
        elif sha256_bytes(payload) != expected:
            issues.append(
                Issue(
                    code,
                    "5",
                    f"{path} does not match committed SHA-256 {expected}",
                )
            )

    require_preimage(
        split_spec,
        SPLIT_SPEC_SHA256,
        "split_spec_preimage_unavailable",
        "the canonical split-specification byte preimage is not present",
    )
    for fold, expected in FOLD_SPEC_SHA256.items():
        require_preimage(
            fold_specs.get(fold),
            expected,
            "fold_spec_preimage_unavailable",
            f"fold {fold} split-specification byte preimage is not present",
        )
    if tokenizer is None:
        issues.append(
            Issue(
                "tokenizer_identity_unspecified",
                "5",
                "no tokenizer path or committed tokenizer SHA-256 is frozen",
            )
        )
    elif not tokenizer.is_file() or tokenizer.is_symlink():
        issues.append(
            Issue(
                "tokenizer_identity_unspecified",
                "5",
                f"{tokenizer} is absent or not a regular file",
            )
        )
    else:
        try:
            _require_safe_data_path(tokenizer)
        except FreezeError as exc:
            issues.append(
                Issue(
                    "tokenizer_identity_unspecified",
                    "5",
                    str(exc),
                )
            )

    issues.extend(
        (
            Issue(
                "seed_domains_unspecified",
                "4.5/5",
                "development and confirmation seed-domain byte material is not frozen",
            ),
            Issue(
                "renderer_grammar_unspecified",
                "4.4",
                "renderer descriptions do not freeze a complete grammar or byte codec",
            ),
            Issue(
                "command_composition_unspecified",
                "4.3",
                "dependent operation grammars for depths 1-6 are not frozen per ontology",
            ),
            Issue(
                "selection_key_encoding_unspecified",
                "5",
                "master_commitment_bytes and canonical_tuple_bytes have no frozen byte schemas",
            ),
            Issue(
                "query_pool_unspecified",
                "4.3/8.3",
                "semantic probe pool and exact balanced selection rule are not frozen",
            ),
            Issue(
                "structural_integration_unspecified",
                "4.4",
                "production variants exist but their WORLD/COMMAND/QUERY integration is not frozen",
            ),
            Issue(
                "graph_isomorphism_audit_unspecified",
                "8.1",
                "cross-ontology canonical graph-isomorphism algorithm is not frozen",
            ),
            Issue(
                "metadata_classifier_unspecified",
                "8.6",
                "classifier, folds, features, tie handling, and chance estimator are not frozen",
            ),
            Issue(
                "confirmation_encryption_unspecified",
                "5",
                "confirmation cipher, recipient key, nonce derivation, and envelope format are not frozen",
            ),
            Issue(
                "root_signature_unspecified",
                "5",
                "root authority key, signature algorithm, and signed manifest envelope are not frozen",
            ),
        )
    )
    return issues


def build_dry_run_report(
    repo_root: Path,
    *,
    prereg_path: Path | None = None,
    split_spec: Path | None = None,
    fold_specs: Mapping[int, Path] | None = None,
    tokenizer: Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic preflight report without opening training assets."""

    root = repo_root.resolve()
    prereg = (prereg_path or root / PREREG_NAME).resolve()
    issues: list[Issue] = []
    if not prereg.is_file() or prereg.is_symlink():
        prereg_text = ""
        prereg_sha = None
        issues.append(
            Issue(
                "prereg_missing",
                "all",
                f"{prereg} is absent or not a regular file",
            )
        )
    else:
        payload = prereg.read_bytes()
        try:
            prereg_text = payload.decode("ascii")
        except UnicodeDecodeError:
            prereg_text = ""
            issues.append(
                Issue("prereg_not_ascii", "all", "preregistration is not ASCII")
            )
        prereg_sha = sha256_bytes(payload)
        issues.extend(_anchor_issues(prereg_text))
    if sha256_bytes(MASTER_MATERIAL.encode("ascii")) != MASTER_COMMITMENT:
        issues.append(
            Issue(
                "master_commitment_mismatch",
                "5",
                "master commitment does not match its declared ASCII preimage",
            )
        )
    source_issues, source_hashes = _source_issues(root)
    capability_issues, capabilities = _production_capability_issues(root)
    issues.extend(source_issues)
    issues.extend(capability_issues)
    issues.extend(
        _unresolved_prereg_issues(
            split_spec=split_spec,
            fold_specs=fold_specs or {},
            tokenizer=tokenizer,
        )
    )
    ordered = sorted(set(issues))

    def optional_input_hash(path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            _require_safe_data_path(path)
        except FreezeError:
            return None
        return (
            sha256_file(path)
            if path.is_file() and not path.is_symlink()
            else None
        )

    input_commitments = {
        "fold_specs": {
            str(fold): digest
            for fold, path in sorted((fold_specs or {}).items())
            if (digest := optional_input_hash(path)) is not None
        },
        "split_spec_sha256": optional_input_hash(split_spec),
        "tokenizer_sha256": optional_input_hash(tokenizer),
    }
    return {
        "all_contracts_pass": not ordered,
        "blocked_clauses": [issue.as_dict() for issue in ordered],
        "materialization_authorized": not ordered,
        "master_commitment": MASTER_COMMITMENT,
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "input_commitments": input_commitments,
        "prereg_sha256": prereg_sha,
        "production_capabilities": capabilities,
        "protocol": PROTOCOL,
        "schema": AUDIT_SCHEMA,
        "score_only_boards": {
            "commitments": {
                name: {"rows": rows, "sha256": digest}
                for name, (rows, digest) in sorted(
                    SCORE_BOARD_COMMITMENTS.items()
                )
            },
            "opened": False,
        },
        "source_hashes": dict(sorted(source_hashes.items())),
        "status": "ready" if not ordered else "blocked",
    }


def _parse_fold_specs(values: Sequence[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        try:
            raw_fold, raw_path = value.split("=", 1)
            fold = int(raw_fold)
        except (ValueError, TypeError) as exc:
            raise FreezeError("--fold-spec must be FOLD=PATH") from exc
        if fold not in FOLD_SPEC_SHA256 or fold in result or not raw_path:
            raise FreezeError("--fold-spec has an invalid or duplicate fold")
        result[fold] = Path(raw_path)
    return result


def _write_report(report: Mapping[str, Any], output: Path | None) -> None:
    payload = canonical_json_bytes(dict(report))
    if output is None:
        sys.stdout.buffer.write(payload)
        return
    _require_safe_data_path(output)
    if output.exists():
        raise FreezeError("refusing to replace an existing report")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo-root", type=Path, default=Path(__file__).parents[1])
    common.add_argument("--prereg", type=Path)
    common.add_argument("--split-spec", type=Path)
    common.add_argument("--fold-spec", action="append", default=[], metavar="FOLD=PATH")
    common.add_argument("--tokenizer", type=Path)
    common.add_argument("--report", type=Path)

    subparsers.add_parser("dry-run", parents=[common])

    audit = subparsers.add_parser("audit", parents=[common])
    audit.add_argument("--rows", type=Path, required=True)
    audit.add_argument("--allow-partial-counts", action="store_true")

    materialize = subparsers.add_parser("materialize", parents=[common])
    materialize.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        fold_specs = _parse_fold_specs(args.fold_spec)
        report = build_dry_run_report(
            args.repo_root,
            prereg_path=args.prereg,
            split_spec=args.split_spec,
            fold_specs=fold_specs,
            tokenizer=args.tokenizer,
        )
        if args.command == "audit":
            rows = read_canonical_jsonl(args.rows)
            report = {
                **report,
                "row_audit": audit_rows(
                    rows,
                    require_full_counts=not args.allow_partial_counts,
                ),
            }
            report["all_contracts_pass"] = bool(
                report["materialization_authorized"]
                and report["row_audit"]["all_contracts_pass"]
            )
        elif args.command == "materialize":
            if args.output.exists():
                raise FreezeError("refusing to replace an existing output")
            if not report["materialization_authorized"]:
                codes = ", ".join(
                    item["code"] for item in report["blocked_clauses"]
                )
                raise FreezeError(
                    "materialization blocked before output creation: " + codes
                )
            raise FreezeError(
                "materialization has no admitted production row generator"
            )
        _write_report(report, args.report)
        return 0 if args.command == "dry-run" or report["all_contracts_pass"] else 1
    except (FreezeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
