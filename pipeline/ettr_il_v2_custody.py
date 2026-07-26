"""Deterministic CPU custody primitives for R12-ETTR-IL-v2.

This module implements only the public, deterministic mechanics frozen in
``R12_ETTR_IL_V2_CUSTODY_SPEC.md``.  It deliberately does not create or load
confirmation secrets, encrypt data, contact an authority, launch jobs, import
model code, or touch checkpoints.

Every parser and dataclass is fail closed:

* CJ1 inputs must be byte-for-byte canonical and use the restricted JSON type
  system.
* Protocol objects reject missing and extra fields.
* split keys exist only for public train/development splits.
* exhaustive graph canonicalization refuses work above an explicit bound.
* inventory and file roots bind strictly validated, deterministically sorted
  records.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import itertools
import json
import math
from pathlib import PurePosixPath
import re
import unicodedata
from typing import Any


PROTOCOL = "R12-ETTR-IL-v2"
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
UINT16_MAX = (1 << 16) - 1
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1

MASTER_PREIMAGE = (
    b"R12_ETTR_IL_V2|2026-07-26|custody-complete|source-deleted\n"
)
MASTER_SHA256 = (
    "8d3201be7e2f0a6e047223a67342971df70bd8be533ab50c80dd42e2208432c6"
)

SPLIT_SPEC_PREIMAGE = (
    b'{"candidate_tuple_fields":["schema","fold","split","ontology","stratum",'
    b'"theory_instance","theory_pool_index","worlds","commands","depth",'
    b'"renderer","presentations","queries","opaque_seed","generator_ordinal"],'
    b'"candidate_tuple_schema":"r12-ettr-il-v2-candidate","counts":'
    b'{"score_rectangles_per_ontology_stratum":96,'
    b'"train_rectangles_per_depth_per_fit_ontology":384,'
    b'"train_rectangles_per_fit_ontology":1152},"folds":[0,1,2],'
    b'"ontologies":["horn","rewrite","resource"],'
    b'"presentations":["base","alpha_reorder","alias_split",'
    b'"relation_reification","type_twin","execution_semantics_twin"],'
    b'"protocol":"R12-ETTR-IL-v2","renderer_ids":[0,1,2,3],'
    b'"schema":"r12-ettr-il-v2-split-spec-v2","seed_domains":'
    b'["candidate-rank","opaque-name","renderer-choice","presentation-choice",'
    b'"query-choice","paraphrase-choice","donor-derangement",'
    b'"classifier-fold","classifier-permutation"],'
    b'"splits":["train","development","confirmation"],'
    b'"strata":["seen_id","rule","composition","renderer","rule_composition",'
    b'"rule_renderer","composition_renderer","all_axes"]}\n'
)
SPLIT_SPEC_SHA256 = (
    "a09f82684c8a118a633b0bb23e244de961166ebdd3593485d897c8c27deb9747"
)

FOLD_SPEC_PREIMAGES = (
    b'{"fit_ontologies":["rewrite","resource"],"fold":0,'
    b'"protocol":"R12-ETTR-IL-v2",'
    b'"schema":"r12-ettr-il-v2-fold-spec-v1","seed_context":"fold/0",'
    b'"withheld_ontology":"horn"}\n',
    b'{"fit_ontologies":["horn","resource"],"fold":1,'
    b'"protocol":"R12-ETTR-IL-v2",'
    b'"schema":"r12-ettr-il-v2-fold-spec-v1","seed_context":"fold/1",'
    b'"withheld_ontology":"rewrite"}\n',
    b'{"fit_ontologies":["horn","rewrite"],"fold":2,'
    b'"protocol":"R12-ETTR-IL-v2",'
    b'"schema":"r12-ettr-il-v2-fold-spec-v1","seed_context":"fold/2",'
    b'"withheld_ontology":"resource"}\n',
)
FOLD_SPEC_SHA256S = (
    "a4293ae0cf972abfdfb155ad4268dceeafcb7d9ebc4df975002d08896ea65ab8",
    "6e0b45dbdb28af3684db1649767a50bae6dd5ba9d8c7fdfbae6b4edad16af425",
    "ff127ca341c8215fe4d08883d2aef9a29a3ca810adc8cf44fbe7a565c4961f68",
)
FOLD_COMMITMENTS = (
    "cd21d2501e57a275267080ceec35089f5d89e8c83c4d7e3a2ac22c2a39f6eb60",
    "c8509e61b93cbac341c42a2cd73e5d58cd02edbb0eff0b06173df729d83c7d01",
    "8487125d8354be89ff15dceca987a06af2e2dfd457890b387e696002771768b5",
)

PUBLIC_SEED_ROOT_PREIMAGE = b"R12-ETTR-IL-v2/public-seed-root\n"
PUBLIC_SEED_ROOT_SHA256 = (
    "bba84905d8f0d574ddb7e348bde9dc83b19b55a0374984988ac47664c07128a4"
)
PUBLIC_SEED_ROOT = bytes.fromhex(PUBLIC_SEED_ROOT_SHA256)

CANDIDATE_SCHEMA = "r12-ettr-il-v2-candidate"
AUDIT_GRAPH_SCHEMA = "r12-ettr-il-v2-audit-graph-v1"
SOURCE_INVENTORY_SCHEMA = "r12-ettr-il-v2-source-inventory-v1"

FOLDS = (0, 1, 2)
PUBLIC_SPLITS = ("train", "development")
SPLITS = (*PUBLIC_SPLITS, "confirmation")
ONTOLOGIES = ("horn", "rewrite", "resource")
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
)
SEED_DOMAINS = (
    "candidate-rank",
    "opaque-name",
    "renderer-choice",
    "presentation-choice",
    "query-choice",
    "paraphrase-choice",
    "donor-derangement",
    "classifier-fold",
    "classifier-permutation",
)
STAGES = ("WORLD", "COMMAND", "QUERY")
CONFIDENTIALITIES = ("public", "candidate", "assessor", "ciphertext")
CANDIDATE_FIELDS = (
    "schema",
    "fold",
    "split",
    "ontology",
    "stratum",
    "theory_instance",
    "theory_pool_index",
    "worlds",
    "commands",
    "depth",
    "renderer",
    "presentations",
    "queries",
    "opaque_seed",
    "generator_ordinal",
)

HEX32_RE = re.compile(r"[0-9a-f]{64}\Z")
HEX40_RE = re.compile(r"[0-9a-f]{40}\Z")
OPAQUE_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{15,63}\Z")
MEDIA_TYPE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9!#$&^_.+\-/]{0,126}\Z")
GIT_MODE_RE = re.compile(r"100(?:644|755)\Z")
NORMALIZED_TOKEN_RE = re.compile(rb"[a-z0-9]+")
DEFAULT_MAX_GRAPH_LABELINGS = 1_000_000


class CustodyError(ValueError):
    """A deterministic custody contract was violated."""


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest of literal bytes."""

    return hashlib.sha256(payload).hexdigest()


def _reject_json_token(token: str) -> Any:
    raise CustodyError(f"CJ1 forbids JSON token {token!r}")


def _parse_json_int(token: str) -> int:
    value = int(token, 10)
    if not INT64_MIN <= value <= INT64_MAX:
        raise CustodyError("CJ1 integer is outside signed 64-bit range")
    return value


def _object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CustodyError(f"CJ1 object contains duplicate key {key!r}")
        result[key] = value
    return result


def _validate_cj1_value(value: object, path: str = "$") -> None:
    if value is None or type(value) is bool:
        return
    if type(value) is int:
        if not INT64_MIN <= value <= INT64_MAX:
            raise CustodyError(f"{path} integer is outside signed 64-bit range")
        return
    if type(value) is str:
        try:
            value.encode("utf-8", "strict")
        except UnicodeEncodeError as exc:
            raise CustodyError(f"{path} contains a lone surrogate") from exc
        if unicodedata.normalize("NFC", value) != value:
            raise CustodyError(f"{path} string is not NFC")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_cj1_value(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise CustodyError(f"{path} object key is not a string")
            _validate_cj1_value(key, f"{path}.<key>")
            _validate_cj1_value(item, f"{path}.{key}")
        return
    raise CustodyError(f"{path} has forbidden CJ1 type {type(value).__name__}")


def cj1_dumps(value: object) -> bytes:
    """Serialize one value under the strict CJ1 contract."""

    _validate_cj1_value(value)
    try:
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
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise CustodyError("value cannot be encoded as CJ1") from exc


def cj1_loads(payload: bytes) -> Any:
    """Parse one canonical CJ1 record and require literal round-trip identity."""

    if type(payload) is not bytes:
        raise CustodyError("CJ1 input must be literal bytes")
    if not payload or not payload.endswith(b"\n"):
        raise CustodyError("CJ1 input must be nonempty and end in one LF")
    try:
        text = payload.decode("ascii", "strict")
    except UnicodeDecodeError as exc:
        raise CustodyError("CJ1 input is not strict ASCII") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_from_pairs,
            parse_int=_parse_json_int,
            parse_float=_reject_json_token,
            parse_constant=_reject_json_token,
        )
    except CustodyError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise CustodyError("CJ1 input is not valid restricted JSON") from exc
    _validate_cj1_value(value)
    if cj1_dumps(value) != payload:
        raise CustodyError("CJ1 input is not byte-for-byte canonical")
    return value


def cj1_jsonl_dumps(rows: Iterable[object]) -> bytes:
    """Serialize a nonempty sequence of CJ1 records as canonical JSONL."""

    records = tuple(rows)
    if not records:
        raise CustodyError("CJ1 JSONL must contain at least one record")
    return b"".join(cj1_dumps(row) for row in records)


def cj1_jsonl_loads(payload: bytes) -> list[Any]:
    """Parse canonical nonempty CJ1 JSONL without accepting blank records."""

    if type(payload) is not bytes or not payload or not payload.endswith(b"\n"):
        raise CustodyError("CJ1 JSONL must be nonempty and end in LF")
    lines = payload.splitlines(keepends=True)
    if not lines or any(line == b"\n" for line in lines):
        raise CustodyError("CJ1 JSONL contains an empty record")
    return [cj1_loads(line) for line in lines]


def _strict_object(
    value: object,
    fields: Sequence[str],
    label: str,
) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise CustodyError(f"{label} is not an object")
    expected = frozenset(fields)
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise CustodyError(
            f"{label} fields differ: missing={missing!r}, extra={extra!r}"
        )
    return value


def _plain_int(
    value: object,
    label: str,
    *,
    minimum: int = INT64_MIN,
    maximum: int = INT64_MAX,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise CustodyError(
            f"{label} must be an integer in [{minimum}, {maximum}]"
        )
    return value


def _plain_ascii(value: object, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str:
        raise CustodyError(f"{label} is not a string")
    try:
        value.encode("ascii", "strict")
    except UnicodeEncodeError as exc:
        raise CustodyError(f"{label} is not ASCII") from exc
    if nonempty and not value:
        raise CustodyError(f"{label} is empty")
    return value


def _enum(value: object, allowed: Sequence[str], label: str) -> str:
    text = _plain_ascii(value, label)
    if text not in allowed:
        raise CustodyError(f"{label} has unrecognized value {text!r}")
    return text


def _hex(value: object, pattern: re.Pattern[str], label: str) -> str:
    text = _plain_ascii(value, label)
    if pattern.fullmatch(text) is None:
        raise CustodyError(f"{label} is not canonical lowercase hexadecimal")
    return text


def _relative_path(value: object, label: str) -> str:
    path = _plain_ascii(value, label)
    if path.startswith("/") or "\\" in path:
        raise CustodyError(f"{label} is not a relative POSIX path")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise CustodyError(f"{label} is not normalized")
    if str(PurePosixPath(path)) != path:
        raise CustodyError(f"{label} is not normalized")
    return path


def _u16be(value: int) -> bytes:
    if type(value) is not int or not 0 <= value <= UINT16_MAX:
        raise CustodyError("u16be input is outside range")
    return value.to_bytes(2, "big")


def _u32be(value: int) -> bytes:
    if type(value) is not int or not 0 <= value <= UINT32_MAX:
        raise CustodyError("u32be input is outside range")
    return value.to_bytes(4, "big")


def _u64be(value: int) -> bytes:
    if type(value) is not int or not 0 <= value <= UINT64_MAX:
        raise CustodyError("u64be input is outside range")
    return value.to_bytes(8, "big")


def fold_commitment(
    split_spec_sha256: str,
    fold_spec_sha256: str,
) -> str:
    """Derive the v2 fold commitment from its two literal preimage hashes."""

    split_hash = _hex(split_spec_sha256, HEX32_RE, "split_spec_sha256")
    fold_hash = _hex(fold_spec_sha256, HEX32_RE, "fold_spec_sha256")
    return sha256_bytes(
        PROTOCOL.encode("ascii")
        + b"\x00fold-commitment\x00"
        + bytes.fromhex(split_hash)
        + bytes.fromhex(fold_hash)
    )


@dataclass(frozen=True, slots=True)
class SplitCounts:
    score_rectangles_per_ontology_stratum: int
    train_rectangles_per_depth_per_fit_ontology: int
    train_rectangles_per_fit_ontology: int

    FIELDS = (
        "score_rectangles_per_ontology_stratum",
        "train_rectangles_per_depth_per_fit_ontology",
        "train_rectangles_per_fit_ontology",
    )

    def __post_init__(self) -> None:
        expected = (96, 384, 1152)
        actual = (
            self.score_rectangles_per_ontology_stratum,
            self.train_rectangles_per_depth_per_fit_ontology,
            self.train_rectangles_per_fit_ontology,
        )
        if actual != expected:
            raise CustodyError("split counts differ from the literal preimage")

    @classmethod
    def from_object(cls, value: object) -> SplitCounts:
        item = _strict_object(value, cls.FIELDS, "split counts")
        return cls(**item)

    def to_object(self) -> dict[str, int]:
        return {
            "score_rectangles_per_ontology_stratum": (
                self.score_rectangles_per_ontology_stratum
            ),
            "train_rectangles_per_depth_per_fit_ontology": (
                self.train_rectangles_per_depth_per_fit_ontology
            ),
            "train_rectangles_per_fit_ontology": (
                self.train_rectangles_per_fit_ontology
            ),
        }


@dataclass(frozen=True, slots=True)
class SplitSpecification:
    candidate_tuple_fields: tuple[str, ...]
    candidate_tuple_schema: str
    counts: SplitCounts
    folds: tuple[int, ...]
    ontologies: tuple[str, ...]
    presentations: tuple[str, ...]
    protocol: str
    renderer_ids: tuple[int, ...]
    schema: str
    seed_domains: tuple[str, ...]
    splits: tuple[str, ...]
    strata: tuple[str, ...]

    FIELDS = (
        "candidate_tuple_fields",
        "candidate_tuple_schema",
        "counts",
        "folds",
        "ontologies",
        "presentations",
        "protocol",
        "renderer_ids",
        "schema",
        "seed_domains",
        "splits",
        "strata",
    )

    def __post_init__(self) -> None:
        expected = (
            (self.candidate_tuple_fields, CANDIDATE_FIELDS),
            (self.folds, FOLDS),
            (self.ontologies, ONTOLOGIES),
            (self.presentations, PRESENTATIONS),
            (self.renderer_ids, (0, 1, 2, 3)),
            (self.seed_domains, SEED_DOMAINS),
            (self.splits, SPLITS),
            (self.strata, STRATA),
        )
        if any(actual != frozen for actual, frozen in expected):
            raise CustodyError("split specification differs from literal preimage")
        if (
            self.candidate_tuple_schema != CANDIDATE_SCHEMA
            or self.protocol != PROTOCOL
            or self.schema != "r12-ettr-il-v2-split-spec-v2"
        ):
            raise CustodyError("split specification identity differs")

    @classmethod
    def from_object(cls, value: object) -> SplitSpecification:
        item = _strict_object(value, cls.FIELDS, "split specification")
        array_fields = (
            "candidate_tuple_fields",
            "folds",
            "ontologies",
            "presentations",
            "renderer_ids",
            "seed_domains",
            "splits",
            "strata",
        )
        if any(type(item[field]) is not list for field in array_fields):
            raise CustodyError("split specification arrays differ")
        return cls(
            candidate_tuple_fields=tuple(item["candidate_tuple_fields"]),
            candidate_tuple_schema=item["candidate_tuple_schema"],
            counts=SplitCounts.from_object(item["counts"]),
            folds=tuple(item["folds"]),
            ontologies=tuple(item["ontologies"]),
            presentations=tuple(item["presentations"]),
            protocol=item["protocol"],
            renderer_ids=tuple(item["renderer_ids"]),
            schema=item["schema"],
            seed_domains=tuple(item["seed_domains"]),
            splits=tuple(item["splits"]),
            strata=tuple(item["strata"]),
        )

    def to_object(self) -> dict[str, Any]:
        return {
            "candidate_tuple_fields": list(self.candidate_tuple_fields),
            "candidate_tuple_schema": self.candidate_tuple_schema,
            "counts": self.counts.to_object(),
            "folds": list(self.folds),
            "ontologies": list(self.ontologies),
            "presentations": list(self.presentations),
            "protocol": self.protocol,
            "renderer_ids": list(self.renderer_ids),
            "schema": self.schema,
            "seed_domains": list(self.seed_domains),
            "splits": list(self.splits),
            "strata": list(self.strata),
        }


@dataclass(frozen=True, slots=True)
class FoldSpecification:
    fit_ontologies: tuple[str, str]
    fold: int
    protocol: str
    schema: str
    seed_context: str
    withheld_ontology: str

    FIELDS = (
        "fit_ontologies",
        "fold",
        "protocol",
        "schema",
        "seed_context",
        "withheld_ontology",
    )
    EXPECTED = (
        (("rewrite", "resource"), "horn"),
        (("horn", "resource"), "rewrite"),
        (("horn", "rewrite"), "resource"),
    )

    def __post_init__(self) -> None:
        fold = _plain_int(self.fold, "fold specification fold", minimum=0, maximum=2)
        fit, withheld = self.EXPECTED[fold]
        if self.fit_ontologies != fit or self.withheld_ontology != withheld:
            raise CustodyError("fold ontology allocation differs")
        if (
            self.protocol != PROTOCOL
            or self.schema != "r12-ettr-il-v2-fold-spec-v1"
            or self.seed_context != f"fold/{fold}"
        ):
            raise CustodyError("fold specification identity differs")

    @classmethod
    def from_object(cls, value: object) -> FoldSpecification:
        item = _strict_object(value, cls.FIELDS, "fold specification")
        if type(item["fit_ontologies"]) is not list:
            raise CustodyError("fold fit_ontologies is not an array")
        return cls(
            fit_ontologies=tuple(item["fit_ontologies"]),
            fold=item["fold"],
            protocol=item["protocol"],
            schema=item["schema"],
            seed_context=item["seed_context"],
            withheld_ontology=item["withheld_ontology"],
        )

    def to_object(self) -> dict[str, Any]:
        return {
            "fit_ontologies": list(self.fit_ontologies),
            "fold": self.fold,
            "protocol": self.protocol,
            "schema": self.schema,
            "seed_context": self.seed_context,
            "withheld_ontology": self.withheld_ontology,
        }


def verify_literal_commitments() -> dict[str, Any]:
    """Reparse and verify every public literal preimage from the custody spec."""

    if len(MASTER_PREIMAGE) != 58:
        raise CustodyError("master preimage byte count differs")
    if sha256_bytes(MASTER_PREIMAGE) != MASTER_SHA256:
        raise CustodyError("master preimage digest differs")
    split = SplitSpecification.from_object(cj1_loads(SPLIT_SPEC_PREIMAGE))
    if len(SPLIT_SPEC_PREIMAGE) != 1033:
        raise CustodyError("split preimage byte count differs")
    if sha256_bytes(SPLIT_SPEC_PREIMAGE) != SPLIT_SPEC_SHA256:
        raise CustodyError("split preimage digest differs")
    fold_values = []
    derived_commitments = []
    for index, payload in enumerate(FOLD_SPEC_PREIMAGES):
        if len(payload) != 169:
            raise CustodyError(f"fold {index} preimage byte count differs")
        digest = sha256_bytes(payload)
        if digest != FOLD_SPEC_SHA256S[index]:
            raise CustodyError(f"fold {index} preimage digest differs")
        value = FoldSpecification.from_object(cj1_loads(payload))
        if value.fold != index:
            raise CustodyError(f"fold {index} preimage identity differs")
        commitment = fold_commitment(SPLIT_SPEC_SHA256, digest)
        if commitment != FOLD_COMMITMENTS[index]:
            raise CustodyError(f"fold {index} commitment differs")
        fold_values.append(value.to_object())
        derived_commitments.append(commitment)
    if sha256_bytes(PUBLIC_SEED_ROOT_PREIMAGE) != PUBLIC_SEED_ROOT_SHA256:
        raise CustodyError("public seed root differs")
    return {
        "fold_commitments": derived_commitments,
        "fold_specs": fold_values,
        "master_sha256": MASTER_SHA256,
        "public_seed_root_sha256": PUBLIC_SEED_ROOT_SHA256,
        "split_spec": split.to_object(),
        "split_spec_sha256": SPLIT_SPEC_SHA256,
    }


def derive_public_split_key(fold: int, split: str) -> bytes:
    """Derive the public train/development key; confirmation is unavailable."""

    fold = _plain_int(fold, "fold", minimum=0, maximum=2)
    split = _enum(split, PUBLIC_SPLITS, "split")
    message = (
        PROTOCOL.encode("ascii")
        + b"\x00split-key\x00"
        + _u16be(fold)
        + b"\x00"
        + split.encode("ascii")
    )
    return hmac.new(PUBLIC_SEED_ROOT, message, hashlib.sha256).digest()


def prf(key: bytes, label: str, context: bytes) -> bytes:
    """Apply the sole v2 HMAC PRF with exact label/domain framing."""

    if type(key) is not bytes or len(key) != 32:
        raise CustodyError("PRF key must be exactly 32 bytes")
    label = _enum(label, SEED_DOMAINS, "PRF label")
    if type(context) is not bytes:
        raise CustodyError("PRF context must be literal bytes")
    label_bytes = label.encode("ascii")
    if len(context) > UINT32_MAX:
        raise CustodyError("PRF context exceeds u32 length")
    message = (
        PROTOCOL.encode("ascii")
        + b"\x00seed\x00"
        + _u16be(len(label_bytes))
        + label_bytes
        + _u32be(len(context))
        + context
    )
    return hmac.new(key, message, hashlib.sha256).digest()


def prf_stream_block(
    key: bytes,
    label: str,
    context: bytes,
    counter: int,
) -> bytes:
    """Return stream block ``PRF(K,label,context || u64be(counter))``."""

    return prf(key, label, context + _u64be(counter))


def prf_uniform_index(
    key: bytes,
    label: str,
    context: bytes,
    n: int,
) -> int:
    """Select uniformly from ``[0,n)`` using rejection-sampled PRF words."""

    n = _plain_int(n, "n", minimum=1, maximum=INT64_MAX)
    limit = ((1 << 64) // n) * n
    counter = 0
    while counter <= UINT64_MAX:
        block = prf_stream_block(key, label, context, counter)
        for offset in range(0, len(block), 8):
            word = int.from_bytes(block[offset : offset + 8], "big")
            if word < limit:
                return word % n
        counter += 1
    raise CustodyError("PRF stream exhausted its u64 counter")


def opaque_seed(split_key: bytes, context: bytes) -> int:
    """Derive the low nonnegative signed-63-bit opaque-name seed."""

    return int.from_bytes(prf(split_key, "opaque-name", context), "big") & INT64_MAX


@dataclass(frozen=True, slots=True)
class CandidateTuple:
    """Strict assessor-side candidate tuple from the literal split schema."""

    fold: int
    split: str
    ontology: str
    stratum: str
    theory_instance: str
    theory_pool_index: int
    worlds: tuple[str, str]
    commands: tuple[str, str]
    depth: int
    renderer: int
    presentations: tuple[str, ...]
    queries: tuple[str, str]
    opaque_seed: int
    generator_ordinal: int
    schema: str = CANDIDATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CANDIDATE_SCHEMA:
            raise CustodyError("candidate schema differs")
        _plain_int(self.fold, "candidate.fold", minimum=0, maximum=2)
        _enum(self.split, SPLITS, "candidate.split")
        _enum(self.ontology, ONTOLOGIES, "candidate.ontology")
        _enum(self.stratum, STRATA, "candidate.stratum")
        _hex(self.theory_instance, HEX32_RE, "candidate.theory_instance")
        _plain_int(
            self.theory_pool_index,
            "candidate.theory_pool_index",
            minimum=0,
        )
        self._validate_hash_pair(self.worlds, "worlds")
        self._validate_hash_pair(self.commands, "commands")
        _plain_int(self.depth, "candidate.depth", minimum=1, maximum=6)
        _plain_int(self.renderer, "candidate.renderer", minimum=0, maximum=3)
        if type(self.presentations) is not tuple or not self.presentations:
            raise CustodyError("candidate.presentations must be a nonempty tuple")
        if len(set(self.presentations)) != len(self.presentations):
            raise CustodyError("candidate.presentations contains duplicates")
        for value in self.presentations:
            _enum(value, PRESENTATIONS, "candidate.presentation")
        self._validate_hash_pair(self.queries, "queries")
        _plain_int(
            self.opaque_seed,
            "candidate.opaque_seed",
            minimum=0,
            maximum=INT64_MAX,
        )
        _plain_int(
            self.generator_ordinal,
            "candidate.generator_ordinal",
            minimum=0,
            maximum=INT64_MAX,
        )

    @staticmethod
    def _validate_hash_pair(values: object, label: str) -> None:
        if type(values) is not tuple or len(values) != 2:
            raise CustodyError(f"candidate.{label} must contain exactly two hashes")
        for index, value in enumerate(values):
            _hex(value, HEX32_RE, f"candidate.{label}[{index}]")
        if values[0] == values[1]:
            raise CustodyError(f"candidate.{label} hashes must be distinct")

    @classmethod
    def from_object(cls, value: object) -> CandidateTuple:
        item = _strict_object(value, CANDIDATE_FIELDS, "candidate")
        for field in ("worlds", "commands", "presentations", "queries"):
            if type(item[field]) is not list:
                raise CustodyError(f"candidate.{field} is not an array")
        return cls(
            schema=item["schema"],
            fold=item["fold"],
            split=item["split"],
            ontology=item["ontology"],
            stratum=item["stratum"],
            theory_instance=item["theory_instance"],
            theory_pool_index=item["theory_pool_index"],
            worlds=tuple(item["worlds"]),
            commands=tuple(item["commands"]),
            depth=item["depth"],
            renderer=item["renderer"],
            presentations=tuple(item["presentations"]),
            queries=tuple(item["queries"]),
            opaque_seed=item["opaque_seed"],
            generator_ordinal=item["generator_ordinal"],
        )

    def to_object(self) -> dict[str, Any]:
        return {
            "commands": list(self.commands),
            "depth": self.depth,
            "fold": self.fold,
            "generator_ordinal": self.generator_ordinal,
            "ontology": self.ontology,
            "opaque_seed": self.opaque_seed,
            "presentations": list(self.presentations),
            "queries": list(self.queries),
            "renderer": self.renderer,
            "schema": self.schema,
            "split": self.split,
            "stratum": self.stratum,
            "theory_instance": self.theory_instance,
            "theory_pool_index": self.theory_pool_index,
            "worlds": list(self.worlds),
        }

    def canonical_bytes(self) -> bytes:
        return cj1_dumps(self.to_object())


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    rank: bytes
    canonical_tuple_bytes: bytes
    candidate: CandidateTuple

    def __post_init__(self) -> None:
        if type(self.rank) is not bytes or len(self.rank) != 32:
            raise CustodyError("candidate rank must be 32 bytes")
        if self.canonical_tuple_bytes != self.candidate.canonical_bytes():
            raise CustodyError("ranked candidate bytes differ from candidate")


def candidate_rank(split_key: bytes, candidate: CandidateTuple) -> bytes:
    """Rank one canonical candidate under the split-isolated PRF."""

    return prf(
        split_key,
        "candidate-rank",
        candidate.canonical_bytes(),
    )


def rank_candidates(
    candidates: Iterable[CandidateTuple],
    split_key: bytes,
) -> tuple[RankedCandidate, ...]:
    """Validate, deduplicate, and rank candidates by the frozen tie rule."""

    ranked: list[RankedCandidate] = []
    seen: set[bytes] = set()
    for candidate in candidates:
        if not isinstance(candidate, CandidateTuple):
            raise CustodyError("candidate iterator emitted a non-CandidateTuple")
        canonical = candidate.canonical_bytes()
        if canonical in seen:
            raise CustodyError("candidate iterator emitted duplicate tuple bytes")
        seen.add(canonical)
        ranked.append(
            RankedCandidate(
                rank=candidate_rank(split_key, candidate),
                canonical_tuple_bytes=canonical,
                candidate=candidate,
            )
        )
    ranked.sort(key=lambda item: (item.rank, item.canonical_tuple_bytes))
    return tuple(ranked)


def select_candidates(
    candidates: Iterable[CandidateTuple],
    split_key: bytes,
    quota: int,
    *,
    admissible: Callable[[CandidateTuple], bool] | None = None,
) -> tuple[CandidateTuple, ...]:
    """Apply a pure admissibility filter, then take the exact ranked quota."""

    quota = _plain_int(quota, "quota", minimum=1, maximum=INT64_MAX)
    source = tuple(candidates)
    seen: set[bytes] = set()
    admitted: list[CandidateTuple] = []
    for candidate in source:
        if not isinstance(candidate, CandidateTuple):
            raise CustodyError("candidate iterator emitted a non-CandidateTuple")
        canonical = candidate.canonical_bytes()
        if canonical in seen:
            raise CustodyError("candidate iterator emitted duplicate tuple bytes")
        seen.add(canonical)
        if admissible is None:
            decision = True
        else:
            decision = admissible(candidate)
            if type(decision) is not bool:
                raise CustodyError("admissibility predicate did not return bool")
        if decision:
            admitted.append(candidate)
    ranked = rank_candidates(admitted, split_key)
    if len(ranked) < quota:
        raise CustodyError(
            f"candidate quota unavailable: required={quota}, available={len(ranked)}"
        )
    return tuple(item.candidate for item in ranked[:quota])


def raw_row_preimage(world: bytes, command: bytes, query: bytes) -> bytes:
    """Return the exact raw-row fingerprint preimage."""

    for label, payload in (
        ("world", world),
        ("command", command),
        ("query", query),
    ):
        if type(payload) is not bytes:
            raise CustodyError(f"{label} must be literal bytes")
    return cj1_dumps(
        {
            "command_hex": command.hex(),
            "query_hex": query.hex(),
            "world_hex": world.hex(),
        }
    )


def raw_row_fingerprint(world: bytes, command: bytes, query: bytes) -> str:
    return sha256_bytes(raw_row_preimage(world, command, query))


def semantic_world_fingerprint(alpha_normalized_world: object) -> str:
    return sha256_bytes(cj1_dumps(alpha_normalized_world))


def theory_fingerprint(alpha_normalized_theory: object) -> str:
    return sha256_bytes(cj1_dumps(alpha_normalized_theory))


def semantic_command_fingerprint(alpha_normalized_command: object) -> str:
    return sha256_bytes(cj1_dumps(alpha_normalized_command))


def bound_command_preimage(
    alpha_normalized_command: object,
    semantic_world_sha256: str,
) -> bytes:
    world_hash = _hex(
        semantic_world_sha256,
        HEX32_RE,
        "semantic_world_sha256",
    )
    return cj1_dumps(
        {
            "command": alpha_normalized_command,
            "world_sha256": world_hash,
        }
    )


def bound_command_fingerprint(
    alpha_normalized_command: object,
    semantic_world_sha256: str,
) -> str:
    return sha256_bytes(
        bound_command_preimage(
            alpha_normalized_command,
            semantic_world_sha256,
        )
    )


def opaque_name_bytes(name: str) -> bytes:
    name = _plain_ascii(name, "opaque name")
    if OPAQUE_NAME_RE.fullmatch(name) is None:
        raise CustodyError("opaque name violates the canonical symbol pattern")
    return name.encode("ascii")


def opaque_name_fingerprint(name: str) -> str:
    return sha256_bytes(opaque_name_bytes(name))


def sorted_opaque_name_bytes(names: Iterable[str]) -> tuple[bytes, ...]:
    values = tuple(sorted(opaque_name_bytes(name) for name in names))
    if len(values) != len(set(values)):
        raise CustodyError("opaque symbol table contains duplicate values")
    return values


def stage_token_sequence_preimage(
    stage: str,
    token_ids: Sequence[int],
) -> bytes:
    stage = _enum(stage, STAGES, "stage")
    if isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, Sequence):
        raise CustodyError("token_ids must be a sequence of integers")
    encoded = bytearray(stage.encode("ascii") + b"\x00")
    for index, token_id in enumerate(token_ids):
        encoded.extend(
            _u32be(
                _plain_int(
                    token_id,
                    f"token_ids[{index}]",
                    minimum=0,
                    maximum=UINT32_MAX,
                )
            )
        )
    return bytes(encoded)


def stage_token_sequence_fingerprint(
    stage: str,
    token_ids: Sequence[int],
) -> str:
    return sha256_bytes(stage_token_sequence_preimage(stage, token_ids))


def package_token_sequence_preimage(
    world_token_ids: Sequence[int],
    command_token_ids: Sequence[int],
    query_token_ids: Sequence[int],
) -> bytes:
    return b"".join(
        (
            stage_token_sequence_preimage("WORLD", world_token_ids),
            stage_token_sequence_preimage("COMMAND", command_token_ids),
            stage_token_sequence_preimage("QUERY", query_token_ids),
        )
    )


def package_token_sequence_fingerprint(
    world_token_ids: Sequence[int],
    command_token_ids: Sequence[int],
    query_token_ids: Sequence[int],
) -> str:
    return sha256_bytes(
        package_token_sequence_preimage(
            world_token_ids,
            command_token_ids,
            query_token_ids,
        )
    )


def normalized_13grams(payload: bytes) -> frozenset[bytes]:
    """Return literal normalized 13-token windows for one required scope."""

    if type(payload) is not bytes:
        raise CustodyError("13-gram input must be literal bytes")
    try:
        lowered = payload.decode("ascii", "strict").lower().encode("ascii")
    except UnicodeDecodeError as exc:
        raise CustodyError("13-gram input is not ASCII") from exc
    tokens = NORMALIZED_TOKEN_RE.findall(lowered)
    if len(tokens) < 13:
        raise CustodyError("13-gram scope has fewer than 13 normalized tokens")
    return frozenset(
        b" ".join(tokens[index : index + 13])
        for index in range(len(tokens) - 12)
    )


def package_normalized_13grams(
    world: bytes,
    command: bytes,
    query: bytes,
) -> frozenset[bytes]:
    """Normalize the literal WORLD/COMMAND/QUERY concatenation."""

    if any(type(item) is not bytes for item in (world, command, query)):
        raise CustodyError("package stages must be literal bytes")
    return normalized_13grams(world + b" " + command + b" " + query)


@dataclass(frozen=True, slots=True, order=True)
class AuditGraphNode:
    id: int
    color: str

    def __post_init__(self) -> None:
        _plain_int(self.id, "graph node id", minimum=0)
        _plain_ascii(self.color, "graph node color", nonempty=False)

    @classmethod
    def from_object(cls, value: object) -> AuditGraphNode:
        item = _strict_object(value, ("id", "color"), "graph node")
        return cls(id=item["id"], color=item["color"])

    def to_object(self) -> dict[str, Any]:
        return {"color": self.color, "id": self.id}


@dataclass(frozen=True, slots=True, order=True)
class AuditGraphEdge:
    src: int
    dst: int
    color: str

    def __post_init__(self) -> None:
        _plain_int(self.src, "graph edge src", minimum=0)
        _plain_int(self.dst, "graph edge dst", minimum=0)
        _plain_ascii(self.color, "graph edge color", nonempty=False)

    @classmethod
    def from_object(cls, value: object) -> AuditGraphEdge:
        item = _strict_object(value, ("src", "dst", "color"), "graph edge")
        return cls(src=item["src"], dst=item["dst"], color=item["color"])

    def to_object(self) -> dict[str, Any]:
        return {"color": self.color, "dst": self.dst, "src": self.src}


@dataclass(frozen=True, slots=True)
class AuditGraph:
    nodes: tuple[AuditGraphNode, ...]
    edges: tuple[AuditGraphEdge, ...]
    schema: str = AUDIT_GRAPH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != AUDIT_GRAPH_SCHEMA:
            raise CustodyError("audit graph schema differs")
        if type(self.nodes) is not tuple or type(self.edges) is not tuple:
            raise CustodyError("audit graph nodes/edges must be tuples")
        expected_ids = tuple(range(len(self.nodes)))
        actual_ids = tuple(node.id for node in self.nodes)
        if actual_ids != expected_ids:
            raise CustodyError("audit graph node IDs are not exactly 0..N-1")
        if tuple(sorted(self.edges)) != self.edges:
            raise CustodyError("audit graph edges are not canonically sorted")
        for edge in self.edges:
            if edge.src >= len(self.nodes) or edge.dst >= len(self.nodes):
                raise CustodyError("audit graph edge endpoint is absent")

    @classmethod
    def from_object(cls, value: object) -> AuditGraph:
        item = _strict_object(value, ("schema", "nodes", "edges"), "audit graph")
        if type(item["nodes"]) is not list or type(item["edges"]) is not list:
            raise CustodyError("audit graph nodes/edges are not arrays")
        return cls(
            schema=item["schema"],
            nodes=tuple(AuditGraphNode.from_object(row) for row in item["nodes"]),
            edges=tuple(AuditGraphEdge.from_object(row) for row in item["edges"]),
        )

    def to_object(self) -> dict[str, Any]:
        return {
            "edges": [edge.to_object() for edge in self.edges],
            "nodes": [node.to_object() for node in self.nodes],
            "schema": self.schema,
        }


@dataclass(frozen=True, slots=True)
class CanonicalGraph:
    payload: bytes
    sha256: str
    labeling_count: int

    def __post_init__(self) -> None:
        if sha256_bytes(self.payload) != self.sha256:
            raise CustodyError("canonical graph digest differs")
        _plain_int(self.labeling_count, "graph labeling count", minimum=1)


def graph_labeling_count(graph: AuditGraph) -> int:
    """Return the exact Cartesian-product permutation count."""

    class_sizes = Counter(node.color for node in graph.nodes)
    return math.prod(math.factorial(size) for size in class_sizes.values())


def canonicalize_audit_graph(
    graph: AuditGraph,
    *,
    max_labelings: int = DEFAULT_MAX_GRAPH_LABELINGS,
) -> CanonicalGraph:
    """Exhaustively canonicalize a colored directed multigraph."""

    if not isinstance(graph, AuditGraph):
        raise CustodyError("graph must be an AuditGraph")
    max_labelings = _plain_int(
        max_labelings,
        "max_labelings",
        minimum=1,
        maximum=INT64_MAX,
    )
    classes: dict[str, list[int]] = {}
    for node in graph.nodes:
        classes.setdefault(node.color, []).append(node.id)
    ordered_classes = tuple(
        (color, tuple(classes[color]))
        for color in sorted(classes, key=lambda value: value.encode("ascii"))
    )
    labeling_count = math.prod(
        math.factorial(len(nodes)) for _, nodes in ordered_classes
    )
    if labeling_count > max_labelings:
        raise CustodyError(
            "exact graph canonicalization exceeds resource bound: "
            f"required={labeling_count}, bound={max_labelings}"
        )

    def permutation_products(
        index: int,
        prefix: tuple[tuple[int, ...], ...],
    ) -> Iterable[tuple[tuple[int, ...], ...]]:
        if index == len(ordered_classes):
            yield prefix
            return
        for permutation in itertools.permutations(ordered_classes[index][1]):
            yield from permutation_products(index + 1, (*prefix, permutation))

    node_colors = [
        color
        for color, nodes in ordered_classes
        for _ in range(len(nodes))
    ]
    best: bytes | None = None
    for chosen in permutation_products(0, ()):
        old_to_new: dict[int, int] = {}
        next_id = 0
        for permutation in chosen:
            for old_id in permutation:
                old_to_new[old_id] = next_id
                next_id += 1
        relabeled_edges = sorted(
            (
                {
                    "color": edge.color,
                    "dst": old_to_new[edge.dst],
                    "src": old_to_new[edge.src],
                }
                for edge in graph.edges
            ),
            key=lambda edge: (edge["src"], edge["dst"], edge["color"]),
        )
        payload = cj1_dumps(
            {
                "edges": relabeled_edges,
                "node_colors": node_colors,
            }
        )
        if best is None or payload < best:
            best = payload
    if best is None:
        raise AssertionError("graph permutation product unexpectedly empty")
    return CanonicalGraph(
        payload=best,
        sha256=sha256_bytes(best),
        labeling_count=labeling_count,
    )


def graph_iso_fingerprint(
    graph: AuditGraph,
    *,
    max_labelings: int = DEFAULT_MAX_GRAPH_LABELINGS,
) -> str:
    return canonicalize_audit_graph(
        graph,
        max_labelings=max_labelings,
    ).sha256


def verify_color_preserving_bijection(
    left: AuditGraph,
    right: AuditGraph,
    old_to_new: Sequence[int],
) -> bool:
    """Independently verify one complete color-preserving multigraph bijection."""

    if len(left.nodes) != len(right.nodes) or len(old_to_new) != len(left.nodes):
        return False
    if (
        any(type(value) is not int for value in old_to_new)
        or sorted(old_to_new) != list(range(len(right.nodes)))
    ):
        return False
    for node in left.nodes:
        if node.color != right.nodes[old_to_new[node.id]].color:
            return False
    mapped = Counter(
        (old_to_new[edge.src], old_to_new[edge.dst], edge.color)
        for edge in left.edges
    )
    expected = Counter((edge.src, edge.dst, edge.color) for edge in right.edges)
    return mapped == expected


@dataclass(frozen=True, slots=True, order=True)
class SourceInventoryEntry:
    commit: str
    path: str
    git_mode: str
    git_blob_oid: str
    bytes: int
    sha256: str
    role: str

    FIELDS = (
        "commit",
        "path",
        "git_mode",
        "git_blob_oid",
        "bytes",
        "sha256",
        "role",
    )

    def __post_init__(self) -> None:
        _hex(self.commit, HEX40_RE, "source entry commit")
        _relative_path(self.path, "source entry path")
        if GIT_MODE_RE.fullmatch(self.git_mode) is None:
            raise CustodyError("source entry git_mode is not a regular-file mode")
        _hex(self.git_blob_oid, HEX40_RE, "source entry git_blob_oid")
        _plain_int(self.bytes, "source entry bytes", minimum=0)
        _hex(self.sha256, HEX32_RE, "source entry sha256")
        _plain_ascii(self.role, "source entry role")

    @classmethod
    def from_object(cls, value: object) -> SourceInventoryEntry:
        item = _strict_object(value, cls.FIELDS, "source inventory entry")
        return cls(**item)

    def to_object(self) -> dict[str, Any]:
        return {
            "bytes": self.bytes,
            "commit": self.commit,
            "git_blob_oid": self.git_blob_oid,
            "git_mode": self.git_mode,
            "path": self.path,
            "role": self.role,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True, order=True)
class RuntimeInventoryEntry:
    """Minimal strict identity for one executable/runtime closure member."""

    logical_path: str
    bytes: int
    sha256: str
    role: str

    FIELDS = ("logical_path", "bytes", "sha256", "role")

    def __post_init__(self) -> None:
        _relative_path(self.logical_path, "runtime entry logical_path")
        _plain_int(self.bytes, "runtime entry bytes", minimum=0)
        _hex(self.sha256, HEX32_RE, "runtime entry sha256")
        _plain_ascii(self.role, "runtime entry role")

    @classmethod
    def from_object(cls, value: object) -> RuntimeInventoryEntry:
        item = _strict_object(value, cls.FIELDS, "runtime inventory entry")
        return cls(**item)

    def to_object(self) -> dict[str, Any]:
        return {
            "bytes": self.bytes,
            "logical_path": self.logical_path,
            "role": self.role,
            "sha256": self.sha256,
        }


def source_entries_root(entries: Sequence[SourceInventoryEntry]) -> str:
    """Compute ``H(CJ1(entries))`` after enforcing canonical order."""

    values = tuple(entries)
    expected = tuple(sorted(values, key=lambda item: (item.commit, item.path)))
    if values != expected:
        raise CustodyError("source inventory entries are not canonically sorted")
    keys = [(item.commit, item.path) for item in values]
    if len(keys) != len(set(keys)):
        raise CustodyError("source inventory contains duplicate commit/path")
    return sha256_bytes(cj1_dumps([item.to_object() for item in values]))


@dataclass(frozen=True, slots=True)
class SourceInventory:
    protocol_spec_sha256: str
    legacy_commit: str
    legacy_tree: str
    implementation_commit: str
    implementation_tree: str
    entries: tuple[SourceInventoryEntry, ...]
    runtime_entries: tuple[RuntimeInventoryEntry, ...]
    inventory_sha256: str
    schema: str = SOURCE_INVENTORY_SCHEMA
    protocol: str = PROTOCOL

    FIELDS = (
        "schema",
        "protocol",
        "protocol_spec_sha256",
        "legacy_commit",
        "legacy_tree",
        "implementation_commit",
        "implementation_tree",
        "entries",
        "runtime_entries",
        "inventory_sha256",
    )

    def __post_init__(self) -> None:
        if self.schema != SOURCE_INVENTORY_SCHEMA or self.protocol != PROTOCOL:
            raise CustodyError("source inventory identity differs")
        _hex(
            self.protocol_spec_sha256,
            HEX32_RE,
            "source inventory protocol_spec_sha256",
        )
        for label, value in (
            ("legacy_commit", self.legacy_commit),
            ("legacy_tree", self.legacy_tree),
            ("implementation_commit", self.implementation_commit),
            ("implementation_tree", self.implementation_tree),
        ):
            _hex(value, HEX40_RE, f"source inventory {label}")
        expected_root = source_entries_root(self.entries)
        if self.inventory_sha256 != expected_root:
            raise CustodyError("source inventory_sha256 differs from entries root")
        expected_runtime = tuple(
            sorted(self.runtime_entries, key=lambda item: item.logical_path)
        )
        if self.runtime_entries != expected_runtime:
            raise CustodyError("runtime entries are not sorted by logical_path")
        runtime_paths = [entry.logical_path for entry in self.runtime_entries]
        if len(runtime_paths) != len(set(runtime_paths)):
            raise CustodyError("runtime entries contain duplicate logical_path")

    @classmethod
    def from_object(cls, value: object) -> SourceInventory:
        item = _strict_object(value, cls.FIELDS, "source inventory")
        if type(item["entries"]) is not list:
            raise CustodyError("source inventory entries are not an array")
        if type(item["runtime_entries"]) is not list:
            raise CustodyError("source inventory runtime_entries are not an array")
        return cls(
            schema=item["schema"],
            protocol=item["protocol"],
            protocol_spec_sha256=item["protocol_spec_sha256"],
            legacy_commit=item["legacy_commit"],
            legacy_tree=item["legacy_tree"],
            implementation_commit=item["implementation_commit"],
            implementation_tree=item["implementation_tree"],
            entries=tuple(
                SourceInventoryEntry.from_object(entry)
                for entry in item["entries"]
            ),
            runtime_entries=tuple(
                RuntimeInventoryEntry.from_object(entry)
                for entry in item["runtime_entries"]
            ),
            inventory_sha256=item["inventory_sha256"],
        )

    def to_object(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_object() for entry in self.entries],
            "implementation_commit": self.implementation_commit,
            "implementation_tree": self.implementation_tree,
            "inventory_sha256": self.inventory_sha256,
            "legacy_commit": self.legacy_commit,
            "legacy_tree": self.legacy_tree,
            "protocol": self.protocol,
            "protocol_spec_sha256": self.protocol_spec_sha256,
            "runtime_entries": [
                entry.to_object() for entry in self.runtime_entries
            ],
            "schema": self.schema,
        }


@dataclass(frozen=True, slots=True, order=True)
class FileRecord:
    path: str
    bytes: int
    sha256: str
    row_count: int
    media_type: str
    confidentiality: str

    FIELDS = (
        "path",
        "bytes",
        "sha256",
        "row_count",
        "media_type",
        "confidentiality",
    )

    def __post_init__(self) -> None:
        _relative_path(self.path, "file record path")
        _plain_int(self.bytes, "file record bytes", minimum=0)
        _hex(self.sha256, HEX32_RE, "file record sha256")
        _plain_int(self.row_count, "file record row_count", minimum=0)
        media_type = _plain_ascii(self.media_type, "file record media_type")
        if MEDIA_TYPE_RE.fullmatch(media_type) is None:
            raise CustodyError("file record media_type is not canonical")
        _enum(
            self.confidentiality,
            CONFIDENTIALITIES,
            "file record confidentiality",
        )

    @classmethod
    def from_object(cls, value: object) -> FileRecord:
        item = _strict_object(value, cls.FIELDS, "file record")
        return cls(**item)

    @classmethod
    def from_payload(
        cls,
        *,
        path: str,
        payload: bytes,
        row_count: int,
        media_type: str,
        confidentiality: str,
    ) -> FileRecord:
        if type(payload) is not bytes:
            raise CustodyError("file payload must be literal bytes")
        return cls(
            path=path,
            bytes=len(payload),
            sha256=sha256_bytes(payload),
            row_count=row_count,
            media_type=media_type,
            confidentiality=confidentiality,
        )

    def to_object(self) -> dict[str, Any]:
        return {
            "bytes": self.bytes,
            "confidentiality": self.confidentiality,
            "media_type": self.media_type,
            "path": self.path,
            "row_count": self.row_count,
            "sha256": self.sha256,
        }

    def verify_payload(self, payload: bytes) -> None:
        if type(payload) is not bytes:
            raise CustodyError("file payload must be literal bytes")
        if len(payload) != self.bytes or sha256_bytes(payload) != self.sha256:
            raise CustodyError("file payload differs from its record")


def file_set_root(records: Sequence[FileRecord]) -> str:
    """Compute the v2 file-set root from path-sorted strict records."""

    values = tuple(records)
    ordered = tuple(sorted(values, key=lambda item: item.path))
    if values != ordered:
        raise CustodyError("file records are not sorted by path")
    paths = [record.path for record in values]
    if len(paths) != len(set(paths)):
        raise CustodyError("file records contain duplicate paths")
    return sha256_bytes(cj1_dumps([record.to_object() for record in values]))


def fingerprint_index_root(fingerprints: Iterable[str]) -> str:
    """Bind a unique sorted index of lowercase SHA-256 fingerprints."""

    values = tuple(
        sorted(_hex(value, HEX32_RE, "fingerprint") for value in fingerprints)
    )
    if len(values) != len(set(values)):
        raise CustodyError("fingerprint index contains duplicates")
    return sha256_bytes(cj1_dumps(list(values)))


# Import-time verification catches accidental drift in literal protocol bytes.
verify_literal_commitments()
