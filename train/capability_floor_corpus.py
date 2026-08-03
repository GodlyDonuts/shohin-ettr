#!/usr/bin/env python3
"""Build the byte-identical cross-tokenizer ETTR capability-floor cohort.

The admitted ETTR-v3 release was rendered for Shohin's tokenizer, but the
capability-floor campaign compares frozen backbones with different tokenizers.
This audit keeps the exact public ASCII bytes fixed, maps public operation
roles through character offsets, and admits a semantic core only when every
candidate can consume every source without truncation.  Assessor fields are
used only to construct offline labels and replay strata; they never enter the
candidate tokenization receipt.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Iterable, Mapping, Protocol, Sequence

import torch
from tokenizers import Tokenizer


_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE = _ROOT / "pipeline"
for _path in (_ROOT / "train", _PIPELINE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from audit_ettr_operation_family_state_conditioning import (  # noqa: E402
    _operation_contexts,
)
from audit_ettr_public_opcode_identifiability import (  # noqa: E402
    public_document_indices,
)
from ettr_il_v2_token_native_surface import (  # noqa: E402
    CODEWORD_BYTES,
    TokenNativeSurfaceCodec,
)
from ettr_il_v3_protocol import canonical_json_bytes  # noqa: E402
from ettr_v3_streaming import (  # noqa: E402
    ETTRV3StreamingRelease,
    _identity,
)
from materialize_ettr_il_v3_corpus import (  # noqa: E402
    _iter_records,
    _sha256_file,
)
from token_native_syntax_router import TokenNativeOperationRouter  # noqa: E402


CORPUS_SCHEMA = "shohin-ettr-capability-floor-corpus-v2"
INDEX_SCHEMA = "shohin-ettr-capability-floor-core-index-v2"
TOKENIZER_CONFIG_SCHEMA = "shohin-ettr-capability-floor-tokenizers-v1"
TOKENIZATION_SCHEMA = "shohin-ettr-capability-floor-tokenization-v1"
REQUIRED_CANDIDATES = (
    "protected-shohin-125m-step300k",
    "facebook-mobilellm-r1-360m",
    "qwen3.5-0.8b-text-backbone",
    "smollm3-3b",
)
_SEGMENTS = ("world", "command", "query")


class CapabilityFloorCorpusError(ValueError):
    """The cohort is not byte-identical, complete, or reproducible."""


def _canonical_bytes(value: object) -> bytes:
    return canonical_json_bytes(value)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _hex(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CapabilityFloorCorpusError(f"{label} differs")
    return value


def _relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CapabilityFloorCorpusError(f"{label} differs")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise CapabilityFloorCorpusError(f"{label} is unsafe")
    return value


def _write_no_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = path.open("xb")
    try:
        descriptor.write(payload)
        descriptor.flush()
    finally:
        descriptor.close()


def _file_inventory_sha256(path: Path) -> tuple[str, int, int]:
    """Bind a tokenizer file or its small tokenizer/config inventory."""

    path = path.expanduser().resolve()
    if path.is_file():
        digest, size = _sha256_file(path)
        return digest, size, 1
    if not path.is_dir():
        raise CapabilityFloorCorpusError("tokenizer artifact path is absent")
    names = {
        "added_tokens.json",
        "chat_template.json",
        "config.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    members = sorted(
        child
        for child in path.rglob("*")
        if child.is_file() and child.name in names
    )
    if not members:
        raise CapabilityFloorCorpusError("tokenizer artifact inventory is empty")
    aggregate = hashlib.sha256()
    total = 0
    for member in members:
        metadata = member.lstat()
        if member.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise CapabilityFloorCorpusError("tokenizer inventory member is unsafe")
        relative = member.relative_to(path).as_posix()
        digest, size = _sha256_file(member)
        aggregate.update(
            _canonical_bytes(
                {"bytes": size, "path": relative, "sha256": digest}
            )
        )
        total += size
    return aggregate.hexdigest(), total, len(members)


@dataclass(frozen=True, slots=True)
class TokenizerSpec:
    candidate: str
    path: Path
    source_revision: str
    context_limit: int
    add_bos: bool = False
    bos_token_id: int | None = None

    def validate(self) -> None:
        if (
            self.candidate not in REQUIRED_CANDIDATES
            or not isinstance(self.source_revision, str)
            or len(self.source_revision) not in {40, 64}
            or not isinstance(self.context_limit, int)
            or self.context_limit < 128
            or not isinstance(self.add_bos, bool)
            or (self.add_bos and self.bos_token_id is None)
            or (not self.add_bos and self.bos_token_id is not None)
            or (
                self.bos_token_id is not None
                and (
                    not isinstance(self.bos_token_id, int)
                    or isinstance(self.bos_token_id, bool)
                    or self.bos_token_id < 0
                )
            )
        ):
            raise CapabilityFloorCorpusError("tokenizer specification differs")


@dataclass(frozen=True, slots=True)
class EncodedSource:
    token_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]

    def validate(self, *, text_length: int, context_limit: int) -> None:
        if (
            not self.token_ids
            or len(self.token_ids) != len(self.offsets)
            or len(self.token_ids) > context_limit
            or any(
                not isinstance(token, int)
                or isinstance(token, bool)
                or token < 0
                for token in self.token_ids
            )
        ):
            raise CapabilityFloorCorpusError("candidate tokenization differs")
        previous_end = 0
        for start, end in self.offsets:
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end < start
                or end > text_length
            ):
                raise CapabilityFloorCorpusError("token offsets differ")
            if start == end == 0:
                continue
            if start < previous_end:
                raise CapabilityFloorCorpusError("token offsets overlap or regress")
            previous_end = end

    @property
    def sha256(self) -> str:
        return _digest(
            {
                "offsets": [list(value) for value in self.offsets],
                "schema": TOKENIZATION_SCHEMA,
                "token_ids": list(self.token_ids),
            }
        )


class _TokenizerAdapter(Protocol):
    spec: TokenizerSpec
    artifact_sha256: str
    artifact_bytes: int
    artifact_files: int

    def encode(self, text: str) -> EncodedSource: ...


class _RawTokenizerAdapter:
    def __init__(self, spec: TokenizerSpec):
        spec.validate()
        if not spec.path.is_file():
            raise CapabilityFloorCorpusError(
                "raw tokenizer adapter requires tokenizer.json"
            )
        self.spec = spec
        self.tokenizer = Tokenizer.from_file(str(spec.path))
        (
            self.artifact_sha256,
            self.artifact_bytes,
            self.artifact_files,
        ) = _file_inventory_sha256(spec.path)

    def encode(self, text: str) -> EncodedSource:
        if not isinstance(text, str) or not text.isascii():
            raise CapabilityFloorCorpusError("canonical source is not strict ASCII")
        value = self.tokenizer.encode(text, add_special_tokens=False)
        token_ids = tuple(int(item) for item in value.ids)
        offsets = tuple((int(left), int(right)) for left, right in value.offsets)
        if self.spec.add_bos:
            assert self.spec.bos_token_id is not None
            token_ids = (self.spec.bos_token_id, *token_ids)
            offsets = ((0, 0), *offsets)
        result = EncodedSource(token_ids, offsets)
        result.validate(
            text_length=len(text),
            context_limit=self.spec.context_limit,
        )
        return result


class _HFTokenizerAdapter:
    def __init__(self, spec: TokenizerSpec):
        spec.validate()
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise CapabilityFloorCorpusError("transformers is unavailable") from error
        self.spec = spec
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(spec.path),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        if not self.tokenizer.is_fast:
            raise CapabilityFloorCorpusError("candidate tokenizer lacks exact offsets")
        if (
            spec.add_bos
            and self.tokenizer.bos_token_id != spec.bos_token_id
        ):
            raise CapabilityFloorCorpusError("candidate BOS identity differs")
        (
            self.artifact_sha256,
            self.artifact_bytes,
            self.artifact_files,
        ) = _file_inventory_sha256(spec.path)

    def encode(self, text: str) -> EncodedSource:
        if not isinstance(text, str) or not text.isascii():
            raise CapabilityFloorCorpusError("canonical source is not strict ASCII")
        value = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
            return_token_type_ids=False,
            truncation=False,
        )
        token_ids = tuple(int(item) for item in value["input_ids"])
        offsets = tuple(
            (int(left), int(right)) for left, right in value["offset_mapping"]
        )
        if self.spec.add_bos:
            assert self.spec.bos_token_id is not None
            token_ids = (self.spec.bos_token_id, *token_ids)
            offsets = ((0, 0), *offsets)
        result = EncodedSource(token_ids, offsets)
        result.validate(
            text_length=len(text),
            context_limit=self.spec.context_limit,
        )
        return result


def load_tokenizer_adapter(spec: TokenizerSpec) -> _TokenizerAdapter:
    spec.validate()
    if spec.path.expanduser().resolve().is_file():
        return _RawTokenizerAdapter(spec)
    return _HFTokenizerAdapter(spec)


def token_mask_for_spans(
    offsets: Sequence[tuple[int, int]],
    spans: Sequence[tuple[int, int]],
) -> tuple[bool, ...]:
    """Map public ASCII spans to candidate tokens by exact overlap."""

    normalized = tuple((int(left), int(right)) for left, right in spans)
    if not normalized or any(left < 0 or right <= left for left, right in normalized):
        raise CapabilityFloorCorpusError("public role spans differ")
    mask = tuple(
        any(start < role_end and end > role_start for role_start, role_end in normalized)
        if end > start
        else False
        for start, end in offsets
    )
    if not any(mask):
        raise CapabilityFloorCorpusError("public role has no candidate token support")
    return mask


def _operation_role_atom_positions(
    codec: TokenNativeSurfaceCodec,
    router: TokenNativeOperationRouter,
    source: str,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    payload = source.encode("ascii")
    token_ids = torch.tensor([codec.token_ids(payload)], dtype=torch.long)
    document_length = len(public_document_indices(codec, source))
    document_mask = torch.arange(token_ids.shape[1])[None, :].lt(document_length)
    with torch.no_grad():
        operations, _count = router(token_ids, document_mask)
        roles, valid = router.effect_role_masks(
            token_ids,
            document_mask,
            operations,
            maximum_roles=4,
        )
    result: list[tuple[tuple[int, ...], ...]] = []
    for operation_index in range(operations.shape[1]):
        if not bool(operations[0, operation_index].any()):
            continue
        operation_roles = []
        for role_index in range(roles.shape[2]):
            if not bool(valid[0, operation_index, role_index]):
                continue
            positions = tuple(
                int(value)
                for value in roles[0, operation_index, role_index]
                .nonzero(as_tuple=False)
                .flatten()
                .tolist()
            )
            if len(positions) != 1:
                raise CapabilityFloorCorpusError(
                    "public operation role is not one exact AST atom"
                )
            operation_roles.append(positions)
        result.append(tuple(operation_roles))
    if not result:
        raise CapabilityFloorCorpusError("public command has no operations")
    return tuple(result)


def _role_spans(
    operation_roles: Sequence[Sequence[int]],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    return tuple(
        tuple(
            (position * CODEWORD_BYTES, (position + 1) * CODEWORD_BYTES)
            for position in role
        )
        for role in operation_roles
    )


def query_dependency_strata(
    answer_matrix: Sequence[Sequence[object]],
    query_index: int,
) -> tuple[str, ...]:
    if (
        len(answer_matrix) != 4
        or query_index not in {0, 1}
        or any(len(row) != 2 for row in answer_matrix)
    ):
        raise CapabilityFloorCorpusError("query answer matrix differs")
    values = tuple(row[query_index] for row in answer_matrix)
    result = []
    if values[0] != values[2] or values[1] != values[3]:
        result.append("WORLD")
    if values[0] != values[1] or values[2] != values[3]:
        result.append("COMMAND")
    return tuple(result)


def canonical_rectangle_id(
    *,
    core_id: str,
    view_id: str,
    query_variant: int,
) -> str:
    if (
        not core_id
        or not view_id
        or query_variant not in {0, 1, 2, 3}
    ):
        raise CapabilityFloorCorpusError("rectangle identity differs")
    return _digest(
        {
            "core_id": core_id,
            "query_variant": query_variant,
            "schema": INDEX_SCHEMA,
            "view_id": view_id,
        }
    )


def _family_labels_by_corner(record: object) -> tuple[tuple[str, ...], ...]:
    contexts = iter(_operation_contexts(record))
    commands = tuple(record.assessor_only.semantic_factors.commands)
    if len(commands) != 2:
        raise CapabilityFloorCorpusError("semantic command factors differ")
    command_lengths = tuple(len(value["operations"]) for value in commands)
    result = []
    for _world_index in range(2):
        for command_index in range(2):
            labels = []
            for _ in range(command_lengths[command_index]):
                try:
                    family, _state = next(contexts)
                except StopIteration as error:
                    raise CapabilityFloorCorpusError(
                        "operation family contexts underflow"
                    ) from error
                label = str(family).upper()
                if label not in {"NONE", "WRITE", "LINK"}:
                    raise CapabilityFloorCorpusError(
                        "operation family label differs"
                    )
                labels.append(label)
            result.append(tuple(labels))
    try:
        next(contexts)
    except StopIteration:
        return tuple(result)
    raise CapabilityFloorCorpusError("operation family contexts overflow")


def _iter_release_records(
    stream: ETTRV3StreamingRelease,
    split: str,
) -> Iterable[tuple[str, int, bytes, object]]:
    descriptors = sorted(
        (
            (path, descriptor)
            for path, descriptor in stream.shards.items()
            if descriptor["split"] == split
        ),
        key=lambda value: value[0],
    )
    for path_value, descriptor in descriptors:
        path = stream.data_root / path_value
        before = _identity(
            path,
            "capability-floor source shard",
            require_immutable=True,
        )
        digest, size = _sha256_file(path)
        if digest != descriptor["sha256"] or size != descriptor["bytes"]:
            raise CapabilityFloorCorpusError("source shard identity differs")
        observed = 0
        for row_index, (payload, record) in enumerate(_iter_records(path)):
            if (
                record.canonical_bytes() != payload
                or record.identity.split != split
            ):
                raise CapabilityFloorCorpusError("semantic core record differs")
            yield path_value, row_index, payload, record
            observed += 1
        if observed != descriptor["rows"]:
            raise CapabilityFloorCorpusError("source shard row count differs")
        if (
            _identity(
                path,
                "capability-floor source shard",
                require_immutable=True,
            )
            != before
        ):
            raise CapabilityFloorCorpusError("source shard changed during audit")


def _source_key(segment: str, view_index: int, source_index: int) -> str:
    if segment not in _SEGMENTS:
        raise CapabilityFloorCorpusError("source segment differs")
    return f"{segment}:{view_index}:{source_index}"


def _audit_core(
    *,
    split: str,
    record: object,
    payload: bytes,
    path_value: str,
    row_index: int,
    codec: TokenNativeSurfaceCodec,
    router: TokenNativeOperationRouter,
    adapters: Mapping[str, _TokenizerAdapter],
) -> dict[str, object]:
    if split not in {"train", "development"}:
        raise CapabilityFloorCorpusError("cohort split differs")
    core_id = str(record.identity.core_id)
    views = tuple(record.source_visible.views)
    if len(views) != 4:
        raise CapabilityFloorCorpusError("renderer orbit differs")
    families_by_corner = _family_labels_by_corner(record)
    all_families = tuple(sorted({value for row in families_by_corner for value in row}))
    answer_matrix = tuple(record.assessor_only.targets.answer_matrix)
    query_strata = tuple(
        query_dependency_strata(answer_matrix, query_index)
        for query_index in range(2)
    )
    source_values: dict[str, str] = {}
    for view_index, view in enumerate(views):
        if (
            int(view.renderer) != view_index
            or len(view.world_sources) != 4
            or len(view.command_sources) != 4
            or len(view.query_sources) != 4
        ):
            raise CapabilityFloorCorpusError("source view geometry differs")
        for segment, sources in (
            ("world", view.world_sources),
            ("command", view.command_sources),
            ("query", view.query_sources),
        ):
            for source_index, source in enumerate(sources):
                if not isinstance(source, str) or not source.isascii():
                    raise CapabilityFloorCorpusError("source bytes differ")
                source_values[_source_key(segment, view_index, source_index)] = source

    encoded: dict[str, dict[str, EncodedSource]] = {}
    rejection: dict[str, list[str]] = {}
    for candidate, adapter in adapters.items():
        encoded[candidate] = {}
        for source_key, source in source_values.items():
            try:
                encoded[candidate][source_key] = adapter.encode(source)
            except CapabilityFloorCorpusError as error:
                rejection.setdefault(candidate, []).append(
                    f"{source_key}:{str(error)}"
                )
    if rejection:
        return {
            "accepted": False,
            "core_id": core_id,
            "core_sha256": hashlib.sha256(payload).hexdigest(),
            "rejection": rejection,
        }

    operation_roles_by_source = {
        (view_index, corner_index): _operation_role_atom_positions(
            codec,
            router,
            source,
        )
        for view_index, view in enumerate(views)
        for corner_index, source in enumerate(view.command_sources)
    }
    tokenization_receipts: dict[str, str] = {}
    role_receipts: dict[str, str] = {}
    maximum_tokens: dict[str, dict[str, int]] = {}
    role_examples = 0
    for candidate in adapters:
        tokenization_receipts[candidate] = _digest(
            {
                key: value.sha256
                for key, value in sorted(encoded[candidate].items())
            }
        )
        maximum_tokens[candidate] = {
            segment: max(
                len(value.token_ids)
                for key, value in encoded[candidate].items()
                if key.startswith(f"{segment}:")
            )
            for segment in _SEGMENTS
        }
        role_hasher = hashlib.sha256()
        for view_index, view in enumerate(views):
            for corner_index, source in enumerate(view.command_sources):
                operation_roles = operation_roles_by_source[
                    (view_index, corner_index)
                ]
                if len(operation_roles) != len(families_by_corner[corner_index]):
                    raise CapabilityFloorCorpusError(
                        "public operations and family labels differ"
                    )
                value = encoded[candidate][
                    _source_key("command", view_index, corner_index)
                ]
                for operation_index, roles in enumerate(operation_roles):
                    masks = [
                        list(token_mask_for_spans(value.offsets, spans))
                        for spans in _role_spans(roles)
                    ]
                    role_hasher.update(
                        _canonical_bytes(
                            {
                                "candidate": candidate,
                                "corner": corner_index,
                                "family": families_by_corner[corner_index][
                                    operation_index
                                ],
                                "masks": masks,
                                "operation": operation_index,
                                "schema": TOKENIZATION_SCHEMA,
                                "source_sha256": hashlib.sha256(
                                    source.encode("ascii")
                                ).hexdigest(),
                                "view": view_index,
                            }
                        )
                    )
                    if candidate == next(iter(adapters)):
                        role_examples += 1
        role_receipts[candidate] = role_hasher.hexdigest()

    rectangle_entries = []
    for view_index, view in enumerate(views):
        for query_variant in range(4):
            query_index = query_variant // 2
            strata = set(all_families)
            strata.update(query_strata[query_index])
            strata.update(("WORLD-factor", "COMMAND-factor"))
            if set(all_families) & {"WRITE", "LINK"}:
                strata.add("effect-family")
            charged: dict[str, int] = {}
            for candidate in adapters:
                total = 0
                for corner_index in range(4):
                    total += len(
                        encoded[candidate][
                            _source_key("world", view_index, corner_index)
                        ].token_ids
                    )
                    total += len(
                        encoded[candidate][
                            _source_key("command", view_index, corner_index)
                        ].token_ids
                    )
                    total += len(
                        encoded[candidate][
                            _source_key("query", view_index, query_variant)
                        ].token_ids
                    )
                charged[candidate] = total
            rectangle_entries.append(
                {
                    "charged_positions": charged,
                    "query_variant": query_variant,
                    "rectangle_id": canonical_rectangle_id(
                        core_id=core_id,
                        view_id=str(view.view_id),
                        query_variant=query_variant,
                    ),
                    "strata": sorted(strata),
                    "view": view_index,
                }
            )

    return {
        "accepted": True,
        "assessor_fields_in_model_input": False,
        "core_id": core_id,
        "core_sha256": hashlib.sha256(payload).hexdigest(),
        "curriculum_stage": str(record.identity.curriculum_stage),
        "families": list(all_families),
        "generator_ordinal": int(record.identity.generator_ordinal),
        "index_schema": INDEX_SCHEMA,
        "maximum_tokens": maximum_tokens,
        "path": _relative(path_value, "cohort shard path"),
        "query_strata": [list(value) for value in query_strata],
        "rectangles": rectangle_entries,
        "role_examples": role_examples,
        "role_mask_sha256": role_receipts,
        "row_index": row_index,
        "source_payload_sha256": _digest(
            {key: value for key, value in sorted(source_values.items())}
        ),
        "split": split,
        "tokenization_sha256": tokenization_receipts,
    }


def _load_tokenizer_specs(path: Path) -> tuple[TokenizerSpec, ...]:
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CapabilityFloorCorpusError("tokenizer config is unreadable") from error
    if not isinstance(payload, Mapping) or payload.get("schema") != TOKENIZER_CONFIG_SCHEMA:
        raise CapabilityFloorCorpusError("tokenizer config schema differs")
    values = payload.get("tokenizers")
    if not isinstance(values, list) or not values:
        raise CapabilityFloorCorpusError("tokenizer config is empty")
    result = []
    for value in values:
        if not isinstance(value, Mapping):
            raise CapabilityFloorCorpusError("tokenizer config row differs")
        try:
            spec = TokenizerSpec(
                candidate=str(value["candidate"]),
                path=Path(str(value["path"])).expanduser().resolve(),
                source_revision=str(value["source_revision"]),
                context_limit=int(value["context_limit"]),
                add_bos=bool(value.get("add_bos", False)),
                bos_token_id=(
                    int(value["bos_token_id"])
                    if value.get("bos_token_id") is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CapabilityFloorCorpusError(
                "tokenizer config row differs"
            ) from error
        spec.validate()
        result.append(spec)
    if len({value.candidate for value in result}) != len(result):
        raise CapabilityFloorCorpusError("tokenizer candidate repeats")
    return tuple(result)


def build_corpus_receipt(
    *,
    release_root: Path,
    data_root: Path,
    release_sha256: str,
    shohin_tokenizer: Path,
    tokenizer_specs: Sequence[TokenizerSpec],
    index_output: Path,
    maximum_cores_per_split: int | None = None,
) -> dict[str, object]:
    _hex(release_sha256, "release SHA-256")
    if maximum_cores_per_split is not None and maximum_cores_per_split <= 0:
        raise CapabilityFloorCorpusError("maximum core count differs")
    specs = tuple(tokenizer_specs)
    if not specs:
        raise CapabilityFloorCorpusError("tokenizer intersection is empty")
    adapters = {value.candidate: load_tokenizer_adapter(value) for value in specs}
    if len(adapters) != len(specs):
        raise CapabilityFloorCorpusError("tokenizer candidate repeats")
    stream = ETTRV3StreamingRelease(
        release_root,
        expected_release_sha256=release_sha256,
        data_root=data_root,
        tokenizer_path=shohin_tokenizer,
    )
    source_verification = stream.verify_source_shards()
    codec = stream.codec
    router = TokenNativeOperationRouter(
        codec.codebook.token_ids,
        vocab_size=codec.tokenizer.get_vocab_size(),
        maximum_positions=96,
        maximum_operations=6,
    )
    index_output = index_output.expanduser().resolve()
    index_output.parent.mkdir(parents=True, exist_ok=True)
    index_file = index_output.open("xb")
    index_hasher = hashlib.sha256()
    rectangle_hasher = hashlib.sha256()
    splits: dict[str, dict[str, object]] = {}
    candidate_maximums = {
        candidate: {segment: 0 for segment in _SEGMENTS}
        for candidate in adapters
    }
    failed = True
    try:
        for split in ("train", "development"):
            seen = 0
            accepted = 0
            rejected = 0
            rectangles = 0
            role_examples = 0
            strata_counts: dict[str, int] = {}
            rejection_counts: dict[str, int] = {}
            for path_value, row_index, payload, record in _iter_release_records(
                stream, split
            ):
                if maximum_cores_per_split is not None and seen >= maximum_cores_per_split:
                    break
                seen += 1
                value = _audit_core(
                    split=split,
                    record=record,
                    payload=payload,
                    path_value=path_value,
                    row_index=row_index,
                    codec=codec,
                    router=router,
                    adapters=adapters,
                )
                if not value["accepted"]:
                    rejected += 1
                    for candidate, failures in value["rejection"].items():
                        rejection_counts[candidate] = (
                            rejection_counts.get(candidate, 0) + len(failures)
                        )
                    continue
                accepted += 1
                rectangle_values = value["rectangles"]
                rectangles += len(rectangle_values)
                role_examples += int(value["role_examples"])
                for candidate, segment_values in value["maximum_tokens"].items():
                    for segment, maximum in segment_values.items():
                        candidate_maximums[candidate][segment] = max(
                            candidate_maximums[candidate][segment],
                            int(maximum),
                        )
                for rectangle in rectangle_values:
                    rectangle_hasher.update(_canonical_bytes(rectangle))
                    for stratum in rectangle["strata"]:
                        strata_counts[stratum] = strata_counts.get(stratum, 0) + 1
                row = _canonical_bytes(value)
                index_file.write(row)
                index_hasher.update(row)
            splits[split] = {
                "accepted_cores": accepted,
                "rectangles": rectangles,
                "rejected_cores": rejected,
                "rejection_events_by_candidate": rejection_counts,
                "role_examples": role_examples,
                "seen_cores": seen,
                "strata_counts": dict(sorted(strata_counts.items())),
            }
        failed = False
    finally:
        index_file.flush()
        index_file.close()
        if failed:
            index_output.unlink(missing_ok=True)

    complete_candidates = tuple(sorted(adapters)) == tuple(sorted(REQUIRED_CANDIDATES))
    expected_counts = stream.release["training_split_core_counts"]
    full_release = maximum_cores_per_split is None
    full_population = full_release and all(
        splits[split]["seen_cores"] == expected_counts[split]
        and splits[split]["rejected_cores"] == 0
        for split in ("train", "development")
    )
    status = (
        "pass-four-tokenizer-intersection"
        if complete_candidates and full_population
        else "preflight-incomplete"
    )
    tokenizer_receipts = {}
    for candidate, adapter in adapters.items():
        tokenizer_receipts[candidate] = {
            "add_bos": adapter.spec.add_bos,
            "artifact_bytes": adapter.artifact_bytes,
            "artifact_files": adapter.artifact_files,
            "artifact_sha256": adapter.artifact_sha256,
            "bos_token_id": adapter.spec.bos_token_id,
            "context_limit": adapter.spec.context_limit,
            "source_revision": adapter.spec.source_revision,
        }
    return {
        "assessor_fields_in_model_input": False,
        "canonical_source_encoding": "strict-ascii-identical-bytes",
        "candidate_set_complete": complete_candidates,
        "index": {
            "bytes": index_output.stat().st_size,
            "path": index_output.name,
            "sha256": index_hasher.hexdigest(),
        },
        "maximum_cores_per_split": maximum_cores_per_split,
        "rectangle_manifest_sha256": rectangle_hasher.hexdigest(),
        "release_sha256": release_sha256,
        "schema": CORPUS_SCHEMA,
        "source_verification": source_verification,
        "splits": splits,
        "status": status,
        "token_truncation": "forbidden-global-core-rejection",
        "token_maximums": candidate_maximums,
        "tokenizers": tokenizer_receipts,
    }


def validate_corpus_receipt(payload: Mapping[str, object]) -> None:
    if (
        payload.get("schema") != CORPUS_SCHEMA
        or payload.get("assessor_fields_in_model_input") is not False
        or payload.get("canonical_source_encoding")
        != "strict-ascii-identical-bytes"
        or payload.get("token_truncation") != "forbidden-global-core-rejection"
    ):
        raise CapabilityFloorCorpusError("corpus receipt custody differs")
    _hex(payload.get("release_sha256"), "release SHA-256")
    _hex(payload.get("rectangle_manifest_sha256"), "rectangle manifest SHA-256")
    index = payload.get("index")
    splits = payload.get("splits")
    tokenizers = payload.get("tokenizers")
    token_maximums = payload.get("token_maximums")
    if (
        not isinstance(index, Mapping)
        or not isinstance(splits, Mapping)
        or set(splits) != {"train", "development"}
        or not isinstance(tokenizers, Mapping)
        or not tokenizers
        or not isinstance(token_maximums, Mapping)
        or set(token_maximums) != set(tokenizers)
    ):
        raise CapabilityFloorCorpusError("corpus receipt structure differs")
    _hex(index.get("sha256"), "cohort index SHA-256")
    for candidate, value in tokenizers.items():
        if candidate not in REQUIRED_CANDIDATES or not isinstance(value, Mapping):
            raise CapabilityFloorCorpusError("tokenizer receipt differs")
        _hex(value.get("artifact_sha256"), "tokenizer artifact SHA-256")
        maxima = token_maximums[candidate]
        if (
            not isinstance(maxima, Mapping)
            or set(maxima) != set(_SEGMENTS)
            or any(not isinstance(maximum, int) or maximum < 0 for maximum in maxima.values())
        ):
            raise CapabilityFloorCorpusError("token maximum receipt differs")
        accepted = sum(int(splits[split]["accepted_cores"]) for split in splits)
        if accepted and any(maximum < 1 for maximum in maxima.values()):
            raise CapabilityFloorCorpusError("token maximum receipt is empty")
    complete = set(tokenizers) == set(REQUIRED_CANDIDATES)
    if payload.get("candidate_set_complete") is not complete:
        raise CapabilityFloorCorpusError("candidate-set completion differs")
    full_population = payload.get("maximum_cores_per_split") is None and all(
        isinstance(splits[split], Mapping)
        and splits[split].get("rejected_cores") == 0
        and splits[split].get("seen_cores") == splits[split].get("accepted_cores")
        for split in ("train", "development")
    )
    expected_status = (
        "pass-four-tokenizer-intersection"
        if complete and full_population
        else "preflight-incomplete"
    )
    if payload.get("status") != expected_status:
        raise CapabilityFloorCorpusError("corpus receipt status differs")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--shohin-tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--index-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--maximum-cores-per-split", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.index_output.exists() or args.receipt_output.exists():
        raise CapabilityFloorCorpusError("corpus output already exists")
    specs = _load_tokenizer_specs(args.tokenizer_config)
    receipt = build_corpus_receipt(
        release_root=args.release_root,
        data_root=args.data_root,
        release_sha256=args.release_sha256,
        shohin_tokenizer=args.shohin_tokenizer,
        tokenizer_specs=specs,
        index_output=args.index_output,
        maximum_cores_per_split=args.maximum_cores_per_split,
    )
    validate_corpus_receipt(receipt)
    try:
        _write_no_replace(args.receipt_output, _canonical_bytes(receipt) + b"\n")
    except Exception:
        args.index_output.unlink(missing_ok=True)
        raise
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
