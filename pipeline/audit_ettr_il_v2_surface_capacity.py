"""Reproducible source-capacity audit for R12-ETTR-IL-v2."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from cross_ontology_horn_board import (
    THEORIES as HORN_THEORIES,
    all_ground_atoms,
    challenge_initials as horn_initials,
)
from cross_ontology_resource_board import (
    THEORIES as RESOURCE_THEORIES,
    input_markings as resource_initials,
)
from cross_ontology_rewrite_board import (
    THEORIES as REWRITE_THEORIES,
    challenge_terms as rewrite_initials,
)
from ettr_il_v2_canary import _rectangle_id
from ettr_il_v2_candidate_search import (
    find_first_depth1_checkerboard,
    semantic_core_id,
)
from ettr_il_v2_custody import derive_public_split_key, prf
from ettr_il_v2_semantics import (
    HornCommand,
    HornPolicy,
    HornWorld,
    Ontology,
    ResourceCommand,
    ResourcePolicy,
    ResourceWorld,
    RewriteCommand,
    RewritePolicy,
    RewriteWorld,
)
from ettr_il_v2_surface import (
    SurfaceCall,
    SurfaceInteger,
    SurfaceNode,
    SurfaceRenderer,
    SurfaceSymbol,
)
from ettr_il_v2_surface_adapter import (
    IMPLEMENTED_PRESENTATIONS,
    SurfaceAdapterContext,
    build_base_surface_bundle,
    canonical_factor_ast,
    presented_factor_ast,
)
from ettr_il_v2_token_native_surface import TokenNativeSurfaceCodec


REPORT_SCHEMA = "r12-ettr-il-v2-surface-capacity-audit-v2"
TOKENIZER_SHA256 = (
    "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
)
WIDTHS = {"command": 96, "query": 48, "world": 192}


class CapacityAuditError(ValueError):
    """The surface-capacity audit cannot be reproduced."""


@dataclass(frozen=True, slots=True)
class NodeCount:
    calls: int
    integers: int
    symbols: int

    @property
    def total(self) -> int:
        return self.calls + self.integers + self.symbols

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "integers": self.integers,
            "symbols": self.symbols,
            "total": self.total,
        }


def canonical_json_bytes(value: object) -> bytes:
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


def count_nodes(node: SurfaceNode) -> NodeCount:
    if isinstance(node, SurfaceInteger):
        return NodeCount(0, 1, 0)
    if isinstance(node, SurfaceSymbol):
        return NodeCount(0, 0, 1)
    if not isinstance(node, SurfaceCall):
        raise TypeError("node must be a SurfaceNode")
    children = tuple(count_nodes(child) for child in node.children)
    return NodeCount(
        1 + sum(child.calls for child in children),
        sum(child.integers for child in children),
        sum(child.symbols for child in children),
    )


def _bounded_factors() -> dict[str, tuple[object, ...]]:
    evidence = "0" * 64
    largest_horn_atom = max(
        all_ground_atoms(),
        key=lambda atom: len(atom.arguments),
    )
    return {
        "horn_world": tuple(
            HornWorld(
                evidence,
                theory_index,
                initial,
                HornPolicy.PERSISTENT,
            )
            for theory_index in range(len(HORN_THEORIES))
            for initial in horn_initials()
        ),
        "rewrite_world": tuple(
            RewriteWorld(
                evidence,
                theory_index,
                initial,
                RewritePolicy.CONTEXTUAL,
            )
            for theory_index in range(len(REWRITE_THEORIES))
            for initial in rewrite_initials()
            if initial.type_index == 0
        ),
        "resource_world": tuple(
            ResourceWorld(
                evidence,
                theory_index,
                initial,
                ResourcePolicy.ATOMIC_DEADLOCK,
            )
            for theory_index in range(len(RESOURCE_THEORIES))
            for initial in resource_initials()
        ),
        "horn_command": tuple(
            HornCommand(depth, (largest_horn_atom,) * depth)
            for depth in range(1, 7)
        ),
        "rewrite_command": tuple(
            RewriteCommand(depth, (0,) * depth)
            for depth in range(1, 7)
        ),
        "resource_command": tuple(
            ResourceCommand(depth, (0,) * depth)
            for depth in range(1, 7)
        ),
    }


def bounded_node_report() -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for name, factors in _bounded_factors().items():
        counts = tuple(
            count_nodes(canonical_factor_ast(factor)) for factor in factors
        )
        report[name] = {
            "factor_count": len(factors),
            "maximum": max(counts, key=lambda value: value.total).as_dict(),
            "minimum": min(counts, key=lambda value: value.total).as_dict(),
        }
    return report


def bounded_presentation_report(
    codec: TokenNativeSurfaceCodec,
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for name, factors in _bounded_factors().items():
        stage = "world" if name.endswith("_world") else "command"
        width = WIDTHS[stage]
        presentations: dict[str, Any] = {}
        for presentation in IMPLEMENTED_PRESENTATIONS:
            asts = tuple(
                presented_factor_ast(
                    factor,
                    presentation=presentation,
                )
                for factor in factors
            )
            node_counts = tuple(count_nodes(ast).total for ast in asts)
            token_counts = tuple(
                len(
                    codec.serialize(
                        ast,
                        SurfaceRenderer.CANONICAL_JSON,
                    ).token_ids
                )
                for ast in asts
            )
            presentations[presentation] = {
                "maximum_ast_nodes": max(node_counts),
                "maximum_token_count": max(token_counts),
                "minimum_ast_nodes": min(node_counts),
                "minimum_token_count": min(token_counts),
                "within_width": max(token_counts) <= width,
            }
        report[name] = {
            "factor_count": len(factors),
            "presentations": presentations,
            "stage": stage,
            "width": width,
        }
    return report


def _token_count(tokenizer: Any, payload: bytes) -> int:
    return len(tokenizer.encode(payload.decode("ascii")).ids)


def verbose_checkerboard_report(tokenizer: Any) -> dict[str, object]:
    key = derive_public_split_key(0, "train")
    report: dict[str, object] = {}
    for ontology in Ontology:
        candidate = find_first_depth1_checkerboard(ontology)
        core_id = semantic_core_id(candidate)
        renderers: dict[str, object] = {}
        for renderer in SurfaceRenderer:
            rectangle_id = _rectangle_id(core_id, renderer)
            bundle = build_base_surface_bundle(
                candidate.worlds,
                candidate.commands,
                (
                    candidate.queries.slot_0,
                    candidate.queries.slot_1,
                ),
                context=SurfaceAdapterContext(
                    fold=0,
                    split="train",
                    semantic_core_id=core_id,
                    semantic_rectangle_id=rectangle_id,
                    renderer=renderer,
                    prf=lambda label, payload: prf(key, label, payload),
                ),
            )
            lengths = {
                "command": tuple(
                    _token_count(tokenizer, document.source)
                    for pair in bundle.command_variants
                    for document in pair
                ),
                "query": tuple(
                    _token_count(tokenizer, prefix.prefix)
                    for pair in bundle.query_prefixes
                    for prefix in pair
                ),
                "world": tuple(
                    _token_count(tokenizer, document.source)
                    for pair in bundle.world_variants
                    for document in pair
                ),
            }
            renderers[renderer.name.lower()] = {
                stage: {
                    "maximum": max(values),
                    "minimum": min(values),
                    "width": WIDTHS[stage],
                    "within_width": max(values) <= WIDTHS[stage],
                }
                for stage, values in lengths.items()
            }
        report[ontology.value] = renderers
    return report


def audit(tokenizer_path: Path) -> dict[str, object]:
    payload = tokenizer_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != TOKENIZER_SHA256:
        raise CapacityAuditError("tokenizer SHA-256 differs")
    try:
        from tokenizers import Tokenizer  # noqa: PLC0415
    except ImportError as exc:
        raise CapacityAuditError("tokenizers runtime is unavailable") from exc
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    codec = TokenNativeSurfaceCodec(tokenizer_path)
    node_report = bounded_node_report()
    presentation_report = bounded_presentation_report(codec)
    return {
        "bounded_ast_nodes": node_report,
        "bounded_presentations": presentation_report,
        "codebook_sha256": codec.codebook_sha256,
        "compact_one_token_per_node_feasible": all(
            entry["maximum"]["total"]
            <= WIDTHS[
                "world" if name.endswith("_world") else "command"
            ]
            for name, entry in node_report.items()
        ),
        "fixed_transport_capacity_pass": all(
            presentation["within_width"]
            for factor in presentation_report.values()
            for presentation in factor["presentations"].values()
        ),
        "current_verbose_checkerboards": verbose_checkerboard_report(tokenizer),
        "schema": REPORT_SCHEMA,
        "status": "token_native_transport_capacity_pass",
        "tokenizer_sha256": TOKENIZER_SHA256,
        "widths": WIDTHS,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.tokenizer)
    encoded = canonical_json_bytes(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(encoded.decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CapacityAuditError",
    "NodeCount",
    "REPORT_SCHEMA",
    "TOKENIZER_SHA256",
    "WIDTHS",
    "audit",
    "bounded_node_report",
    "bounded_presentation_report",
    "canonical_json_bytes",
    "count_nodes",
    "verbose_checkerboard_report",
]
