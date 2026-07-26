from __future__ import annotations

from dataclasses import replace
import hashlib
import hmac
import json

import pytest

from ettr_il_v2_custody import (
    AUDIT_GRAPH_SCHEMA,
    FOLD_COMMITMENTS,
    FOLD_SPEC_PREIMAGES,
    FOLD_SPEC_SHA256S,
    MASTER_PREIMAGE,
    MASTER_SHA256,
    PROTOCOL,
    PUBLIC_SEED_ROOT,
    PUBLIC_SEED_ROOT_PREIMAGE,
    PUBLIC_SEED_ROOT_SHA256,
    SEED_DOMAINS,
    SPLIT_SPEC_PREIMAGE,
    SPLIT_SPEC_SHA256,
    AuditGraph,
    AuditGraphEdge,
    AuditGraphNode,
    CandidateTuple,
    CustodyError,
    FileRecord,
    RuntimeInventoryEntry,
    FoldSpecification,
    SourceInventory,
    SourceInventoryEntry,
    SplitSpecification,
    bound_command_fingerprint,
    candidate_rank,
    canonicalize_audit_graph,
    cj1_dumps,
    cj1_jsonl_dumps,
    cj1_jsonl_loads,
    cj1_loads,
    derive_public_split_key,
    file_set_root,
    fingerprint_index_root,
    fold_commitment,
    graph_iso_fingerprint,
    normalized_13grams,
    opaque_name_fingerprint,
    opaque_seed,
    package_normalized_13grams,
    package_token_sequence_fingerprint,
    prf,
    prf_stream_block,
    prf_uniform_index,
    rank_candidates,
    raw_row_fingerprint,
    semantic_command_fingerprint,
    semantic_world_fingerprint,
    select_candidates,
    sha256_bytes,
    source_entries_root,
    stage_token_sequence_preimage,
    theory_fingerprint,
    verify_color_preserving_bijection,
    verify_literal_commitments,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _candidate(ordinal: int, *, split: str = "train") -> CandidateTuple:
    return CandidateTuple(
        fold=1,
        split=split,
        ontology="horn",
        stratum="rule_composition",
        theory_instance=_digest(f"theory:{ordinal}"),
        theory_pool_index=ordinal,
        worlds=(_digest(f"w0:{ordinal}"), _digest(f"w1:{ordinal}")),
        commands=(_digest(f"c0:{ordinal}"), _digest(f"c1:{ordinal}")),
        depth=1 + ordinal % 6,
        renderer=ordinal % 4,
        presentations=("base", "alpha_reorder"),
        queries=(_digest(f"q0:{ordinal}"), _digest(f"q1:{ordinal}")),
        opaque_seed=ordinal,
        generator_ordinal=ordinal,
    )


def _graph(permutation: tuple[int, ...] = (0, 1, 2)) -> AuditGraph:
    colors = ("term", "term", "query")
    original_edges = (
        (0, 1, "arg"),
        (0, 1, "arg"),
        (1, 2, "ref"),
        (2, 2, "self"),
    )
    inverse = {old: new for new, old in enumerate(permutation)}
    nodes = tuple(
        AuditGraphNode(id=new, color=colors[old])
        for new, old in enumerate(permutation)
    )
    edges = tuple(
        sorted(
            (
                AuditGraphEdge(
                    src=inverse[src],
                    dst=inverse[dst],
                    color=color,
                )
                for src, dst, color in original_edges
            )
        )
    )
    return AuditGraph(nodes=nodes, edges=edges)


def test_literal_preimages_reproduce_all_frozen_commitments() -> None:
    report = verify_literal_commitments()
    assert len(MASTER_PREIMAGE) == 58
    assert sha256_bytes(MASTER_PREIMAGE) == MASTER_SHA256
    assert len(SPLIT_SPEC_PREIMAGE) == 1033
    assert sha256_bytes(SPLIT_SPEC_PREIMAGE) == SPLIT_SPEC_SHA256
    assert report["split_spec"]["candidate_tuple_schema"] == (
        "r12-ettr-il-v2-candidate"
    )
    assert (
        SplitSpecification.from_object(report["split_spec"]).to_object()
        == report["split_spec"]
    )
    for fold, payload in enumerate(FOLD_SPEC_PREIMAGES):
        assert len(payload) == 169
        assert sha256_bytes(payload) == FOLD_SPEC_SHA256S[fold]
        assert (
            fold_commitment(SPLIT_SPEC_SHA256, FOLD_SPEC_SHA256S[fold])
            == FOLD_COMMITMENTS[fold]
        )
        assert FoldSpecification.from_object(
            report["fold_specs"][fold]
        ).fold == fold
    assert sha256_bytes(PUBLIC_SEED_ROOT_PREIMAGE) == PUBLIC_SEED_ROOT_SHA256


def test_cj1_round_trip_is_canonical_and_deterministic() -> None:
    expected = b'{"a":["\\u00e9",null,true],"z":3}\n'
    value = {"z": 3, "a": ["é", None, True]}
    assert cj1_dumps(value) == expected
    assert cj1_loads(expected) == {"a": ["é", None, True], "z": 3}
    rows = [value, {"next": -4}]
    payload = cj1_jsonl_dumps(rows)
    assert cj1_jsonl_loads(payload) == rows


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b'{"a":1,"a":2}\n', "duplicate key"),
        (b'{"x":1.0}\n', "forbids JSON token"),
        (b'{"x":NaN}\n', "forbids JSON token"),
        (b'{"x":9223372036854775808}\n', "outside signed 64-bit"),
        (b'{ "x":1}\n', "not byte-for-byte canonical"),
        (b'{"x":1}', "end in one LF"),
        (b'{"x":"\xc3\xa9"}\n', "not strict ASCII"),
    ],
)
def test_cj1_rejects_noncanonical_or_forbidden_bytes(
    payload: bytes,
    message: str,
) -> None:
    with pytest.raises(CustodyError, match=message):
        cj1_loads(payload)


def test_cj1_rejects_non_nfc_lone_surrogate_and_non_json_types() -> None:
    with pytest.raises(CustodyError, match="not NFC"):
        cj1_dumps({"x": "e\u0301"})
    with pytest.raises(CustodyError, match="lone surrogate"):
        cj1_dumps({"x": "\ud800"})
    with pytest.raises(CustodyError, match="forbidden CJ1 type"):
        cj1_dumps({"x": (1, 2)})
    with pytest.raises(CustodyError, match="at least one"):
        cj1_jsonl_dumps([])
    with pytest.raises(CustodyError, match="empty record"):
        cj1_jsonl_loads(b'{"x":1}\n\n')


def test_split_keys_and_prf_match_independent_hmac_construction() -> None:
    assert PUBLIC_SEED_ROOT == hashlib.sha256(
        PUBLIC_SEED_ROOT_PREIMAGE
    ).digest()
    expected_key = hmac.new(
        PUBLIC_SEED_ROOT,
        b"R12-ETTR-IL-v2\x00split-key\x00\x00\x01\x00development",
        hashlib.sha256,
    ).digest()
    key = derive_public_split_key(1, "development")
    assert key == expected_key
    label = b"candidate-rank"
    context = b"candidate-context"
    message = (
        b"R12-ETTR-IL-v2\x00seed\x00"
        + len(label).to_bytes(2, "big")
        + label
        + len(context).to_bytes(4, "big")
        + context
    )
    assert prf(key, label.decode("ascii"), context) == hmac.new(
        key,
        message,
        hashlib.sha256,
    ).digest()
    assert prf_stream_block(key, "candidate-rank", b"x", 4) == prf(
        key,
        "candidate-rank",
        b"x" + (4).to_bytes(8, "big"),
    )
    assert 0 <= opaque_seed(key, b"row") <= (1 << 63) - 1
    assert 0 <= prf_uniform_index(key, "query-choice", b"row", 17) < 17


def test_public_domains_are_isolated_and_confirmation_key_is_unavailable() -> None:
    keys = {
        (fold, split): derive_public_split_key(fold, split)
        for fold in range(3)
        for split in ("train", "development")
    }
    assert len(set(keys.values())) == 6
    key = keys[(0, "train")]
    outputs = {label: prf(key, label, b"same") for label in SEED_DOMAINS}
    assert len(set(outputs.values())) == len(SEED_DOMAINS)
    with pytest.raises(CustodyError, match="unrecognized"):
        derive_public_split_key(0, "confirmation")
    with pytest.raises(CustodyError, match="unrecognized"):
        prf(key, "new-domain", b"x")
    with pytest.raises(CustodyError, match="exactly 32"):
        prf(b"short", "candidate-rank", b"x")


def test_candidate_schema_round_trip_and_exact_ranking() -> None:
    candidate = _candidate(3)
    parsed = CandidateTuple.from_object(cj1_loads(candidate.canonical_bytes()))
    assert parsed == candidate
    key = derive_public_split_key(1, "train")
    assert candidate_rank(key, candidate) == prf(
        key,
        "candidate-rank",
        candidate.canonical_bytes(),
    )
    source = [_candidate(index) for index in range(8)]
    first = rank_candidates(source, key)
    second = rank_candidates(reversed(source), key)
    assert first == second
    assert tuple(item.rank for item in first) == tuple(
        sorted(item.rank for item in first)
    )
    selected = select_candidates(
        source,
        key,
        2,
        admissible=lambda item: item.generator_ordinal % 2 == 0,
    )
    expected = tuple(
        item.candidate
        for item in rank_candidates(source[::2], key)[:2]
    )
    assert selected == expected


def test_candidate_schema_and_quota_fail_closed() -> None:
    value = _candidate(0).to_object()
    value["extra"] = 1
    with pytest.raises(CustodyError, match="fields differ"):
        CandidateTuple.from_object(value)
    with pytest.raises(CustodyError, match="must be distinct"):
        replace(_candidate(0), worlds=(_digest("same"), _digest("same")))
    key = derive_public_split_key(1, "train")
    with pytest.raises(CustodyError, match="duplicate tuple"):
        rank_candidates([_candidate(0), _candidate(0)], key)
    with pytest.raises(CustodyError, match="duplicate tuple"):
        select_candidates(
            [_candidate(0), _candidate(0)],
            key,
            1,
            admissible=lambda _: False,
        )
    with pytest.raises(CustodyError, match="quota unavailable"):
        select_candidates([_candidate(0)], key, 2)
    with pytest.raises(CustodyError, match="did not return bool"):
        select_candidates([_candidate(0)], key, 1, admissible=lambda _: 1)


def test_fingerprint_preimages_bind_exact_semantics_and_tokens() -> None:
    world = b"(world x)"
    command = b"(write x y)"
    query = b"(query y)"
    expected_raw = cj1_dumps(
        {
            "command_hex": command.hex(),
            "query_hex": query.hex(),
            "world_hex": world.hex(),
        }
    )
    assert raw_row_fingerprint(world, command, query) == sha256_bytes(expected_raw)
    world_ast = {"nodes": [{"id": 0, "type": "entity"}]}
    theory_ast = {"rules": [["entity", "reachable"]]}
    command_ast = {"op": "write", "value": 7}
    world_hash = semantic_world_fingerprint(world_ast)
    assert world_hash == sha256_bytes(cj1_dumps(world_ast))
    assert theory_fingerprint(theory_ast) == sha256_bytes(cj1_dumps(theory_ast))
    assert semantic_command_fingerprint(command_ast) == sha256_bytes(
        cj1_dumps(command_ast)
    )
    assert bound_command_fingerprint(command_ast, world_hash) == sha256_bytes(
        cj1_dumps({"command": command_ast, "world_sha256": world_hash})
    )
    assert opaque_name_fingerprint("Opaque_symbol_0001") == sha256_bytes(
        b"Opaque_symbol_0001"
    )
    stage = b"WORLD\x00" + b"".join(
        value.to_bytes(4, "big") for value in (0, 7, 32767)
    )
    assert stage_token_sequence_preimage("WORLD", (0, 7, 32767)) == stage
    package = (
        stage_token_sequence_preimage("WORLD", (1,))
        + stage_token_sequence_preimage("COMMAND", (2,))
        + stage_token_sequence_preimage("QUERY", (3,))
    )
    assert package_token_sequence_fingerprint((1,), (2,), (3,)) == (
        sha256_bytes(package)
    )


def test_normalized_13grams_follow_literal_ascii_algorithm() -> None:
    payload = b"A,b__C!d e f g h i j k l m n O"
    assert normalized_13grams(payload) == {
        b"a b c d e f g h i j k l m",
        b"b c d e f g h i j k l m n",
        b"c d e f g h i j k l m n o",
    }
    world = b"a b c d e"
    command = b"f g h i"
    query = b"j k l m n"
    assert package_normalized_13grams(world, command, query) == {
        b"a b c d e f g h i j k l m",
        b"b c d e f g h i j k l m n",
    }
    with pytest.raises(CustodyError, match="fewer than 13"):
        normalized_13grams(b"too short")
    with pytest.raises(CustodyError, match="not ASCII"):
        normalized_13grams("é ".encode("utf-8") * 13)


def test_exact_graph_canonicalization_is_isomorphism_invariant() -> None:
    left = _graph((0, 1, 2))
    right = _graph((1, 0, 2))
    left_result = canonicalize_audit_graph(left, max_labelings=2)
    right_result = canonicalize_audit_graph(right, max_labelings=2)
    assert left_result.labeling_count == 2
    assert left_result.payload == right_result.payload
    assert graph_iso_fingerprint(left, max_labelings=2) == (
        graph_iso_fingerprint(right, max_labelings=2)
    )
    assert verify_color_preserving_bijection(left, right, (1, 0, 2))
    assert not verify_color_preserving_bijection(left, right, (0, 1, 2))
    parsed = cj1_loads(left_result.payload)
    assert parsed["node_colors"] == ["query", "term", "term"]
    assert len(parsed["edges"]) == 4


def test_graph_canonicalization_distinguishes_edges_and_fails_above_bound() -> None:
    graph = _graph()
    mutant = AuditGraph(
        nodes=graph.nodes,
        edges=tuple(
            sorted(
                (
                    *graph.edges[:-1],
                    AuditGraphEdge(src=2, dst=1, color="self"),
                )
            )
        ),
    )
    assert graph_iso_fingerprint(graph, max_labelings=2) != (
        graph_iso_fingerprint(mutant, max_labelings=2)
    )
    six = AuditGraph(
        nodes=tuple(AuditGraphNode(index, "same") for index in range(6)),
        edges=(),
    )
    with pytest.raises(CustodyError, match="exceeds resource bound"):
        canonicalize_audit_graph(six, max_labelings=719)


def test_graph_schema_rejects_extra_unsorted_and_invalid_records() -> None:
    value = _graph().to_object()
    value["extra"] = False
    with pytest.raises(CustodyError, match="fields differ"):
        AuditGraph.from_object(value)
    with pytest.raises(CustodyError, match="canonically sorted"):
        AuditGraph(nodes=_graph().nodes, edges=tuple(reversed(_graph().edges)))
    with pytest.raises(CustodyError, match="endpoint is absent"):
        AuditGraph(
            nodes=(AuditGraphNode(0, "x"),),
            edges=(AuditGraphEdge(0, 1, "edge"),),
        )
    assert _graph().schema == AUDIT_GRAPH_SCHEMA


def _source_entry(path: str, *, commit: str = "1" * 40) -> SourceInventoryEntry:
    payload = path.encode("ascii")
    return SourceInventoryEntry(
        commit=commit,
        path=path,
        git_mode="100644",
        git_blob_oid=hashlib.sha1(b"blob").hexdigest(),
        bytes=len(payload),
        sha256=sha256_bytes(payload),
        role="custody_source",
    )


def test_source_inventory_root_and_strict_schema() -> None:
    entries = (
        _source_entry("pipeline/a.py"),
        _source_entry("pipeline/b.py"),
    )
    root = source_entries_root(entries)
    runtime = (
        RuntimeInventoryEntry(
            logical_path="bin/python",
            bytes=10,
            sha256=_digest("python"),
            role="executable",
        ),
    )
    inventory = SourceInventory(
        protocol_spec_sha256=_digest("spec"),
        legacy_commit="2" * 40,
        legacy_tree="3" * 40,
        implementation_commit="4" * 40,
        implementation_tree="5" * 40,
        entries=entries,
        runtime_entries=runtime,
        inventory_sha256=root,
    )
    parsed = SourceInventory.from_object(
        json.loads(cj1_dumps(inventory.to_object()))
    )
    assert parsed == inventory
    assert inventory.inventory_sha256 == sha256_bytes(
        cj1_dumps([entry.to_object() for entry in entries])
    )
    hostile = inventory.to_object()
    hostile["unexpected"] = 1
    with pytest.raises(CustodyError, match="fields differ"):
        SourceInventory.from_object(hostile)
    with pytest.raises(CustodyError, match="differs from entries root"):
        replace(inventory, inventory_sha256="0" * 64)


def test_source_inventory_rejects_order_duplicates_links_and_bad_paths() -> None:
    first = _source_entry("pipeline/a.py")
    second = _source_entry("pipeline/b.py")
    with pytest.raises(CustodyError, match="not canonically sorted"):
        source_entries_root((second, first))
    with pytest.raises(CustodyError, match="duplicate"):
        source_entries_root((first, first))
    with pytest.raises(CustodyError, match="regular-file mode"):
        replace(first, git_mode="120000")
    with pytest.raises(CustodyError, match="not normalized"):
        replace(first, path="pipeline/../secret.py")


def test_file_records_and_roots_bind_sorted_literal_payloads() -> None:
    first_payload = b'{"a":1}\n'
    second_payload = b"opaque bytes"
    first = FileRecord.from_payload(
        path="assessor/a.jsonl",
        payload=first_payload,
        row_count=1,
        media_type="application/x-ndjson",
        confidentiality="assessor",
    )
    second = FileRecord.from_payload(
        path="candidate/b.bin",
        payload=second_payload,
        row_count=0,
        media_type="application/octet-stream",
        confidentiality="candidate",
    )
    first.verify_payload(first_payload)
    root = file_set_root((first, second))
    assert root == sha256_bytes(cj1_dumps([first.to_object(), second.to_object()]))
    assert FileRecord.from_object(first.to_object()) == first
    with pytest.raises(CustodyError, match="not sorted"):
        file_set_root((second, first))
    with pytest.raises(CustodyError, match="duplicate"):
        file_set_root((first, first))
    with pytest.raises(CustodyError, match="differs"):
        first.verify_payload(first_payload + b"x")


def test_fingerprint_root_is_order_independent_but_rejects_duplicates() -> None:
    left = [_digest("a"), _digest("b")]
    assert fingerprint_index_root(left) == fingerprint_index_root(reversed(left))
    with pytest.raises(CustodyError, match="duplicates"):
        fingerprint_index_root([left[0], left[0]])


def test_module_has_no_training_or_secret_side_effect_dependencies() -> None:
    source = __import__("inspect").getsource(
        __import__("ettr_il_v2_custody")
    )
    for forbidden in (
        "import torch",
        "import cryptography",
        "AESGCM",
        "subprocess",
        "os.environ",
        "confirmation_seed",
        "torch.load(",
    ):
        assert forbidden not in source
    assert PROTOCOL in source
