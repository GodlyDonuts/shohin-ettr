from __future__ import annotations

# ruff: noqa: E402

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from endogenous_typed_theory_reactor import TheoryReactorConfig
from ettr_checkpoint import (
    DataStreamState,
    EpisodeLifecycleState,
    TrainingProgress,
    _capture_rng_state,
    tree_sha256,
)
from ettr_data_contract import (
    ETTR_CONTINUATION_SCHEMA,
    ETTRContinuationManifest,
    ETTRPacketSufficiencyIndex,
    continuation_batch_payload_sha256,
)
from ettr_episode import ETTREpisodeSegment
from ettr_il_v2_controls import (  # noqa: E402
    BindingDerangement,
    DerangementAssignment,
    canonical_json_bytes as control_json_bytes,
)
from ettr_il_v2_materialize import (  # noqa: E402
    Disposition,
    GenericCell,
    GenericCommand,
    GenericCorner,
    GenericEdge,
    GenericInvariantPair,
    GenericMutation,
    GenericOperationTrace,
    GenericPacket,
    GenericQuery,
    GenericSemanticRectangle,
    GenericWorld,
    MaterializationRequest,
    Opcode,
    ValueRef,
    materialize_ettr_il_v2,
)
from ettr_il_v2_readiness import (
    ARCHITECTURE_PARAMETERS,
    ARM_MECHANISMS,
    ArmReadinessReceipt,
    BINDING_DERANGEMENT_ASSIGNMENTS,
    COMPLETE_SYSTEM_PARAMETERS,
    DATASET_INDEX_SCHEMA,
    ENCODED_POSITIONS_PER_UPDATE,
    ETTRILV2ReadinessError,
    ETTRILV2ReadinessRequest,
    OBJECTIVE_FAMILIES,
    PRIMARY_ARMS,
    PROTECTED_BASE_PARAMETERS,
    RESUME_CURSOR_SCHEMA,
    ResumeReadinessReceipt,
    SUPERVISED_POSITIONS_PER_UPDATE,
    objective_weights_sha256,
    validate_readiness,
)
from ettr_il_v2_schedule import (  # noqa: E402
    FIT_ONTOLOGIES,
    InvariantPairRecord,
    MODEL_SEEDS,
    build_pair_schedule,
)
from ettr_model_assembly import (
    ETTR_MODEL_ASSEMBLY_SCHEMA,
    ETTRModelAssemblyReceipt,
)
from ettr_objectives import ETTRObjectiveConfig
from ettr_train_step import ETTRTrainStepConfig
from model import GPTConfig


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object, *, newline: bool) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (payload + ("\n" if newline else "")).encode("ascii")


def _freeze(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    path.chmod(0o444)
    return _digest(payload)


class _Encoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _ByteTokenizer:
    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> _Encoding:
        assert add_special_tokens is False
        return _Encoding([ord(character) + 1 for character in text])


def _initial_packet(world: int) -> GenericPacket:
    return GenericPacket(
        cells=(
            GenericCell(0, 0, ValueRef.static(100)),
            GenericCell(1, 1, ValueRef.static(-5)),
            GenericCell(32, 4, ValueRef.local_id(world)),
            GenericCell(33, 4, ValueRef.local_id(3)),
        ),
        edges=(GenericEdge(8, 32, 33),),
        root=0,
    )


def _terminal_packet(world: int, command: int, result: int) -> GenericPacket:
    cells = list(_initial_packet(world).cells)
    cells[2] = GenericCell(32, 4, ValueRef.local_id(result))
    cells.extend(
        GenericCell(
            slot,
            5,
            (
                ValueRef.command_atom(10 + command)
                if slot == 48
                else ValueRef.empty()
            ),
        )
        for slot in range(48, 54)
    )
    cells.extend(
        (
            GenericCell(54, 6, ValueRef.small_uint(1)),
            GenericCell(55, 6, ValueRef.execute()),
        )
    )
    return GenericPacket(
        cells=tuple(cells),
        edges=(GenericEdge(8, 32, 33),),
        root=0,
        committed=True,
    )


def _rectangle(name: str, suffix: bytes) -> GenericSemanticRectangle:
    worlds = tuple(
        GenericWorld(
            (
                f"WORLD-{world}-A".encode("ascii") + suffix,
                f"WORLD-{world}-B".encode("ascii") + suffix,
            ),
            _initial_packet(world),
        )
        for world in range(2)
    )
    commands = tuple(
        GenericCommand(
            (
                f"COMMAND-{command}-A".encode("ascii") + suffix,
                f"COMMAND-{command}-B".encode("ascii") + suffix,
            ),
            (10 + command,),
        )
        for command in range(2)
    )
    corners: list[list[GenericCorner]] = [[], []]
    for world in range(2):
        for command in range(2):
            result = 8 + 2 * world + command
            corners[world].append(
                GenericCorner(
                    operation_traces=(
                        GenericOperationTrace(
                            mutations=(
                                GenericMutation(
                                    Opcode.WRITE,
                                    source=32,
                                    value=ValueRef.local_id(result),
                                ),
                            ),
                            cursor=1,
                        ),
                    ),
                    terminal_packet=_terminal_packet(world, command, result),
                    disposition=Disposition.ANSWER,
                    outcome=ValueRef.execute(),
                    answers=(
                        bool(world ^ command),
                        not bool(world ^ command),
                    ),
                )
            )
    return GenericSemanticRectangle(
        semantic_rectangle_id=_digest(name.encode("ascii")),
        presentation_id=f"presentation-{name}",
        worlds=(worlds[0], worlds[1]),
        commands=(commands[0], commands[1]),
        queries=(
            GenericQuery(
                (
                    b"Is alpha true? " + suffix,
                    b"Alpha verdict: " + suffix,
                )
            ),
            GenericQuery(
                (
                    b"Is beta true? " + suffix,
                    b"Beta verdict: " + suffix,
                )
            ),
        ),
        corners=(
            (corners[0][0], corners[0][1]),
            (corners[1][0], corners[1][1]),
        ),
    )


def _full_transport(batch):
    def segment(value: ETTREpisodeSegment) -> ETTREpisodeSegment:
        return ETTREpisodeSegment.from_tokens(
            value.tokens,
            attention_mask=torch.ones_like(
                value.attention_mask,
                dtype=torch.bool,
            ),
        )

    return replace(
        batch,
        episodes=replace(
            batch.episodes,
            world=segment(batch.episodes.world),
            command=segment(batch.episodes.command),
            query=segment(batch.episodes.query),
        ),
    )


def _batch(index: int):
    left = _rectangle(f"batch-{index}-left", f"-{index}-L".encode("ascii"))
    right = _rectangle(f"batch-{index}-right", f"-{index}-R".encode("ascii"))
    batch = materialize_ettr_il_v2(
        MaterializationRequest(
            manifest_sha256="a" * 64,
            dataset_sha256="b" * 64,
            vocab_size=512,
            rectangles=(left, right),
            invariant_pairs=(GenericInvariantPair(0, 1),),
        ),
        _ByteTokenizer(),
    )
    return _full_transport(batch)


def _schedule():
    records = []
    fold = 0
    for ontology in FIT_ONTOLOGIES[fold]:
        for depth in (1, 2, 3):
            for core in range(96):
                core_id = _digest(
                    f"{ontology}-{depth}-{core}".encode("ascii")
                )
                for pair in range(2):
                    records.append(
                        InvariantPairRecord(
                            pair_id=_digest(
                                f"pair-{ontology}-{depth}-{core}-{pair}".encode(
                                    "ascii"
                                )
                            ),
                            semantic_core_id=core_id,
                            ontology=ontology,
                            depth=depth,
                            left_semantic_rectangle_id=_digest(
                                f"left-{ontology}-{depth}-{core}-{pair}".encode(
                                    "ascii"
                                )
                            ),
                            right_semantic_rectangle_id=_digest(
                                f"right-{ontology}-{depth}-{core}-{pair}".encode(
                                    "ascii"
                                )
                            ),
                        )
                    )
    return build_pair_schedule(
        records,
        fold=fold,
        seed=MODEL_SEEDS[0],
    )


def _derangement() -> BindingDerangement:
    recipient_ids = tuple(
        _digest(f"recipient-{index}".encode("ascii"))
        for index in range(BINDING_DERANGEMENT_ASSIGNMENTS)
    )
    assignments = tuple(
        DerangementAssignment(
            recipient_id=recipient,
            donor_id=recipient_ids[(index + 1) % len(recipient_ids)],
            donor_rank=0,
            donor_digest=_digest(
                f"donor-{index}".encode("ascii")
            ),
        )
        for index, recipient in enumerate(recipient_ids)
    )
    assignment_sha256 = _digest(
        control_json_bytes([value.as_dict() for value in assignments])
    )
    return BindingDerangement(
        fold=0,
        assignments=assignments,
        assignment_sha256=assignment_sha256,
    )


def _write_request(tmp_path: Path) -> ETTRILV2ReadinessRequest:
    train = tuple(_batch(index) for index in range(4))
    validation = _batch(99)
    sufficiency = ETTRPacketSufficiencyIndex.from_splits(
        train,
        (validation,),
    )
    packet_receipt = sufficiency.receipt
    manifest = ETTRContinuationManifest(
        schema=ETTR_CONTINUATION_SCHEMA,
        protected_checkpoint_sha256="1" * 64,
        tokenizer_sha256="2" * 64,
        qualification_payload_sha256="3" * 64,
        hybrid_payload_sha256="4" * 64,
        train_rows=128,
        validation_rows=32,
        train_payload_sha256=sufficiency.train_payload_sha256,
        validation_payload_sha256=sufficiency.validation_payload_sha256,
        dataset_sha256=ETTRContinuationManifest.combined_dataset_sha256(
            sufficiency.train_payload_sha256,
            sufficiency.validation_payload_sha256,
        ),
        packet_sufficiency_train_batches=4,
        packet_sufficiency_validation_batches=1,
        packet_sufficiency_rows=packet_receipt.rows,
        packet_sufficiency_unique_contexts=packet_receipt.unique_contexts,
        packet_sufficiency_train_contexts=sufficiency.train_contexts,
        packet_sufficiency_validation_contexts=(
            sufficiency.validation_contexts
        ),
        packet_sufficiency_context_sha256=packet_receipt.context_sha256,
        packet_sufficiency_target_bound_sha256=(
            packet_receipt.target_bound_sha256
        ),
        source_deleted=True,
        immutable_snapshot=True,
        live_writer_input=False,
        family_label_fields=(),
    )
    manifest_sha256 = manifest.sha256()
    train = tuple(
        replace(
            batch,
            manifest_sha256=manifest_sha256,
            dataset_sha256=manifest.dataset_sha256,
        )
        for batch in train
    )
    validation = replace(
        validation,
        manifest_sha256=manifest_sha256,
        dataset_sha256=manifest.dataset_sha256,
    )
    manifest_path = tmp_path / "manifest.json"
    assert _freeze(
        manifest_path,
        _canonical(asdict(manifest), newline=False),
    ) == manifest_sha256
    dataset_path = tmp_path / "dataset.index"
    train_batch_payload_sha256s = sorted(
        continuation_batch_payload_sha256(batch) for batch in train
    )
    validation_batch_payload_sha256s = [
        continuation_batch_payload_sha256(validation)
    ]
    dataset_payload = _canonical(
        {
            "dataset_sha256": manifest.dataset_sha256,
            "manifest_sha256": manifest_sha256,
            "schema": DATASET_INDEX_SCHEMA,
            "train_batch_payload_sha256s": train_batch_payload_sha256s,
            "train_payload_sha256": manifest.train_payload_sha256,
            "validation_batch_payload_sha256s": (
                validation_batch_payload_sha256s
            ),
            "validation_payload_sha256": (
                manifest.validation_payload_sha256
            ),
        },
        newline=True,
    )
    dataset_artifact_sha256 = _freeze(dataset_path, dataset_payload)

    reactor_config = TheoryReactorConfig()
    config_path = tmp_path / "reactor.json"
    config_sha256 = _freeze(
        config_path,
        _canonical(asdict(reactor_config), newline=True),
    )
    assembly = ETTRModelAssemblyReceipt(
        schema=ETTR_MODEL_ASSEMBLY_SCHEMA,
        config_sha256=config_sha256,
        checkpoint_sha256=manifest.protected_checkpoint_sha256,
        checkpoint_step=300_000,
        compiler_sha256="5" * 64,
        reactor_sha256="6" * 64,
        query_reader_sha256="7" * 64,
        complete_model_sha256="8" * 64,
        base_parameters=PROTECTED_BASE_PARAMETERS,
        architecture_parameters=ARCHITECTURE_PARAMETERS,
        total_parameters=COMPLETE_SYSTEM_PARAMETERS,
        parameter_cap=200_000_000,
        remaining_under_cap=200_000_000 - COMPLETE_SYSTEM_PARAMETERS,
    )
    assembly_path = tmp_path / "assembly.json"
    assembly_sha256 = _freeze(assembly_path, assembly.canonical_bytes())

    schedule = _schedule()
    pair_ids = tuple(
        value.pair_id for value in schedule.exposures[:4]
    )
    derangement = _derangement()
    arms = tuple(
        ArmReadinessReceipt(
            arm=arm,
            mechanism=ARM_MECHANISMS[arm],
            arm_config_sha256=_digest(f"config-{arm}".encode("ascii")),
            control_receipt_sha256=(
                derangement.assignment_sha256
                if arm == "binding_deranged"
                else _digest(f"control-{arm}".encode("ascii"))
            ),
            manifest_sha256=manifest_sha256,
            dataset_sha256=manifest.dataset_sha256,
            schedule_sha256=schedule.schedule_sha256,
            source_payload_sha256=manifest.train_payload_sha256,
            objective_weights_sha256=objective_weights_sha256(),
            objective_families=OBJECTIVE_FAMILIES,
            microsteps_per_update=4,
            rows_per_microstep=32,
            rows_per_update=128,
            semantic_rectangles_per_update=8,
            causal_rectangles_per_update=32,
            encoded_positions_per_update=ENCODED_POSITIONS_PER_UPDATE,
            supervised_positions_per_update=(
                SUPERVISED_POSITIONS_PER_UPDATE
            ),
            updates=6_000,
            trainable_parameters=ARCHITECTURE_PARAMETERS,
            complete_system_parameters=COMPLETE_SYSTEM_PARAMETERS,
        )
        for arm in PRIMARY_ARMS
    )
    rng_state = _capture_rng_state()
    resume = ResumeReadinessReceipt(
        progress=TrainingProgress(
            global_step=300_000,
            optimizer_step=0,
            micro_step=0,
            gradient_accumulation_steps=4,
            tokens_seen=0,
        ),
        data_stream=DataStreamState(
            manifest_sha256=manifest_sha256,
            dataset_sha256=manifest.dataset_sha256,
            generation=0,
            seed=schedule.seed,
            epoch=0,
            shard_index=0,
            sample_index=0,
            token_offset=0,
            sampler_state={
                "next_pair_ids": list(pair_ids),
                "pair_exposure_index": 0,
                "schedule_sha256": schedule.schedule_sha256,
                "schema": RESUME_CURSOR_SCHEMA,
            },
        ),
        episode_lifecycle=EpisodeLifecycleState(
            episode_index=0,
            phase="between_episodes",
            episode_sha256=None,
            token_offset=0,
            reactor_step=0,
            source_deleted=False,
            committed=False,
            halted=False,
        ),
        rng_state=rng_state,
        rng_state_sha256=tree_sha256(rng_state),
    )
    return ETTRILV2ReadinessRequest(
        manifest_path=manifest_path,
        expected_manifest_sha256=manifest_sha256,
        dataset_path=dataset_path,
        expected_dataset_artifact_sha256=dataset_artifact_sha256,
        model_assembly_receipt_path=assembly_path,
        expected_model_assembly_receipt_sha256=assembly_sha256,
        reactor_config_path=config_path,
        expected_reactor_config_sha256=config_sha256,
        base_config=GPTConfig(),
        objective_config=ETTRObjectiveConfig(vocab_size=32_768),
        step_config=ETTRTrainStepConfig(
            gradient_accumulation_steps=4,
        ),
        packet_sufficiency=sufficiency,
        update_batches=train,
        update_pair_ids=pair_ids,
        schedule=schedule,
        arms=arms,
        binding_derangement=derangement,
        resume=resume,
        evaluator_batch=validation,
    )


def test_readiness_passes_without_weight_updates(tmp_path: Path) -> None:
    request = _write_request(tmp_path)
    before = tree_sha256(_capture_rng_state())
    report = validate_readiness(request)
    after = tree_sha256(_capture_rng_state())
    assert report.status == "pass"
    assert report.mode == "validate_only"
    assert report.weight_updates == 0
    assert report.update_rows == 128
    assert (
        report.supervised_positions_per_update
        == SUPERVISED_POSITIONS_PER_UPDATE
    )
    assert report.encoded_positions_per_update == ENCODED_POSITIONS_PER_UPDATE
    assert report.arm_count == 5
    assert report.evaluator_query_shape == (32, 48)
    assert report.complete_system_parameters == COMPLETE_SYSTEM_PARAMETERS
    assert before == after


def test_unequal_arm_supervision_fails_closed(tmp_path: Path) -> None:
    request = _write_request(tmp_path)
    arms = list(request.arms)
    arms[1] = replace(
        arms[1],
        supervised_positions_per_update=(
            SUPERVISED_POSITIONS_PER_UPDATE - 1
        ),
    )
    with pytest.raises(
        ETTRILV2ReadinessError,
        match="matched arm budget|arm contract",
    ):
        validate_readiness(replace(request, arms=tuple(arms)))


def test_resume_rng_or_pair_cursor_drift_fails_closed(tmp_path: Path) -> None:
    request = _write_request(tmp_path)
    bad_rng = replace(
        request.resume,
        rng_state_sha256="f" * 64,
    )
    with pytest.raises(ETTRILV2ReadinessError, match="RNG receipt"):
        validate_readiness(replace(request, resume=bad_rng))
    pair_ids = list(request.update_pair_ids)
    pair_ids[0], pair_ids[1] = pair_ids[1], pair_ids[0]
    with pytest.raises(
        ETTRILV2ReadinessError,
        match="resume cursor|deterministic pair cursor",
    ):
        validate_readiness(
            replace(request, update_pair_ids=tuple(pair_ids))
        )


def test_writable_manifest_and_evaluator_shape_fail_closed(
    tmp_path: Path,
) -> None:
    request = _write_request(tmp_path)
    request.manifest_path.chmod(0o644)
    with pytest.raises(ETTRILV2ReadinessError, match="immutable"):
        validate_readiness(request)
    request.manifest_path.chmod(0o444)
    query = request.evaluator_batch.episodes.query
    short_query = ETTREpisodeSegment.from_tokens(
        query.tokens[:, :-1],
        attention_mask=query.attention_mask[:, :-1],
    )
    evaluator = replace(
        request.evaluator_batch,
        episodes=replace(
            request.evaluator_batch.episodes,
            query=short_query,
        ),
    )
    with pytest.raises(
        ETTRILV2ReadinessError,
        match="input shape",
    ):
        validate_readiness(
            replace(request, evaluator_batch=evaluator)
        )


def test_internally_valid_wrong_model_cap_fails_closed(
    tmp_path: Path,
) -> None:
    request = _write_request(tmp_path)
    original = ETTRModelAssemblyReceipt.from_path(
        request.model_assembly_receipt_path
    )
    wrong_cap = 199_000_000
    altered = replace(
        original,
        parameter_cap=wrong_cap,
        remaining_under_cap=wrong_cap - original.total_parameters,
    )
    altered_path = tmp_path / "wrong-cap-assembly.json"
    altered_sha256 = _freeze(altered_path, altered.canonical_bytes())
    with pytest.raises(
        ETTRILV2ReadinessError,
        match="parameter cap",
    ):
        validate_readiness(
            replace(
                request,
                model_assembly_receipt_path=altered_path,
                expected_model_assembly_receipt_sha256=altered_sha256,
            )
        )


def test_sidecar_source_has_no_update_or_launch_surface() -> None:
    source = (ROOT / "train" / "ettr_il_v2_readiness.py").read_text(
        encoding="ascii"
    )
    forbidden = (
        ".backward(",
        "optimizer.step(",
        "ETTROptimizerBundle(",
        "save_ettr_checkpoint(",
        "subprocess.",
        "sbatch",
    )
    assert not any(value in source for value in forbidden)
