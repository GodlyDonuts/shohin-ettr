"""Adversarial RED tests for the unlaunched Track-S development custody G."""

import hashlib
import inspect
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from pipeline import adjudicate_acw_hidden_basis as adjudicator
from pipeline import acw_immutable_publication as publication
from pipeline import acw_hidden_basis_training as trainer
from pipeline import build_acw_development_manifest as builder
from pipeline import evaluate_acw_hidden_basis as evaluator
from pipeline import freeze_acw_curriculum as freezer
from pipeline import generate_acw_hidden_basis as generator
from pipeline.freeze_acw_curriculum import (
    CANONICAL_PILOT_STATIC_ENV,
    CANONICAL_PILOT_UID,
)


REPOSITORY = Path(__file__).resolve().parents[1]
PLAN_PATH = REPOSITORY / "R12_ACW_DEVELOPMENT_PLAN_V1.json"
JOB_PATH = REPOSITORY / "pipeline/jobs/run_acw_development_stokes.sbatch"
MONITOR_PATH = REPOSITORY / "pipeline/jobs/run_acw_terminal_monitor_stokes.sbatch"
STAGE_ROLES = (
    "phase1_producer",
    "phase1_verifier",
    "phase2_producer",
    "phase2_verifier",
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _attempt_ids() -> list[str]:
    return [
        f"{arm}__{seed}"
        for arm in (adjudicator.DIRECT_STATE_ARM, *adjudicator.SCORED_ARMS)
        for seed in adjudicator.DEVELOPMENT_SEEDS
    ]


class GFourStageProtocolTests(unittest.TestCase):
    def test_plan_freezes_four_distinct_top_level_jobs_and_exact_attempts(self) -> None:
        plan = json.loads(PLAN_PATH.read_bytes())
        stages = plan.get("custody_stages")
        self.assertIsInstance(stages, list, "G must freeze four top-level stages")
        self.assertEqual(
            [stage.get("role") for stage in stages],
            list(STAGE_ROLES),
        )
        job_ids = [str(stage.get("held_slurm_job_id", "")) for stage in stages]
        self.assertTrue(all(job_id.isdigit() for job_id in job_ids))
        self.assertEqual(len(set(job_ids)), 4)
        self.assertEqual(
            [stage.get("expected_node") for stage in stages],
            ["ec51", "ec52", "ec51", "ec52"],
        )
        self.assertNotEqual(job_ids[0], job_ids[1])
        self.assertNotEqual(job_ids[2], job_ids[3])
        self.assertEqual(
            plan.get("attempt_registry", {}).get("attempt_ids"), _attempt_ids()
        )

    def test_stage_jobs_use_batch_cgroups_and_never_normal_srun_steps(self) -> None:
        script = JOB_PATH.read_text()
        self.assertRegex(script, r"(?m)^#SBATCH --ntasks=1$")
        self.assertIsNone(
            re.search(r"(?:^|\s)(?:/\S+)?srun(?:\s|$)", script),
            "a normal srun step is not an independent Slurm job",
        )
        for role in STAGE_ROLES:
            self.assertIn(role, script)
        self.assertIn("step_batch/task_0", script)

    def test_plan_binds_fixed_reviewed_branch_without_main_or_caller_override(
        self,
    ) -> None:
        plan = json.loads(PLAN_PATH.read_bytes())
        self.assertEqual(
            plan["reviewed_branch_contract"], builder.REVIEWED_BRANCH_CONTRACT
        )
        self.assertTrue(plan["reviewed_branch_contract"]["history_rewrite_forbidden"])
        self.assertTrue(
            plan["reviewed_branch_contract"]["caller_selected_branch_forbidden"]
        )
        for path in (JOB_PATH, MONITOR_PATH):
            source = path.read_text()
            self.assertIn("refs/heads/codex/acw-g2", source)
            self.assertIn("refs/remotes/origin/codex/acw-g2", source)
            self.assertNotIn("refs/heads/main", source)

    def test_plan_is_explicit_no_go_and_both_held_chains_are_nonreleasable(
        self,
    ) -> None:
        plan = json.loads(PLAN_PATH.read_bytes())
        self.assertFalse(plan["ready_for_g_commit"])
        self.assertEqual(plan["review_disposition"], builder.REVIEW_DISPOSITION)
        self.assertEqual(
            plan["review_disposition"]["held_job_ids_must_never_release"],
            [str(job_id) for job_id in range(741065, 741075)],
        )
        self.assertEqual(
            {stage["script"]["sha256"] for stage in plan["custody_stages"]},
            {builder.QUARANTINED_STAGE_SCRIPT_SHA256},
        )
        self.assertEqual(
            _sha256(JOB_PATH.read_bytes()),
            builder.REPAIRED_STAGE_SCRIPT_SHA256,
        )
        self.assertNotEqual(
            builder.QUARANTINED_STAGE_SCRIPT_SHA256,
            builder.REPAIRED_STAGE_SCRIPT_SHA256,
        )
        builder.validate_plan(plan, require_ready=False)
        with self.assertRaisesRegex(ValueError, "NO-GO pending review"):
            builder.validate_plan(plan, require_ready=True)

    def test_both_verifier_argv_produce_adjudicator_accepted_receipts(self) -> None:
        plan = json.loads(PLAN_PATH.read_bytes())
        stage_by_role = {stage["role"]: stage for stage in plan["custody_stages"]}
        examples = {
            builder.ROLE_DIRECT_VERIFIER: plan["attempt_table"][0],
            builder.ROLE_FINAL_VERIFIER: plan["attempt_table"][3],
        }
        for attempt in plan["attempt_table"]:
            self.assertNotIn("--verification-replay", attempt["producer"]["train_argv"])
            self.assertEqual(
                attempt["verifier"]["train_argv"].count("--verification-replay"),
                1,
            )

        for role, attempt in examples.items():
            with self.subTest(role=role):
                side = attempt["verifier"]
                args = trainer.build_parser().parse_args(side["train_argv"][4:])
                self.assertTrue(args.verification_replay)
                stage = stage_by_role[role]
                environment = {
                    **CANONICAL_PILOT_STATIC_ENV,
                    "SLURM_CPUS_PER_TASK": "4",
                    "SLURM_JOB_ID": str(stage["held_slurm_job_id"]),
                    "SLURM_JOB_NAME": stage["job_name"],
                    "SLURM_JOB_NODELIST": stage["expected_node"],
                    "SLURM_NODELIST": stage["expected_node"],
                    "SLURM_SUBMIT_DIR": str(builder.BASE),
                }
                job_id = str(stage["held_slurm_job_id"])
                receipt = trainer._development_execution_receipt(
                    scientific_commit="d" * 40,
                    runtime_sha256=trainer.CANONICAL_DEVELOPMENT_RUNTIME_SHA256,
                    canonical_environment=environment,
                    batch_script_sha256=_sha256(JOB_PATH.read_bytes()),
                    held_job_id=job_id,
                    job_name=stage["job_name"],
                    node_list=stage["expected_node"],
                    cpus_per_task="4",
                    membership={
                        "cpu_list": "0-3",
                        "memory_list": "0",
                        "task_cgroup": (
                            f"/slurm/uid_{CANONICAL_PILOT_UID}/job_{job_id}"
                            "/step_batch/task_0"
                        ),
                    },
                    role=role,
                    attempt_id=attempt["attempt_id"],
                    verification_replay=args.verification_replay,
                )
                accepted = adjudicator._validate_development_execution_receipt(
                    receipt,
                    label=f"{role} receipt",
                )
                self.assertEqual(accepted["role"], role)
                self.assertTrue(accepted["verification_replay"])

    def test_every_g2_scientific_publisher_uses_the_durable_shared_contract(
        self,
    ) -> None:
        tree_publishers = (
            generator.generate_dataset,
            freezer.build_trainer_bundle,
        )
        file_publishers = (
            trainer.write_checkpoint,
            evaluator.write_report,
            adjudicator._write_immutable_binary,
            adjudicator.write_immutable_json,
            builder._atomic_publish_bytes,
        )
        for function in tree_publishers:
            source = inspect.getsource(function)
            self.assertIn("publish_tree_no_replace", source)
            self.assertNotIn(".replace(", source)
        for function in file_publishers:
            source = inspect.getsource(function)
            self.assertIn("publish_bytes_no_replace", source)
            self.assertNotIn(".replace(", source)

        source = inspect.getsource(publication)
        for required in (
            "tempfile.mkdtemp",
            "tempfile.mkstemp",
            "os.fsync",
            "rename_no_replace",
            "fsync_directory",
            "_descriptor_record",
            '"st_dev"',
            '"st_ino"',
        ):
            self.assertIn(required, source)

    def test_wrapper_publishes_plan_through_shared_no_replace_contract(self) -> None:
        script = JOB_PATH.read_text()
        phase1 = script.split("phase1_producer)", 1)[1].split(";;", 1)[0]
        self.assertIn('builder publish-plan --root "$ROOT"', phase1)
        self.assertNotRegex(
            phase1,
            r'/usr/bin/install\s+-m\s+0444\s+"\$PLAN"',
        )
        source = inspect.getsource(builder.publish_development_plan_copy)
        self.assertIn("_load_committed_plan_with_raw(require_ready=True)", source)
        self.assertIn("_atomic_publish_bytes", source)

    def test_verifier_layout_is_durable_before_any_stage_publication(self) -> None:
        script = JOB_PATH.read_text()
        for role, predecessor in (
            ("phase1_verifier", "phase1_producer"),
            ("phase2_verifier", "phase2_producer"),
        ):
            branch = script.split(f"\n  {role})", 1)[1].split(";;", 1)[0]
            layout = 'builder verify-layout --root "$ROOT" --role "$ROLE"'
            accounting = f"record_predecessor_accounting {predecessor}"
            stage_start = 'builder stage-start --root "$ROOT" --role "$ROLE"'
            self.assertLess(
                branch.index("/usr/bin/install -d -m 700"), branch.index(layout)
            )
            self.assertLess(branch.index(layout), branch.index(accounting))
            self.assertLess(branch.index(layout), branch.index(stage_start))

    def test_generated_tree_modes_are_durably_finalized_before_consumption(
        self,
    ) -> None:
        publication_source = inspect.getsource(publication.publish_tree_no_replace)
        self.assertIn(
            "TREE_PUBLICATION_RECEIPT_FD_ENV in os.environ", publication_source
        )
        self.assertIn("file_mode = 0o444", publication_source)
        self.assertIn("directory_mode = 0o555", publication_source)
        self.assertLess(
            publication_source.index("_sync_tree("),
            publication_source.index("rename_no_replace"),
        )
        self.assertIn("retained_evidence_snapshot", publication_source)
        self.assertLess(
            publication_source.index("fsync_directory"),
            publication_source.index("_retain_complete_tree"),
        )
        durable_freeze_source = inspect.getsource(publication.freeze_tree)
        self.assertIn("retained_evidence_snapshot", durable_freeze_source)
        self.assertIn("retained.retain_tree", durable_freeze_source)
        self.assertIn("retained.refresh_tree", durable_freeze_source)
        run_inputs_source = inspect.getsource(builder.run_inputs)
        self.assertIn("expected_tree_publication", run_inputs_source)
        self.assertIn("publication.retained_evidence_snapshot", run_inputs_source)
        self.assertGreaterEqual(run_inputs_source.count("retained.retain_tree"), 3)
        self.assertNotIn("publication.freeze_tree", run_inputs_source)
        self.assertNotIn(".chmod(", run_inputs_source)
        freeze_source = inspect.getsource(builder._freeze_tree)
        self.assertIn("publication.freeze_tree", freeze_source)
        self.assertNotIn(".chmod(", freeze_source)

        attempt_source = inspect.getsource(builder.run_attempt)
        self.assertGreaterEqual(
            attempt_source.count("publication.freeze_tree(task)"), 2
        )
        self.assertNotIn("task.chmod", attempt_source)

    def test_terminal_monitor_is_a_fifth_held_retry_safe_job(self) -> None:
        plan = json.loads(PLAN_PATH.read_bytes())
        monitor = plan["accounting"]["monitor_stage"]
        scientific_job_ids = {
            str(stage["held_slurm_job_id"]) for stage in plan["custody_stages"]
        }
        self.assertNotIn(str(monitor["held_slurm_job_id"]), scientific_job_ids)
        self.assertEqual(
            monitor["dependency"],
            {
                "type": "afterok",
                "held_slurm_job_id": plan["custody_stages"][-1]["held_slurm_job_id"],
            },
        )
        script = MONITOR_PATH.read_text()
        self.assertIn("terminal-accounting --root", script)
        self.assertIn("verify-terminal-accounting --root", script)
        self.assertIn("retrying resumable terminal accounting", script)
        self.assertIn("retrying terminal verification", script)
        self.assertIsNone(re.search(r"(?:^|\s)(?:/\S+)?srun(?:\s|$)", script))
        self.assertIn("step_batch/task_0", script)

        verification_source = inspect.getsource(
            builder.publish_terminal_verification_attestation
        )
        self.assertIn("publish_tree_no_replace", verification_source)
        self.assertIn("file_mode=0o444", verification_source)
        self.assertIn("directory_mode=0o555", verification_source)
        self.assertIn("verify_frozen_tree", verification_source)
        self.assertNotIn("freeze_tree", verification_source)
        main_source = inspect.getsource(builder.main)
        verify_branch = main_source.split(
            'if args.kind == "verify-terminal-accounting":', 1
        )[1].split("return", 1)[0]
        self.assertLess(
            verify_branch.index("publish_terminal_verification_attestation"),
            verify_branch.index("[acw-development-terminal-verify]"),
        )
        self.assertIn("verification_sha256", verify_branch)
        anchor_source = inspect.getsource(builder.build_monitor_anchor_ready_envelope)
        self.assertIn("externally_anchored_verification_sha256", anchor_source)
        self.assertIn("external terminal verification anchor differs", anchor_source)
        publication_source = inspect.getsource(
            builder.publish_monitor_anchor_ready_envelope
        )
        self.assertIn("retained_evidence_snapshot", publication_source)
        self.assertGreaterEqual(publication_source.count("retained.retain_file"), 5)
        self.assertIn("retained.retain_directory", publication_source)
        self.assertIn("_retain_terminal_closed_world_from_receipt", publication_source)
        self.assertIn("retained_bytes_publication", publication_source)
        self.assertLess(
            publication_source.index("retained_evidence_snapshot"),
            publication_source.index("retained_bytes_publication"),
        )

    def test_final_worker_seals_evidence_without_publishing_metrics(self) -> None:
        script = JOB_PATH.read_text()
        final_role = script.rsplit("phase2_verifier)", 1)[1].split(";;", 1)[0]
        self.assertIn("builder development --root", final_role)
        self.assertIn("builder stage-complete", final_role)
        self.assertNotIn("development_baseline", final_role)
        self.assertNotIn("best_development_checkpoint", final_role)
        self.assertNotIn("adjudicate", final_role)

    def test_plan_narrows_same_uid_and_external_anchor_claims(self) -> None:
        plan = json.loads(PLAN_PATH.read_bytes())
        closed_world = plan["closed_world"]
        self.assertTrue(closed_world["exact_registered_root_required"])
        self.assertTrue(closed_world["same_uid_external_compute_not_excluded"])
        self.assertTrue(
            closed_world["ordinary_batch_children_not_independently_attested"]
        )
        self.assertTrue(
            plan["accounting"][
                "external_sha256_anchor_required_before_performance_claim"
            ]
        )
        self.assertTrue(
            plan["accounting"]["monitor_must_be_completed_before_anchor_ready_envelope"]
        )
        self.assertEqual(
            plan["accounting"]["monitor_anchor_ready_envelope"],
            str(builder.MONITOR_ANCHOR_PATH),
        )
        self.assertFalse(plan["accounting"]["performance_claim_ready_at_g_launch"])

    def test_manifests_require_external_stage_and_private_refit_receipts(self) -> None:
        verifier_source = inspect.getsource(
            adjudicator._verify_evidence_with_private_workspace
        )
        for required in (
            "attempt_claim",
            "stage_receipts",
            "private_refit_verification",
        ):
            self.assertIn(f'"{required}"', verifier_source)
        self.assertIn("_validate_private_refit_verification_reference", verifier_source)
        self.assertIn(
            "stage_receipts", inspect.getsource(builder.build_direct_manifest)
        )
        self.assertIn(
            "stage_receipts", inspect.getsource(builder.build_development_manifest)
        )


class GDataAttackTests(unittest.TestCase):
    def _dataset_manifest(self, *, fingerprint: str) -> dict:
        arrays = {
            relative: {
                "bytes": 1,
                "dtype": "uint8",
                "shape": [1],
                "sha256": _sha256(relative.encode("ascii")),
            }
            for relative in sorted(adjudicator._required_dataset_arrays())
        }
        return adjudicator.with_payload_hash(
            {
                "protocol": adjudicator.GENERATOR_PROTOCOL,
                "seed_identity": {
                    "kind": "development",
                    "seed": adjudicator.DEVELOPMENT_SEEDS[0],
                },
                "seed_fingerprint": fingerprint,
                "field_size": 17,
                "dimension": 3,
                "source_dim": 96,
                "event_dim": 96,
                "event_count": 48,
                "event_address_counts": {"0": 16, "1": 16, "2": 16},
                "public_queries": 24,
                "new_queries": 8,
                "counts": {
                    "train": 4096,
                    "adaptation": 1024,
                    "evaluation_per_depth": 2048,
                },
                "evaluation_depths": list(adjudicator.EVALUATION_DEPTHS),
                "visited_buckets": {},
                "depth_counts": {},
                "arrays": arrays,
            }
        )

    def test_registered_seed_rejects_attacker_selected_fingerprint(self) -> None:
        seed = adjudicator.DEVELOPMENT_SEEDS[0]
        expected = _sha256(generator.development_seed_material(seed))
        attacker = _sha256(b"same seed identity, relabeled hidden coordinates")
        self.assertNotEqual(attacker, expected)
        manifest = self._dataset_manifest(fingerprint=attacker)
        with self.assertRaisesRegex(
            adjudicator.EvidenceError,
            "seed.*fingerprint|fingerprint.*seed|deterministic.*replay|dataset.*replay",
        ):
            adjudicator._validate_dataset_manifest(manifest, "relabel_attack")

    def test_rounds_1_through_12_answers_are_replayed_against_oracle_truth(
        self,
    ) -> None:
        histories = 4096
        oracle_answers = np.zeros((histories, 24), dtype=np.int8)
        initial_queries = np.tile(np.array([[0, 1]], dtype=np.int8), (histories, 1))
        initial_answers = np.zeros((histories, 2), dtype=np.int8)
        rows = []
        for history_id in range(histories):
            rows.extend(
                (
                    {"history_id": history_id, "query_id": 0, "answer": 0, "round": 0},
                    {"history_id": history_id, "query_id": 1, "answer": 0, "round": 0},
                )
            )
            for round_index in range(1, 13):
                rows.append(
                    {
                        "history_id": history_id,
                        "query_id": round_index + 1,
                        "answer": 0,
                        "round": round_index,
                    }
                )
        rows[2]["answer"] = 1
        raw = b"".join(adjudicator.canonical_json_bytes(row) + b"\n" for row in rows)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "curriculum.jsonl").write_bytes(raw)
            files = {
                "curriculum.jsonl": {
                    "bytes": len(raw),
                    "rows": len(rows),
                    "sha256": _sha256(raw),
                }
            }
            kwargs = {}
            if (
                "oracle_answers"
                in inspect.signature(adjudicator._validate_curriculum).parameters
            ):
                kwargs["oracle_answers"] = oracle_answers
            with self.assertRaisesRegex(
                adjudicator.EvidenceError,
                "curriculum.*answer|answer.*replay|oracle",
            ):
                adjudicator._validate_curriculum(
                    root,
                    files,
                    initial_queries,
                    initial_answers,
                    "poisoned_rounds",
                    **kwargs,
                )


class GReceiptAttackTests(unittest.TestCase):
    def _producer_receipt(self, *, task_cgroup: str) -> tuple[dict, bytes, str]:
        job_id = "740999"
        role = builder.ROLE_PHASE1
        job_name = builder.ROLE_JOB_NAMES[role]
        attempt_id = _attempt_ids()[0]
        plan = {
            "custody_stages": [
                {
                    "role": role,
                    "held_slurm_job_id": job_id,
                    "job_name": job_name,
                    "expected_node": "ec51",
                    "script": {
                        "path": str(JOB_PATH.relative_to(REPOSITORY)),
                        "sha256": adjudicator.sha256_file(JOB_PATH),
                    },
                }
            ],
            "attempt_table": [
                {
                    "attempt_id": attempt_id,
                    "producer": {"job_role": role},
                    "verifier": {"job_role": builder.ROLE_DIRECT_VERIFIER},
                }
            ],
        }
        plan_raw = adjudicator.canonical_json_bytes(plan) + b"\n"
        plan_sha256 = _sha256(plan_raw)
        dynamic = {
            "SLURM_CPUS_PER_TASK": "4",
            "SLURM_JOB_ID": job_id,
            "SLURM_JOB_NAME": job_name,
            "SLURM_JOB_NODELIST": "ec51",
            "SLURM_NODELIST": "ec51",
            "SLURM_SUBMIT_DIR": "/lustre/fs1/home/sa305415/shohin_acw",
        }
        environment = dict(CANONICAL_PILOT_STATIC_ENV)
        environment.update(dynamic)
        receipt = {
            "schema": "r12_acw_development_execution_receipt_v1",
            "protocol": "R12-ACW-DEVELOPMENT-EXECUTION-v1",
            "scientific_commit": "d" * 40,
            "canonical_runtime_sha256": (
                adjudicator.CANONICAL_DEVELOPMENT_RUNTIME_SHA256
            ),
            "development_plan_sha256": plan_sha256,
            "environment_sha256": _sha256(
                adjudicator.canonical_json_bytes(environment)
            ),
            "batch_script_sha256": adjudicator.sha256_file(JOB_PATH),
            "slurm": {
                "job_id": job_id,
                "job_name": job_name,
                "node_list": "ec51",
                "cpus_per_task": "4",
            },
            "process_membership": {
                "cpu_list": "0-3",
                "memory_list": "0",
                "task_cgroup": task_cgroup,
            },
            "role": role,
            "attempt_id": attempt_id,
            "verification_replay": False,
        }
        return receipt, plan_raw, plan_sha256

    def test_normal_srun_step_cgroup_is_not_accepted_as_stage_custody(self) -> None:
        receipt, plan_raw, plan_sha256 = self._producer_receipt(
            task_cgroup=(f"/slurm/uid_{CANONICAL_PILOT_UID}/job_740999/step_0/task_0")
        )
        with (
            mock.patch.object(
                adjudicator,
                "DEVELOPMENT_PLAN_RAW_SHA256",
                plan_sha256,
            ),
            mock.patch.object(
                adjudicator,
                "_read_regular_file",
                return_value=(plan_raw, plan_sha256),
            ),
        ):
            with self.assertRaisesRegex(
                adjudicator.EvidenceError,
                "batch.*cgroup|srun|top-level.*job",
            ):
                adjudicator._validate_development_execution_receipt(
                    receipt, label="ordinary_srun_step"
                )

    def test_terminal_accounting_rejects_extra_steps_and_allocation_drift(
        self,
    ) -> None:
        plan = json.loads(PLAN_PATH.read_bytes())
        stage = plan["custody_stages"][0]
        stage["held_slurm_job_id"] = "740999"
        rows = "\n".join(
            (
                "740999|shohin-acw-dev-p1|COMPLETED|0:0|ec51|4|120|",
                "740999.batch|batch|COMPLETED|0:0|ec51|4|120|1G",
                "740999.extern|extern|COMPLETED|0:0|ec51|4|120|1M",
            )
        )
        validated = builder._validated_terminal_rows(plan, builder.ROLE_PHASE1, rows)
        self.assertEqual(
            [row["job_id_raw"] for row in validated],
            [
                "740999",
                "740999.batch",
                "740999.extern",
            ],
        )
        with self.assertRaisesRegex(ValueError, "row set|step-free"):
            builder._validated_terminal_rows(
                plan,
                builder.ROLE_PHASE1,
                rows + "\n740999.0|bash|COMPLETED|0:0|ec51|4|1|1M",
            )
        with self.assertRaisesRegex(ValueError, "allocation"):
            builder._validated_terminal_rows(
                plan,
                builder.ROLE_PHASE1,
                rows.replace("ec51|4|120|1G", "ec51|8|120|1G"),
            )

    def test_consumer_handoff_includes_predecessor_terminal_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            plan = {"input_table": [], "attempt_table": []}
            relative_paths = (
                "development_plan.json",
                "attempt_claim.json",
                "attempt_start.json",
                "custody/stages/phase1_producer_start.json",
                "custody/stages/phase1_producer_completion.json",
                "custody/stages/phase1_producer_accounting.json",
                "direct_state_producer_manifest.json",
            )
            for relative_path in relative_paths:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"{}\n")
                path.chmod(0o444)
            with mock.patch.object(builder, "_plan_for_root", return_value=plan):
                summary = builder.closed_world_scan(
                    root,
                    "phase1",
                    include_current_accounting=True,
                )
            handoff = {
                "predecessor_role": "phase1_producer",
                "predecessor_stage": "phase1",
                "predecessor_completion": builder._reference(
                    root / "custody/stages/phase1_producer_completion.json", root
                ),
                "predecessor_terminal_accounting": builder._reference(
                    root / "custody/stages/phase1_producer_accounting.json", root
                ),
                "live_closed_world_before_consumer_scoring": {"main": summary},
                "consumer_observed_before_role_scoring": True,
            }
            validated = adjudicator._validate_predecessor_handoff(
                handoff,
                plan=plan,
                base=root,
                role="phase1_verifier",
            )
            self.assertEqual(validated["predecessor_role"], "phase1_producer")

    def test_closed_world_summary_reopens_every_claimed_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            record = {
                "path": "ghost.bin",
                "bytes": 1,
                "mode": "0444",
                "sha256": _sha256(b"x"),
            }
            digest = hashlib.sha256(
                adjudicator.canonical_json_bytes(record) + b"\n"
            ).hexdigest()
            summary = {
                "stage": "phase1",
                "root": str(root),
                "file_count": 1,
                "directory_count": 1,
                "files": [record],
                "tree_sha256": digest,
                "exact_file_set": True,
                "exact_directory_set": True,
                "symlinks": 0,
                "special_files": 0,
            }
            with self.assertRaisesRegex(
                adjudicator.EvidenceError, "unavailable|bytes differ"
            ):
                adjudicator._validate_closed_world_summary(
                    summary,
                    expected_root=root,
                    expected_stage="phase1",
                    expected_paths=["ghost.bin"],
                    expected_directory_count=1,
                    label="forged closed world",
                )


if __name__ == "__main__":
    unittest.main()
