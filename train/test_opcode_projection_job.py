from pathlib import Path


JOB = Path(__file__).with_name("jobs") / "eval_algebraic_query_joint_state.sbatch"


def test_opcode_projection_job_is_hash_bound_and_optional() -> None:
    text = JOB.read_text()
    for expected in (
        "OPCODE_PROJECTION_REGISTRY",
        "OPCODE_PROJECTION_REGISTRY_SHA256",
        "sha256sum \"$OPCODE_PROJECTION_REGISTRY\"",
        "--opcode-projection-registry",
        "--opcode-projection-registry-sha256",
        '"${PROJECTION_ARGS[@]}"',
    ):
        assert expected in text
