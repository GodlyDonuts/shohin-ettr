import copy
import json
from pathlib import Path

import pytest

from pipeline.validate_pretrain_sources import RegistryError, validate_registry


REGISTRY_PATH = Path(__file__).with_name("pretrain_sources.json")


def load_registry():
    return json.loads(REGISTRY_PATH.read_text())


def test_checked_in_registry_is_fail_closed():
    receipt = validate_registry(load_registry())
    assert receipt["phase2_candidate_mix_total_pct"] == 100
    assert receipt["status"] == "valid_candidate_registry_not_training_admission"


def test_registry_rejects_training_admission_status():
    registry = load_registry()
    registry["status"] = "approved"
    with pytest.raises(RegistryError, match="deny training admission"):
        validate_registry(registry)


def test_registry_rejects_open_unknown_code_license_gate():
    registry = load_registry()
    registry["global_requirements"]["fail_on_unknown_code_license"] = False
    with pytest.raises(RegistryError, match="fail-closed"):
        validate_registry(registry)


def test_registry_rejects_bad_candidate_mix_total():
    registry = load_registry()
    registry["phase2_quality_first_mix_candidate_pct"]["fineweb_edu_selected"] -= 1
    with pytest.raises(RegistryError, match="must total 100"):
        validate_registry(registry)


def test_registry_rejects_positive_weight_held_source():
    registry = copy.deepcopy(load_registry())
    held = next(source for source in registry["sources"] if source["priority"] == "hold")
    held["target_mix_pct"] = 1
    with pytest.raises(RegistryError, match="positive weight"):
        validate_registry(registry)
