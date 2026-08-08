#!/usr/bin/env python3
"""Fail-closed checkpoint configuration contracts for pretraining resume."""

import unittest
from dataclasses import asdict, replace

from model import GPTConfig
from train import (
    validate_model_data_identity,
    validate_resume_config,
    validate_resume_data_bindings,
)


class ResumeConfigContractTests(unittest.TestCase):
    def setUp(self):
        self.cfg = GPTConfig()

    def test_exact_matching_checkpoint_config_is_allowed(self):
        validate_resume_config(self.cfg, asdict(self.cfg))

    def test_every_behavior_field_mismatch_is_rejected(self):
        mutations = {
            "vocab_size": self.cfg.vocab_size + 1,
            "n_layer": self.cfg.n_layer + 1,
            "n_head": self.cfg.n_head + 1,
            "n_kv_head": self.cfg.n_kv_head + 1,
            "d_model": self.cfg.d_model + 1,
            "d_ff": self.cfg.d_ff + 1,
            "seq_len": self.cfg.seq_len + 1,
            "rope_theta": self.cfg.rope_theta + 1.0,
            "qk_norm": not self.cfg.qk_norm,
            "tie_embeddings": not self.cfg.tie_embeddings,
            "zloss": self.cfg.zloss * 2.0,
            "n_loop": self.cfg.n_loop + 1,
        }
        for field_name, requested_value in mutations.items():
            with self.subTest(field=field_name):
                requested = replace(self.cfg, **{field_name: requested_value})
                with self.assertRaisesRegex(ValueError, field_name):
                    validate_resume_config(requested, asdict(self.cfg))

    def test_legacy_checkpoint_without_config_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, "legacy checkpoints cannot be resumed safely"):
            validate_resume_config(self.cfg, None)

    def test_incomplete_or_future_checkpoint_schema_is_rejected(self):
        missing = asdict(self.cfg)
        del missing["n_loop"]
        with self.assertRaisesRegex(ValueError, "missing fields: n_loop"):
            validate_resume_config(self.cfg, missing)

        unknown = asdict(self.cfg)
        unknown["future_behavior"] = True
        with self.assertRaisesRegex(ValueError, "unknown fields: future_behavior"):
            validate_resume_config(self.cfg, unknown)

    def test_tokenizer_vocabulary_must_match_model(self):
        validate_model_data_identity(49_152, {"tokenizer_vocab_size": 49_152})
        with self.assertRaisesRegex(ValueError, "model vocabulary differs"):
            validate_model_data_identity(32_768, {"tokenizer_vocab_size": 49_152})

    def test_phase2_resume_requires_bound_admission(self):
        binding = {"admission_sha256": "a" * 64}
        with self.assertRaisesRegex(ValueError, "lacks a Phase-2 admission"):
            validate_resume_data_bindings(
                size="shohin_390m",
                checkpoint_data_binding=None,
                data_binding={"contract_sha256": "b" * 64},
                checkpoint_admission=None,
                admission_binding=binding,
                allow_transition=False,
            )
        validate_resume_data_bindings(
            size="shohin_390m",
            checkpoint_data_binding=None,
            data_binding={"contract_sha256": "b" * 64},
            checkpoint_admission=None,
            admission_binding=binding,
            allow_transition=True,
        )

    def test_resume_rejects_admission_substitution(self):
        with self.assertRaisesRegex(ValueError, "Phase-2 admission differs"):
            validate_resume_data_bindings(
                size="shohin_920m",
                checkpoint_data_binding={"contract_sha256": "b" * 64},
                data_binding={"contract_sha256": "b" * 64},
                checkpoint_admission={"admission_sha256": "a" * 64},
                admission_binding={"admission_sha256": "c" * 64},
                allow_transition=False,
            )


if __name__ == "__main__":
    unittest.main()
