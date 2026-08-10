import unittest

from shared_post_mlp_revision import SharedPostMLPConfig
from train_dset_transfer import DSETTransferError, validate_warm_start_metadata


class DSETTransferWarmStartTest(unittest.TestCase):
    def setUp(self):
        self.config = SharedPostMLPConfig(
            hidden_size=2048, controlled_layers=16, rank=18, alpha=18.0
        )
        self.metadata = {
            "architecture": "shohin-shared-post-mlp-revision-v1",
            "model_revision": "995ad96e",
            "quantization": "nf4",
            "shared_post_mlp_config": {
                "hidden_size": 2048,
                "controlled_layers": 16,
                "rank": 18,
                "alpha": 18.0,
            },
            "draft_control": "normal",
            "dset1_arm": "aligned",
            "trainable_parameters": 1_179_648,
            "trainable_parameter_name_sha256": "names",
        }

    def validate(self, arm="aligned"):
        validate_warm_start_metadata(
            self.metadata,
            arm=arm,
            model_revision="995ad96e",
            quantization="nf4",
            config=self.config,
            trainable_parameters=1_179_648,
            trainable_name_sha256="names",
        )

    def test_aligned_owner_can_warm_aligned_and_swapped(self):
        self.validate("aligned")
        self.validate("swapped")

    def test_hidden_requires_hidden_owner(self):
        with self.assertRaises(DSETTransferError):
            self.validate("hidden")
        self.metadata["draft_control"] = "draft_unavailable"
        self.metadata["dset1_arm"] = "hidden"
        self.validate("hidden")

    def test_geometry_drift_fails(self):
        self.metadata["trainable_parameters"] += 1
        with self.assertRaises(DSETTransferError):
            self.validate()


if __name__ == "__main__":
    unittest.main()
