import unittest

from aggregate_pstc1_stack import _intervention_loss


class PSTC1AggregateTest(unittest.TestCase):
    def test_intervention_loss(self):
        normal = {"details": [{"identity_sha256": "a", "selected": True, "exact_skeleton": True}]}
        control = {"details": [{"identity_sha256": "a", "selected": True, "exact_skeleton": False}]}
        self.assertEqual(_intervention_loss(normal, control, lambda row: row["selected"]), 1.0)


if __name__ == "__main__":
    unittest.main()
