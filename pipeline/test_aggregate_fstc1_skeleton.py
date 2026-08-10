import unittest

from aggregate_fstc1_skeleton import _group_rate


class FSTC1AggregateTest(unittest.TestCase):
    def test_group_rate(self):
        self.assertEqual(_group_rate({"rows": 10, "complete_skeleton_exact": 9}, "complete_skeleton_exact"), 0.9)


if __name__ == "__main__":
    unittest.main()
