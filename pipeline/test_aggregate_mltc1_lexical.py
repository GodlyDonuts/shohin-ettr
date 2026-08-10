import unittest

from aggregate_mltc1_lexical import group


class MLTC1AggregateTest(unittest.TestCase):
    def test_missing_group_fails_closed(self):
        self.assertEqual(group({"groups": {}}, "mixed:true"), 0.0)


if __name__ == "__main__":
    unittest.main()
