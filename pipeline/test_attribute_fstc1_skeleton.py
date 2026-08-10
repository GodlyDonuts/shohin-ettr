import unittest

from attribute_fstc1_skeleton import _bucket


class FSTC1AttributionTest(unittest.TestCase):
    def test_nested_unary_mixed_bucket(self):
        buckets = _bucket("Calculate -(2 + 3) * 4.", "basic_arithmetic")
        self.assertIn("parentheses:1-2", buckets)
        self.assertIn("unary_group:true", buckets)
        self.assertIn("mixed_precedence:true", buckets)


if __name__ == "__main__":
    unittest.main()
