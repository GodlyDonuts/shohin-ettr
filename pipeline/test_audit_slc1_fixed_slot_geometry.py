import unittest

from audit_slc1_fixed_slot_geometry import audit_rows, numeric_spans


class FixedSlotGeometryTest(unittest.TestCase):
    def test_numeric_spans_ignore_binary_minus(self):
        spans = numeric_spans("Compute (6.220-8.724) and -9 * 4")
        self.assertEqual([span["surface"] for span in spans], ["6.220", "8.724", "9", "4"])

    def test_audit_maps_literals_and_dependencies(self):
        rows = [
            {
                "identity_sha256": "a" * 64,
                "question": "Calculate 2 + 3 + 4.",
                "records": [
                    {
                        "operation": "ADD",
                        "operands": [
                            {"numerator": 2, "denominator": 1},
                            {"numerator": 3, "denominator": 1},
                        ],
                        "result": {"numerator": 5, "denominator": 1},
                        "dependencies": [],
                    },
                    {
                        "operation": "ADD",
                        "operands": [
                            {"numerator": 5, "denominator": 1},
                            {"numerator": 4, "denominator": 1},
                        ],
                        "result": {"numerator": 9, "denominator": 1},
                        "dependencies": [
                            {"operand_role": "left", "record_index": 0}
                        ],
                    },
                ],
            }
        ]
        report = audit_rows(rows)
        self.assertEqual(report["counts"]["fully_pointer_supervisable_rows"], 1)
        self.assertEqual(report["counts"]["source_literal_operands"], 3)
        self.assertEqual(report["counts"]["dependency_operands"], 1)
        self.assertEqual(report["rates"]["unmatched_source_literal_operand"], 0.0)

    def test_audit_maps_negated_prior_result(self):
        rows = [
            {
                "identity_sha256": "b" * 64,
                "question": "Calculate -(2 + 3) * 4.",
                "records": [
                    {
                        "operation": "ADD",
                        "operands": [
                            {"numerator": 2, "denominator": 1},
                            {"numerator": 3, "denominator": 1},
                        ],
                        "result": {"numerator": 5, "denominator": 1},
                        "dependencies": [],
                    },
                    {
                        "operation": "MUL",
                        "operands": [
                            {"numerator": -5, "denominator": 1},
                            {"numerator": 4, "denominator": 1},
                        ],
                        "result": {"numerator": -20, "denominator": 1},
                        "dependencies": [],
                    },
                ],
            }
        ]
        report = audit_rows(rows)
        self.assertEqual(report["counts"]["negated_dependency_operands"], 1)
        self.assertEqual(report["counts"]["fully_pointer_supervisable_rows"], 1)


if __name__ == "__main__":
    unittest.main()
