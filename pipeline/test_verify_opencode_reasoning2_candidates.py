#!/usr/bin/env python3
"""Focused non-network tests for OCR2 source reconstruction and replay."""

import json

from verify_opencode_reasoning2_candidates import (
    code_contests_cases,
    question_and_cases,
    solution_is_safe,
    stdio_cases,
    verify_match,
)


def test_stdio_cases_supports_json_and_all_cases():
    raw = json.dumps({"inputs": ["1\n", "2\n"], "outputs": ["2\n", "4\n"]})
    assert stdio_cases(raw, max_tests=0, max_case_chars=20) == [
        ("1\n", "2\n"),
        ("2\n", "4\n"),
    ]


def test_stdio_cases_rejects_function_contract():
    assert (
        stdio_cases(
            {"fn_name": "solve", "inputs": ["1"], "outputs": ["2"]},
            max_tests=0,
            max_case_chars=20,
        )
        is None
    )


def test_code_contests_cases_preserves_test_family_order():
    row = {
        "public_tests": {"input": ["1\n"], "output": ["2\n"]},
        "generated_tests": {"input": ["2\n"], "output": ["4\n"]},
        "private_tests": {"input": ["3\n"], "output": ["6\n"]},
    }
    assert code_contests_cases(row, max_tests=2, max_case_chars=20) == [
        ("1\n", "2\n"),
        ("2\n", "4\n"),
    ]


def test_question_mapping_matches_ocr2_card():
    question, cases = question_and_cases(
        "apps",
        {
            "question": "Double it",
            "input_output": {"inputs": ["2\n"], "outputs": ["4\n"]},
        },
        max_tests=0,
        max_case_chars=20,
    )
    assert question == "Double it"
    assert cases == [("2\n", "4\n")]


def test_solution_safety_allows_normal_contest_imports():
    assert solution_is_safe("import sys\nfrom collections import deque\nprint(1)")


def test_solution_safety_rejects_host_capabilities():
    assert not solution_is_safe("import os\nos.system('echo unsafe')")
    assert not solution_is_safe("print(open('/etc/passwd').read())")
    assert not solution_is_safe("getattr(__builtins__, 'eval')('1+1')")


def test_verify_match_accepts_exact_source_execution():
    candidate = {
        "identity_sha256": "a" * 64,
        "solution": "value = int(input())\nprint(value * 2)",
        "response": "<think>Double the value.</think>\n```python\nvalue = int(input())\nprint(value * 2)\n```",
    }
    clean, drop = verify_match(
        (
            7,
            candidate,
            {
                "question": "Given an integer, print twice its value.",
                "input_output": {
                    "inputs": ["2\n", "5\n"],
                    "outputs": ["4\n", "10\n"],
                },
            },
        ),
        dataset="taco",
        test_grams=set(),
        ngram=13,
        max_tests=0,
        max_case_chars=20,
        min_tests=2,
        timeout=1.0,
    )
    assert drop is None
    assert clean["question"] == "Given an integer, print twice its value."
    assert clean["verified_cases"] == 2
    assert clean["verification"] == "execution_verified_source_tests"


def test_verify_match_rejects_wrong_program():
    clean, drop = verify_match(
        (
            9,
            {"identity_sha256": "b" * 64, "solution": "print(0)"},
            {
                "question": "Print one.",
                "input_output": {"inputs": [""], "outputs": ["1\n"]},
            },
        ),
        dataset="apps",
        test_grams=set(),
        ngram=13,
        max_tests=0,
        max_case_chars=20,
        min_tests=1,
        timeout=1.0,
    )
    assert clean is None
    assert drop == "execution"
