from build_dset1_span_edit_data import convert_pair, registered_final_span


def _pair(clean="The answer is B.", fault="The answer is C.", family="choice_final"):
    start = clean.rfind("B")
    base = {
        "schema": "shohin-dseo1-paired-presentation-v1",
        "pair_identity_sha256": "pair",
        "source_identity_sha256": "source",
        "corruption_family": family,
        "final_response": clean,
        "changed_character_span": [start, start + 1],
        "task": "test",
        "training_group": "science",
    }
    instruction = (
        "\n\nFirst emit exactly one edit action on its own line: <KEEP>, "
        "<FIX_FINAL>, <FIX_STEP>, <FIX_CODE>, or <REWRITE>. Then emit the "
        "complete final trajectory."
    )
    return [
        {**base, "pair_member": "clean", "draft": clean, "question": "q" + instruction, "identity_sha256": "c"},
        {**base, "pair_member": "fault", "draft": fault, "question": "q" + instruction, "identity_sha256": "f"},
    ]


def test_convert_pair_builds_draft_specific_script() -> None:
    rows, reason = convert_pair(_pair())
    assert reason is None
    assert rows is not None
    assert rows[0]["script"] == "<KEEP>\n"
    assert rows[1]["script"] == "<REPLACE_LAST>\nC\nB\n"
    assert rows[0]["swapped_script"] == rows[1]["script"]


def test_convert_pair_rejects_latex_command_letter_mutation() -> None:
    clean = "F\n\\boxed{\\text{F}}"
    fault = "F\n\\boxed{\\taxt{F}}"
    pair = _pair(clean, fault)
    start = clean.index("text") + 1
    for row in pair:
        row["changed_character_span"] = [start, start + 1]
        row["final_response"] = clean
    rows, reason = convert_pair(pair)
    assert rows is None
    assert reason == "registered_span_not_semantic_final"


def test_registered_final_span_accepts_latex_payload_not_command() -> None:
    text = "F\n\\boxed{\\text{F}}"
    command_e = text.index("text") + 1
    answer_f = text.rfind("F")
    assert not registered_final_span(text, command_e, command_e + 1)
    assert registered_final_span(text, answer_f, answer_f + 1)
