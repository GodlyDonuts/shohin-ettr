from build_obr1_broad_owner_data import (
    development_question,
    grams,
    normalized_question,
    overlap_kind,
)


def test_normalization_and_grams_are_case_and_punctuation_stable():
    assert normalized_question("Find X, then Y!") == "find x then y"
    assert grams("A b c d", 3) == {"a b c", "b c d"}


def test_overlap_rejects_exact_and_thirteen_word_collision():
    held = "one two three four five six seven eight nine ten eleven twelve thirteen"
    exact = {normalized_question("An exact heldout question")}
    ngrams = grams(held, 13)
    assert overlap_kind("AN EXACT HELDOUT QUESTION!", exact, ngrams, 13) == "exact"
    assert overlap_kind(f"prefix {held} suffix", exact, ngrams, 13) == "ngram"
    assert overlap_kind("a genuinely distinct prompt", exact, ngrams, 13) is None


def test_development_question_prefers_model_owned_source():
    row = {
        "internal_draft": {"question": "owned source"},
        "assessor": {"question": "assessor source"},
    }
    assert development_question(row) == "owned source"
