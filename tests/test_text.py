import pytest

from athex_rag.text import (
    extract_numbers,
    normalize,
    numbers_match,
    parse_number,
    token_f1,
    token_recall,
    tokenize,
)


def test_normalize_accents_and_final_sigma():
    assert normalize("Κέρδη Ομίλου ΧΡΉΣΗΣ") == "κερδη ομιλου χρησησ"
    assert normalize("χρήσης") == normalize("ΧΡΗΣΗΣ")


def test_tokenize_drops_stopwords():
    assert tokenize("Τα καθαρά κέρδη της εταιρείας") == ["καθαρα", "κερδη", "εταιρειασ"]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1.234.567,89", 1234567.89),
        ("1,234,567.89", 1234567.89),
        ("1.234", 1234.0),
        ("12,5", 12.5),
        ("(1.234)", -1234.0),
        ("-45,2", -45.2),
        ("2024", 2024.0),
    ],
)
def test_parse_number(raw, expected):
    assert parse_number(raw) == pytest.approx(expected)


def test_extract_numbers_with_scale_word():
    values = extract_numbers("Κέρδη 1,2 εκατ. ευρώ")
    assert 1.2 in values and pytest.approx(1_200_000) in values


def test_numbers_match_across_units():
    assert numbers_match("τα κέρδη ήταν 1.234,5 εκατ. €", 1_234_500_000)
    assert numbers_match("σε χιλ. €: 1.234.500", 1_234_500_000)
    assert not numbers_match("τα κέρδη ήταν 999 εκατ.", 1_234_500_000)


def test_token_f1():
    assert token_f1("εννέα μέλη", "Εννέα μέλη") == 1.0
    assert token_f1("δέκα άτομα", "εννέα μέλη") == 0.0


def test_token_recall_tolerates_wordy_answers():
    wordy = "Σύμφωνα με την έκθεση, το συμβούλιο έχει εννέα μέλη [1]."
    assert token_recall(wordy, "εννέα μέλη") == 1.0


def test_numbers_match_ignores_sign_of_losses():
    assert numbers_match("Ζημιές από διακοπείσες δραστηριότητες: (224,9) εκατ.", 224_900_000)
    assert numbers_match("ζημιές 224,9 εκατ. ευρώ", -224_900_000)


@pytest.mark.parametrize(
    "text,gold,ok",
    [
        ("1.232,90 εκατ. ευρώ", 1_232_904_830, True),  # rounded to the digits shown
        ("Ευρώ 3,6 δισ.", 3_590_800_000, True),
        ("21,1 %", 21.2, False),  # a different ratio, not a rounding of it
        ("18,8%", 18.8, True),
        ("0,8777 ευρώ", 0.8777, True),
        ("0,88 ευρώ", 0.8777, True),
        ("0,87 ευρώ", 0.8777, False),
        ("€0,45 δισ.", 359_453_000, False),
        ("492,8", 478.8, False),
    ],
)
def test_numbers_match_requires_consistent_rounding(text, gold, ok):
    assert numbers_match(text, gold) is ok


def test_min_digits_ignores_coarse_numbers():
    assert numbers_match("κέρδη €0,36 δισ.", 359_453_000) is True  # 0,36 δισ. is a rounding
    assert numbers_match("κέρδη €0,36 δισ.", 359_453_000, min_digits=4) is False
