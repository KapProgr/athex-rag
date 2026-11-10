"""Greek-aware text utilities: normalization, tokenization and number parsing.

Greek financial reports write numbers as ``1.234.567,89`` and negatives as ``(1.234)``,
and words carry accents (τόνοι) and a final sigma (ς). Both break naive keyword search
and naive answer checking, so everything that compares text goes through this module.
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
# a thousands group is exactly three digits: "0,8777" must not split into "0,877" + "7"
_NUMBER_RE = re.compile(r"\(?-?\d{1,3}(?:[.,]\d{3}(?!\d))+(?:[.,]\d+)?\)?|\(?-?\d+(?:[.,]\d+)?\)?")

# Small stopword list; enough to stop BM25 from rewarding "και", "της", "the".
_RAW_STOPWORDS = """
    ο η το οι τα του της των τον την στο στη στην στον στα στις στους σε με για απο
    και ή να θα δεν μη που πως ως αν οτι ενα ενας μια μιας ενος ειναι ηταν εχει ειχε
    κατα μετα προς εως χωρις ομως αυτο αυτη αυτα αυτος αυτων τι ποιο ποια ποιος ποσο ποσα
    the a an of and or to in on for by with is was are were be this that what which how
    """

# Words that scale a number that follows/precedes them ("1,2 εκατ.", "3 billion").
SCALE_WORDS = {
    "χιλ": 1e3,
    "χιλιαδες": 1e3,
    "thousand": 1e3,
    "εκατ": 1e6,
    "εκατομμυρια": 1e6,
    "million": 1e6,
    "m": 1e6,
    "δισ": 1e9,
    "δισεκατομμυρια": 1e9,
    "billion": 1e9,
    "bn": 1e9,
}


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def normalize(text: str) -> str:
    """Lowercase, remove accents and fold final sigma so 'Κέρδη' == 'κερδη'."""
    return strip_accents(text).lower().replace("ς", "σ")


# stored in normalized form so lookups match tokens ("της" -> "τησ")
STOPWORDS = frozenset(normalize(w) for w in _RAW_STOPWORDS.split())


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(normalize(text))
    out = []
    for tok in tokens:
        if len(tok) < 2 and not tok.isdigit():
            continue
        if drop_stopwords and tok in STOPWORDS:
            continue
        out.append(tok)
    return out


def search_tokens(text: str, prefix: int | None = 6) -> list[str]:
    """Tokens for keyword search, with light stemming by prefix truncation.

    Greek is highly inflected ("ελεγκτική", "ελεγκτές", "ελεγκτών"); cutting words to
    their first ``prefix`` characters is a crude but effective stemmer for retrieval in
    such languages. Numbers are never truncated. ``prefix=None`` disables stemming.
    """
    tokens = tokenize(text)
    if not prefix:
        return tokens
    return [t if t.isdigit() or len(t) <= prefix else t[:prefix] for t in tokens]


def _parse(raw: str, decimal: str = ",") -> tuple[float, int, int]:
    """(value, decimal places, significant digits) of one numeric string."""
    s = raw.strip()
    negative = (s.startswith("(") and s.endswith(")")) or s.startswith("-")
    s = s.strip("()").lstrip("-")
    has_dot, has_comma = "." in s, "," in s
    dec: str | None
    if has_dot and has_comma:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
    elif has_dot or has_comma:
        sep = "," if has_comma else "."
        parts = s.split(sep)
        if len(parts) > 2:
            dec = None  # 1.234.567 -> only thousands separators
        elif sep == decimal or len(parts[1]) != 3:
            dec = sep
        else:
            dec = None  # 1.234 with decimal="," -> thousands
    else:
        dec = None
    for thousands_sep in {".", ","} - {dec}:
        s = s.replace(thousands_sep, "")
    decimals = len(s.split(dec)[1]) if dec else 0
    if dec:
        s = s.replace(dec, ".")
    value = float(s)
    digits = len(s.replace(".", "").lstrip("0"))
    return (-value if negative else value), decimals, digits


def parse_number(raw: str, decimal: str = ",") -> float:
    """Parse one numeric string written in Greek or English convention.

    ``decimal`` is the separator assumed when a single separator is ambiguous
    (e.g. ``12,5``). Thousands groups of exactly three digits are detected.
    """
    return _parse(raw, decimal)[0]


def _numbers_with_precision(text: str, decimal: str = ",") -> list[tuple[float, float, int]]:
    """(value, unit of the last written digit, significant digits) for every number,
    plus a scaled copy when a scale word follows ("1,2 εκατ." -> 1.2e6, unit 1e5)."""
    out: list[tuple[float, float, int]] = []
    norm = normalize(text)
    for match in _NUMBER_RE.finditer(norm):
        try:
            value, decimals, digits = _parse(match.group(), decimal=decimal)
        except ValueError:
            continue
        unit = 10.0**-decimals
        out.append((value, unit, digits))
        following = norm[match.end() : match.end() + 16]
        word = _TOKEN_RE.search(following)
        if word and following[: word.start()].strip() in ("", "."):
            scale = SCALE_WORDS.get(word.group())
            if scale:
                out.append((value * scale, unit * scale, digits))
    return out


def extract_numbers(text: str, decimal: str = ",") -> list[float]:
    """All numbers in ``text``, each also scaled if followed by a scale word."""
    return [value for value, _, _ in _numbers_with_precision(text, decimal)]


def numbers_match(predicted: str, gold: float, min_digits: int = 0) -> bool:
    """True if a number in ``predicted`` is ``gold`` written exactly or *rounded*.

    "1.232,90 εκατ." matches 1.232.904.830 and "3,6 δισ." matches 3.590,8 εκατ.
    (each is the gold value rounded to the digits shown), but "21,1%" does not match
    21,2%: a fixed relative tolerance would accept that, which is wrong for ratios.

    Reports mix units (euro vs thousands vs millions), so any common scale factor is
    accepted. Signs are ignored: a loss is written "(224,9)" in a table and
    "ζημιές 224,9" in prose, and the question already fixes the direction.
    ``min_digits`` ignores numbers with fewer significant digits (used when checking
    whether a passage *states* a value: "0,36 δισ." is too coarse to count).
    """
    gold = abs(gold)
    for value, unit, digits in _numbers_with_precision(predicted):
        if digits < min_digits:
            continue
        value = abs(value)
        for scale in (1, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9):
            tolerance = 0.5 * unit * scale + 1e-9 * max(gold, 1.0)
            if abs(value * scale - gold) <= tolerance:
                return True
    return False


def token_f1(predicted: str, gold: str) -> float:
    """SQuAD-style token F1 after Greek normalization."""
    p, g = tokenize(predicted), tokenize(gold)
    if not p or not g:
        return float(p == g)
    common: dict[str, int] = {}
    g_counts: dict[str, int] = {}
    for tok in g:
        g_counts[tok] = g_counts.get(tok, 0) + 1
    for tok in p:
        if g_counts.get(tok, 0) > common.get(tok, 0):
            common[tok] = common.get(tok, 0) + 1
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def token_recall(predicted: str, gold: str) -> float:
    """Share of gold-answer tokens present in the prediction (order-free)."""
    g = set(tokenize(gold))
    if not g:
        return 1.0
    return len(g & set(tokenize(predicted))) / len(g)
