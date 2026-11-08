"""Greek-aware text utilities: normalization, tokenization and number parsing.

Greek financial reports write numbers as ``1.234.567,89`` and negatives as ``(1.234)``,
and words carry accents (τόνοι) and a final sigma (ς). Both break naive keyword search
and naive answer checking, so everything that compares text goes through this module.
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_NUMBER_RE = re.compile(r"\(?-?\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?\)?|\(?-?\d+(?:[.,]\d+)?\)?")

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


def parse_number(raw: str, decimal: str = ",") -> float:
    """Parse one numeric string written in Greek or English convention.

    ``decimal`` is the separator assumed when a single separator is ambiguous
    (e.g. ``12,5``). Thousands groups of exactly three digits are detected.
    """
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
    if dec:
        s = s.replace(dec, ".")
    value = float(s)
    return -value if negative else value


def extract_numbers(text: str, decimal: str = ",") -> list[float]:
    """All numbers in ``text``, each also scaled if followed by a scale word."""
    values: list[float] = []
    norm = normalize(text)
    for match in _NUMBER_RE.finditer(norm):
        try:
            value = parse_number(match.group(), decimal=decimal)
        except ValueError:
            continue
        values.append(value)
        following = norm[match.end() : match.end() + 16]
        word = _TOKEN_RE.search(following)
        if word and following[: word.start()].strip() in ("", "."):
            scale = SCALE_WORDS.get(word.group())
            if scale:
                values.append(value * scale)
    return values


def numbers_match(predicted: str, gold: float, rel_tol: float = 0.01) -> bool:
    """True if any number in ``predicted`` equals ``gold`` within ``rel_tol``.

    Reports mix units (euro vs thousands of euro vs millions), so we accept a match
    at any common scale factor. Signs are ignored: a loss is written "(224,9)" in a
    table and "ζημιές 224,9" in prose, and the question already fixes the direction.
    """
    if gold == 0:
        return any(abs(v) < 1e-9 for v in extract_numbers(predicted))
    gold = abs(gold)
    for value in extract_numbers(predicted):
        value = abs(value)  # a loss may be written "(224,9)" or "ζημιές 224,9"
        for scale in (1, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9):
            if abs(value * scale - gold) <= rel_tol * gold:
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
