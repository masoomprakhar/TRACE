"""India-specific license-plate correction.

Generic OCR returns plates that are *almost* right: it confuses 0/O, 1/I,
8/B, 5/S, 2/Z, etc. Because Indian plates have a known structure
(STATE-RTO-SERIES-NUMBER), we can fix most of these by coercing each
position to the character class it must be.

Canonical format:  LL DD L(1-3) DDDD   e.g.  MH 01 AB 1234
Also supported:     DD BH DDDD L(1-2)   (Bharat / BH series)
"""

from __future__ import annotations

import re

# Standard plate: 2 letters, 1-2 digits, 1-3 letters, 4 digits.
_STANDARD_RE = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")
# Bharat (BH) series: 2 digits, "BH", 4 digits, 1-2 letters.
_BH_RE = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")

# Letter that was probably a digit -> digit.
_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "A": "4",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
}
# Digit that was probably a letter -> letter.
_TO_ALPHA = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "G",
    "8": "B",
}


def _coerce_digits(s: str) -> str:
    return "".join(_TO_DIGIT.get(c, c) for c in s)


def _coerce_alpha(s: str) -> str:
    return "".join(_TO_ALPHA.get(c, c) for c in s)


def normalize_plate(text: str) -> str:
    """Uppercase and strip everything that isn't A-Z/0-9."""
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def is_valid_plate(text: str) -> bool:
    s = normalize_plate(text)
    return bool(_STANDARD_RE.match(s) or _BH_RE.match(s))


def correct_plate(raw: str) -> tuple[str, bool]:
    """Return (corrected_plate, is_valid).

    If the cleaned string already matches a known format it is returned
    unchanged. Otherwise we coerce each segment to its expected character
    class and re-validate.
    """
    s = normalize_plate(raw)
    if not s:
        return "", False
    if _STANDARD_RE.match(s) or _BH_RE.match(s):
        return s, True

    # Plausible length for a structured correction (LL DD [LLL] DDDD).
    if 8 <= len(s) <= 11:
        state = _coerce_alpha(s[:2])
        last4 = _coerce_digits(s[-4:])
        middle = s[2:-4]
        if middle:
            rto_len = 2 if len(middle) >= 3 else 1
            rto = _coerce_digits(middle[:rto_len])
            series = _coerce_alpha(middle[rto_len:])
        else:
            rto, series = "", ""
        corrected = f"{state}{rto}{series}{last4}"
        if _STANDARD_RE.match(corrected):
            return corrected, True
        return corrected, False

    # Couldn't structure it confidently — hand back the cleaned string.
    return s, False


def format_plate(text: str) -> str:
    """Insert conventional spacing: 'MH01AB1234' -> 'MH 01 AB 1234'."""
    s = normalize_plate(text)
    m = re.match(r"^([A-Z]{2})([0-9]{1,2})([A-Z]{1,3})([0-9]{4})$", s)
    if m:
        return " ".join(m.groups())
    return s


def plate_similarity(a: str, b: str) -> float:
    """1.0 == identical, 0.0 == fully different. Normalized Levenshtein over
    the cleaned strings — used for fuzzy plate search."""
    a, b = normalize_plate(a), normalize_plate(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    dist = prev[lb]
    return 1.0 - dist / max(la, lb)
