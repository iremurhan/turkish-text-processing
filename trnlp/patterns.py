"""Regular expressions for entities that must stay in one piece (URLs, e-mails, dates, numbers, ...)."""

import re

MONTHS = "Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık"

# Multi-word expressions kept as single tokens. Extend with `Tokenizer(mwes=...)`.
DEFAULT_MWES = (
    "New York", "San Francisco", "Boğaz Köprüsü", "Galata Kulesi", "Kız Kulesi", "Selimiye Camii",
)

SPECIAL = {
    "url": r"https?://\S+?(?=[.,;:!?)]*(?:\s|$))|www\.\S+?(?=[.,;:!?)]*(?:\s|$))",
    "email": r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
    "hashtag": r"#\w+",
    "mention": r"@\w+",
    "date": r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b|\b\d{1,2}\s(?:" + MONTHS + r")\s\d{4}\b",
    "time": r"\b\d{1,2}:\d{2}\b",
    "currency": r"[₺$€£]\s?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?|\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?\s?(?:TL|₺)",
    "phone": r"(?:\+\d{1,3}\s?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{2}[-.\s]?\d{2}\b",
    "fraction": r"\b\d+/\d+\b",
    "number": r"\b\d{1,3}(?:\.\d{3})+(?:,\d+)?\b|\b\d+(?:[.,]\d+)?\b",
}


def special_pattern(mwes=DEFAULT_MWES) -> re.Pattern:
    parts = [SPECIAL[k] for k in ("url", "email", "hashtag", "mention", "date", "time", "currency", "phone", "fraction")]
    if mwes:
        parts.append(r"\b(?:" + "|".join(re.escape(m) for m in sorted(mwes, key=len, reverse=True)) + r")\b")
    parts.append(SPECIAL["number"])
    return re.compile("|".join(f"(?:{p})" for p in parts), re.IGNORECASE)
