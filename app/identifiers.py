"""Deterministic normalization for identifiers received from typed or voice input."""

import re
from typing import Literal

IdentifierType = Literal["customer_id", "payment_id", "invoice_id", "order_id", "ticket_id"]

_PREFIXES: dict[IdentifierType, str] = {
    "customer_id": "CUST",
    "payment_id": "PAY",
    "invoice_id": "INV",
    "order_id": "ORD",
    "ticket_id": "TKT",
}
_DIGIT_WORDS = {
    "ZERO": "0",
    "OH": "0",
    "ONE": "1",
    "TWO": "2",
    "THREE": "3",
    "FOUR": "4",
    "FIVE": "5",
    "SIX": "6",
    "SEVEN": "7",
    "EIGHT": "8",
    "NINE": "9",
}


def normalize_identifier(value: str, identifier_type: IdentifierType) -> str | None:
    """Normalize one identifier candidate without interpreting arbitrary conversation."""
    if not value or identifier_type not in _PREFIXES:
        return None

    prefix = _PREFIXES[identifier_type]
    text = value.strip().upper()
    text = re.sub(r"\bSEE\s+YOU\s+ESS\s+TEE\b", "CUST", text)
    text = re.sub(r"\bCUSTOMER\b", "CUST", text) if identifier_type == "customer_id" else text
    tokens = re.findall(r"[A-Z0-9]+", text)
    compact = "".join(_DIGIT_WORDS.get(token, token) for token in tokens)
    if compact.isdigit() and all(token in _DIGIT_WORDS or token.isdigit() for token in tokens):
        return compact
    if not compact.startswith(prefix):
        return None

    normalized = prefix + compact[len(prefix) :]
    if identifier_type == "customer_id":
        return normalized if re.fullmatch(r"CUST\d{4}", normalized) else None
    return normalized if re.fullmatch(rf"{prefix}[A-Z0-9]{{6,}}", normalized) else None


def mask_identifier(value: str | None) -> str:
    """Keep logs useful while avoiding complete identifier retention."""
    if not value:
        return "<none>"
    if len(value) <= 4:
        return "****"
    return f"{value[:4]}****{value[-4:]}"


def normalize_spelled_name(value: str) -> str | None:
    """Collapse only names made from individually spelled alphabetic characters."""
    text = value.strip()
    if not re.fullmatch(r"[A-Za-z](?:[\s-]+[A-Za-z])+", text):
        return None
    return "".join(re.findall(r"[A-Za-z]", text)).casefold()