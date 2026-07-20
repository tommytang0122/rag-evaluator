from __future__ import annotations

import hashlib
from collections.abc import Sequence

from rag_evaluator.eval.numeric import extract_values

PURE_REFUSAL = "pure_refusal"
SUBSTANTIVE = "substantive_answer"
MIXED = "mixed_refusal_answer"
EMPTY = "empty_non_answer"

DEFAULT_REFUSAL_PHRASES: tuple[str, ...] = (
    "找不到相關資訊",
    "找不到任何相關的資料",
    "找不到原始資料",
    "沒有相關資訊",
    "無法在文件中找到",
)


def classify_refusal(
    answer: str,
    phrases: Sequence[str] = DEFAULT_REFUSAL_PHRASES,
    min_residual_chars: int = 4,
) -> str:
    text = answer.strip()
    if not text:
        return EMPTY
    matched = [p for p in phrases if p in text]
    residual = text
    for p in matched:
        residual = residual.replace(p, "")
    residual = residual.strip(" \t\n。，,.!！?？")
    substantive = bool(extract_values(residual)) or len(residual) >= min_residual_chars
    if matched and substantive:
        return MIXED
    if matched:
        return PURE_REFUSAL
    if substantive:
        return SUBSTANTIVE
    return EMPTY


def refusal_phrases_hash(phrases: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(phrases).encode("utf-8")).hexdigest()
