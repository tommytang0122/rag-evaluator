from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Mapping

NUMERIC_RULES_VERSION = "v2"

# casefolded unit key → (dimension, multiplier to canonical base unit)
DEFAULT_UNITS: dict[str, tuple[str, Decimal]] = {
    "元": ("money", Decimal(1)),
    "千元": ("money", Decimal(1000)),
    "仟元": ("money", Decimal(1000)),
    "ntd千元": ("money", Decimal(1000)),
    "新台幣千元": ("money", Decimal(1000)),
    "萬元": ("money", Decimal(10000)),
    "百萬元": ("money", Decimal(1_000_000)),
    "億元": ("money", Decimal(100_000_000)),
    "美元": ("usd", Decimal(1)),
    "千美元": ("usd", Decimal(1000)),
    "萬美元": ("usd", Decimal(10000)),
    "百萬美元": ("usd", Decimal(1_000_000)),
    "億美元": ("usd", Decimal(100_000_000)),
    "年": ("year", Decimal(1)),
    "%": ("percent", Decimal(1)),
    "％": ("percent", Decimal(1)),
}

_FULLWIDTH = str.maketrans("０１２３４５６７８９．，－", "0123456789.,-")
_NUM_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class ExtractedValue:
    raw: str
    value: Decimal
    unit: str | None
    dimension: str
    canonical: Decimal


def extract_values(
    text: str, units: Mapping[str, tuple[str, Decimal]] = DEFAULT_UNITS
) -> list[ExtractedValue]:
    text = text.translate(_FULLWIDTH)
    keys = sorted(units, key=len, reverse=True)
    out: list[ExtractedValue] = []
    for m in _NUM_RE.finditer(text):
        raw = m.group()
        try:
            value = Decimal(raw.replace(",", ""))
        except InvalidOperation:  # pragma: no cover - regex prevents this
            continue
        rest = text[m.end() : m.end() + 12].lstrip().casefold()
        unit = next((k for k in keys if rest.startswith(k)), None)
        if unit is None:
            out.append(ExtractedValue(raw, value, None, "none", value))
        else:
            dim, mult = units[unit]
            out.append(ExtractedValue(raw, value, unit, dim, value * mult))
    return out


def gold_dimension(
    unit: str | None, units: Mapping[str, tuple[str, Decimal]] = DEFAULT_UNITS
) -> str:
    return _resolve_gold_unit(unit, units)[0]


def answer_signature(
    answer: str,
    *,
    dimension: str | None = None,
    units: Mapping[str, tuple[str, Decimal]] = DEFAULT_UNITS,
) -> str:
    values = extract_values(answer, units)
    if dimension is not None:
        values = [v for v in values if v.dimension == dimension]
    return ",".join(sorted(str(v.canonical) for v in values))


@dataclass(frozen=True)
class NumericResult:
    status: str  # match | unit_mismatch | number_mismatch | no_number | ambiguous | unknown_unit
    canonical: str | None = None


def _resolve_gold_unit(
    unit: str | None, units: Mapping[str, tuple[str, Decimal]]
) -> tuple[str, Decimal]:
    if unit is None:
        return ("none", Decimal(1))
    key = unit.casefold()
    if key in units:
        return units[key]
    return (f"other:{key}", Decimal(1))


def compare_numeric(
    answer: str,
    gold_number: Decimal,
    gold_unit: str | None,
    *,
    units: Mapping[str, tuple[str, Decimal]] = DEFAULT_UNITS,
    tolerance: Decimal = Decimal(0),
) -> NumericResult:
    values = extract_values(answer, units)
    if not values:
        return NumericResult("no_number")
    gdim, gmult = _resolve_gold_unit(gold_unit, units)
    if gdim.startswith("other:"):
        # gold 用了字典外的單位:規則層無法建立可比較的維度,
        # 回報 inconclusive 讓上層升級給 judge,而不是硬判 unit_mismatch。
        return NumericResult("unknown_unit")
    gold_canonical = gold_number * gmult
    same_dim = [v for v in values if v.dimension == gdim]
    matches = [v for v in same_dim if abs(v.canonical - gold_canonical) <= tolerance]
    if matches:
        others = {v.canonical for v in same_dim} - {v.canonical for v in matches}
        if others:
            return NumericResult("ambiguous")
        return NumericResult("match", canonical=str(matches[0].canonical))
    if any(v.value == gold_number for v in values):
        return NumericResult("unit_mismatch")
    return NumericResult("number_mismatch")
