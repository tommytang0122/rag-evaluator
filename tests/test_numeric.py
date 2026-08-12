from decimal import Decimal

from rag_evaluator.eval.numeric import (
    DEFAULT_UNITS,
    answer_signature,
    compare_numeric,
    extract_values,
    gold_dimension,
)

D = Decimal


def test_extract_thousands_fullwidth_and_units():
    vals = extract_values("營收為１２，４１５千元,另有 3.5% 成長,共 2025 年")
    assert [(v.value, v.dimension) for v in vals] == [
        (D("12415"), "money"),
        (D("3.5"), "percent"),
        (D("2025"), "year"),
    ]
    assert vals[0].canonical == D("12415000")  # 千元 → 元


def test_extract_ntd_unit_and_negative():
    vals = extract_values("-1,200 NTD千元")
    assert vals[0].value == D("-1200")
    assert vals[0].dimension == "money"
    assert vals[0].canonical == D("-1200000")


def test_compare_match():
    r = compare_numeric("約為 12,415 千元", D("12415"), "千元")
    assert r.status == "match"
    assert r.canonical == "12415000"


def test_compare_match_cross_unit():
    # 12,415,000 元 == 12,415 千元 canonically
    assert compare_numeric("12,415,000 元", D("12415"), "千元").status == "match"


def test_compare_unit_mismatch():
    # right number, wrong/missing unit
    assert compare_numeric("12,415 元", D("12415"), "千元").status == "unit_mismatch"
    assert compare_numeric("大約 12,415", D("12415"), "千元").status == "unit_mismatch"


def test_compare_number_mismatch():
    assert compare_numeric("13,000 千元", D("12415"), "千元").status == "number_mismatch"


def test_compare_no_number():
    assert compare_numeric("找不到相關資訊", D("12415"), "千元").status == "no_number"


def test_compare_ambiguous():
    r = compare_numeric("上半年 12,415 千元,下半年 13,000 千元", D("12415"), "千元")
    assert r.status == "ambiguous"


def test_compare_unitless_gold():
    assert compare_numeric("共 42 件", D("42"), None).status == "match"


def test_years_do_not_pollute_money_dimension():
    # 2025 (none) must not be a contradicting money value
    r = compare_numeric("2025年1月營收 12,415 千元", D("12415"), "千元")
    assert r.status == "match"


def test_tolerance():
    r = compare_numeric(
        "12,414 千元", D("12415"), "千元", tolerance=D("1000")
    )  # canonical diff 1000 ≤ tolerance
    assert r.status == "match"


def test_comma_adjacent_numbers_not_merged():
    vals = extract_values("3,500,2,100")
    assert [v.value for v in vals] == [D("3500"), D("2100")]


def test_answer_signature_scoped_to_dimension_excludes_incidental_page_number():
    sig = answer_signature("營收 12,415 千元,詳見第 3 頁", dimension="money")
    assert sig == "12415000"


def test_answer_signature_scoped_to_dimension_matches_across_incidental_numbers():
    sig = answer_signature("13,000 千元", dimension="money")
    assert sig == "13000000"


def test_answer_signature_no_numbers():
    assert answer_signature("no numbers here") == ""


def test_answer_signature_no_dimension_keeps_all_numbers():
    sig = answer_signature("共 12,415 千元 與 3 頁")
    parts = sig.split(",")
    assert "12415000" in parts
    assert "3" in parts


def test_gold_dimension():
    assert gold_dimension("千元") == "money"
    assert gold_dimension(None) == "none"


def test_usd_and_year_units_match():
    from decimal import Decimal as D
    from rag_evaluator.eval.numeric import compare_numeric
    assert compare_numeric("累計虧損3.70億美元", D("3.7"), "億美元").status == "match"
    assert compare_numeric("約1,560萬美元", D("1560"), "萬美元").status == "match"
    assert compare_numeric("1996年提出", D("1996"), "年").status == "match"
    # 台幣與美元不同維度:數字相同也不可 match
    assert compare_numeric("12,415 千元", D("12415"), "千美元").status != "match"


def test_unknown_gold_unit_returns_unknown_unit():
    from decimal import Decimal as D
    from rag_evaluator.eval.numeric import compare_numeric
    assert compare_numeric("3.70 光年", D("3.7"), "光年").status == "unknown_unit"
