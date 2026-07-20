from rag_evaluator.eval.refusal import (
    DEFAULT_REFUSAL_PHRASES,
    classify_refusal,
    refusal_phrases_hash,
)


def test_pure_refusal():
    assert classify_refusal("找不到相關資訊。") == "pure_refusal"
    assert classify_refusal("找不到任何相關的資料！") == "pure_refusal"


def test_substantive_answer():
    assert classify_refusal("2025年1月營收為 12,415 千元。") == "substantive_answer"


def test_mixed_refusal_answer():
    # 含拒答片語但仍給了數字 → 不可算正確拒答
    assert (
        classify_refusal("找不到原始資料,但依推測答案是 12,415 千元。")
        == "mixed_refusal_answer"
    )


def test_empty_non_answer():
    assert classify_refusal("") == "empty_non_answer"
    assert classify_refusal("  ") == "empty_non_answer"


def test_custom_phrases():
    assert (
        classify_refusal("no relevant info found", phrases=("no relevant info found",))
        == "pure_refusal"
    )


def test_phrases_hash_stable():
    h1 = refusal_phrases_hash(DEFAULT_REFUSAL_PHRASES)
    h2 = refusal_phrases_hash(tuple(DEFAULT_REFUSAL_PHRASES))
    assert h1 == h2 and len(h1) == 64
