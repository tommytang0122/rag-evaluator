from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from rag_evaluator.judge import GeminiJudge, JudgeError


class Verdict(BaseModel):
    score: int = Field(ge=0, le=2)
    reason: str = Field(min_length=1)


def _mk_judge(mock_genai, responses):
    """responses: list of .text values returned by successive generate_content calls."""
    client = MagicMock()
    mock_genai.Client.return_value = client
    client.models.generate_content.side_effect = [
        MagicMock(text=t) for t in responses
    ]
    sleeps = []
    judge = GeminiJudge(
        "gemini-test", api_key="fake", qps=0, _sleep=sleeps.append, _clock=lambda: 0.0
    )
    return judge, client, sleeps


@patch("rag_evaluator.judge.genai")
def test_judge_parses_structured_output(mock_genai):
    judge, client, _ = _mk_judge(mock_genai, ['{"score": 2, "reason": "exact"}'])
    v = judge.judge("prompt", Verdict)
    assert v.score == 2
    cfg = client.models.generate_content.call_args.kwargs["config"]
    assert cfg["response_mime_type"] == "application/json"


@patch("rag_evaluator.judge.genai")
def test_judge_retries_on_invalid_json_then_succeeds(mock_genai):
    judge, client, sleeps = _mk_judge(
        mock_genai, ["not json", '{"score": 5, "reason": "x"}', '{"score": 1, "reason": "ok"}']
    )
    v = judge.judge("prompt", Verdict)
    assert v.score == 1
    assert client.models.generate_content.call_count == 3
    assert sleeps == [1.0, 2.0]  # exponential backoff


@patch("rag_evaluator.judge.genai")
def test_judge_raises_after_max_retries(mock_genai):
    judge, _, _ = _mk_judge(mock_genai, ["bad"] * 4)
    with pytest.raises(JudgeError):
        judge.judge("prompt", Verdict)


@patch("rag_evaluator.judge.genai")
def test_judge_requires_api_key(mock_genai, monkeypatch):
    monkeypatch.setattr("rag_evaluator.judge.load_dotenv", lambda: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(JudgeError, match="GEMINI_API_KEY"):
        GeminiJudge("gemini-test")


@patch("rag_evaluator.judge.genai")
def test_judge_sends_images_as_parts(mock_genai, tmp_path):
    img = tmp_path / "page_3.png"
    img.write_bytes(b"\x89PNG fake")
    judge, client, _ = _mk_judge(mock_genai, ['{"score": 0, "reason": "r"}'])
    judge.judge("prompt", Verdict, images=[img])
    contents = client.models.generate_content.call_args.kwargs["contents"]
    assert contents[0] == "prompt"
    mock_genai.types.Part.from_bytes.assert_called_once_with(
        data=b"\x89PNG fake", mime_type="image/png"
    )


@patch("rag_evaluator.judge.genai")
def test_judge_throttles_by_qps(mock_genai):
    client = MagicMock()
    mock_genai.Client.return_value = client
    client.models.generate_content.return_value = MagicMock(
        text='{"score": 1, "reason": "r"}'
    )
    sleeps = []
    clock = iter([0.0, 0.0, 0.1, 0.1]).__next__  # second call 0.1s after first
    judge = GeminiJudge(
        "gemini-test", api_key="fake", qps=2, _sleep=sleeps.append, _clock=clock
    )  # min interval 0.5s
    judge.judge("p1", Verdict)
    judge.judge("p2", Verdict)
    assert pytest.approx(sleeps[-1], abs=1e-6) == 0.4  # waited 0.5 - 0.1
