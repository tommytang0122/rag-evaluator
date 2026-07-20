from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class JudgeError(Exception):
    pass


class GeminiJudge:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        qps: float = 1.0,
        max_retries: int = 3,
        _sleep: Callable[[float], None] = time.sleep,
        _clock: Callable[[], float] = time.monotonic,
    ):
        load_dotenv()
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise JudgeError("GEMINI_API_KEY not set")
        self._client = genai.Client(api_key=key)
        self._model = model
        self._max_retries = max_retries
        self._min_interval = 1.0 / qps if qps > 0 else 0.0
        self._sleep = _sleep
        self._clock = _clock
        self._last_call: float | None = None

    def _throttle(self) -> None:
        if self._min_interval <= 0:
            return
        now = self._clock()
        if self._last_call is not None:
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                self._sleep(wait)
        self._last_call = self._clock()

    def judge(
        self, prompt: str, schema: type[T], images: Sequence[Path] = ()
    ) -> T:
        contents: list = [prompt]
        for img in images:
            contents.append(
                genai.types.Part.from_bytes(
                    data=Path(img).read_bytes(), mime_type="image/png"
                )
            )
        delay = 1.0
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                self._sleep(delay)
                delay *= 2
            self._throttle()
            try:
                resp = self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config={"response_mime_type": "application/json"},
                )
                return schema.model_validate_json(resp.text)
            except Exception as exc:  # noqa: BLE001 - API and validation errors retry
                last_error = exc
        raise JudgeError(f"judge failed after {self._max_retries} retries: {last_error}")
