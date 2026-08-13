from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2] / "project-nas-rag"


@dataclass(frozen=True)
class Settings:
    embedding_url: str
    qdrant_url: str
    gemini_model: str
    timeout_s: float


class GeminiQaService:
    def __init__(self, settings: Settings) -> None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.settings = settings
        self.http = httpx.Client(timeout=settings.timeout_s, trust_env=False)
        self.gemini = genai.Client(api_key=api_key)

    def close(self) -> None:
        self.http.close()

    def query(
        self, question: str, collection_names: list[str], top_k: int
    ) -> dict[str, Any]:
        vector = self._embed(question)
        matches = self._retrieve(vector, collection_names, top_k)
        if not matches:
            return {"answer": "找不到任何相關的資料！", "sources": []}

        prompt, image_parts = self._build_context(question, matches)
        response = self.gemini.models.generate_content(
            model=self.settings.gemini_model,
            contents=[prompt, *image_parts],
            config=types.GenerateContentConfig(temperature=0.1),
        )
        answer = (response.text or "").strip()
        if not answer:
            raise RuntimeError("Gemini returned an empty answer")
        return {
            "answer": answer,
            "sources": [match["source"] for match in matches],
        }

    def _embed(self, question: str) -> list[float]:
        response = self.http.post(
            self.settings.embedding_url,
            json={"inputs": [{"text": question}]},
        )
        response.raise_for_status()
        vector = response.json()["data"][0]["vector"]
        if len(vector) != 2048:
            raise RuntimeError(
                f"Embedding API returned {len(vector)} dimensions; expected 2048"
            )
        return vector

    def _retrieve(
        self, vector: list[float], collection_names: list[str], top_k: int
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for collection in collection_names:
            url = (
                f"{self.settings.qdrant_url.rstrip('/')}/collections/"
                f"{quote(collection, safe='')}/points/query"
            )
            response = self.http.post(
                url,
                json={
                    "query": vector,
                    "limit": 50,
                    "with_payload": True,
                    "with_vector": False,
                },
            )
            if response.status_code == 404:
                continue
            response.raise_for_status()
            for point in response.json().get("result", {}).get("points", []):
                payload = dict(point.get("payload") or {})
                score = float(point.get("score") or 0.0)
                payload["collection"] = collection
                payload["rerank_score"] = score
                candidates.append({"score": score, "source": payload})

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:top_k]

    @staticmethod
    def _build_context(
        question: str, matches: list[dict[str, Any]]
    ) -> tuple[str, list[types.Part]]:
        text_sections: list[str] = []
        image_parts: list[types.Part] = []

        for index, match in enumerate(matches, start=1):
            payload = match["source"]
            heading = (
                f"【第 {index} 筆資料】來源：{payload.get('source', '未知')}，"
                f"頁碼：{payload.get('page', '未知')}，"
                f"單位：{payload.get('unit', '未標示')}"
            )
            text_sections.append(heading)

            content = payload.get("content") or payload.get("schema_text")
            if content:
                text_sections.append(str(content))

            image_path = payload.get("image_path")
            if image_path:
                path = Path(str(image_path))
                if path.is_file():
                    image_parts.append(
                        types.Part.from_bytes(
                            data=path.read_bytes(),
                            mime_type="image/png",
                        )
                    )

        context = "\n\n".join(text_sections)
        prompt = f"""你是企業文件問答助理。請只根據下列檢索文字與隨附頁面圖片回答。

規則：
1. 直接回答，不輸出推理過程。
2. 數字必須附上頁面標示的單位。
3. 找不到答案時，只回答「找不到相關資訊」。
4. 不可使用檢索資料以外的知識補答案。

【檢索文字】
{context}

【使用者問題】
{question}
"""
        return prompt, image_parts


class QueryHandler(BaseHTTPRequestHandler):
    service: GeminiQaService

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "model": self.service.settings.gemini_model,
                    "qdrant_url": self.service.settings.qdrant_url,
                },
            )
            return
        self._send_json(404, {"detail": "not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/query":
            self._send_json(404, {"detail": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            question = str(payload["query"]).strip()
            collections = payload["collection_names"]
            top_k = int(payload.get("top_k", 5))
            if not question:
                raise ValueError("query must not be empty")
            if not isinstance(collections, list) or not collections:
                raise ValueError("collection_names must be a non-empty list")
            if top_k < 1 or top_k > 20:
                raise ValueError("top_k must be between 1 and 20")
            result = self.service.query(
                question,
                [str(item) for item in collections],
                top_k,
            )
            self._send_json(200, result)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"detail": str(exc)})
        except Exception as exc:
            self.log_error("query failed: %s", exc)
            self._send_json(500, {"detail": f"{type(exc).__name__}: {exc}"})

    def log_message(self, message: str, *args: Any) -> None:
        print(f"{self.address_string()} - {message % args}", flush=True)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gemini-backed qa_api for rag-evaluator manual smoke tests"
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / ".env",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument(
        "--embedding-url",
        default=None,
        help="Overrides EMBEDDING_API_URL/NAS_RAG_LOCAL_TEXT_EMBEDDING_URL",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("QA_QDRANT_URL", "http://localhost:6333"),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("QA_GEMINI_MODEL", "gemini-3.6-flash"),
    )
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    embedding_url = (
        args.embedding_url
        or os.environ.get("EMBEDDING_API_URL")
        or os.environ.get("NAS_RAG_LOCAL_TEXT_EMBEDDING_URL")
    )
    if not embedding_url:
        raise RuntimeError("No embedding API URL is configured")

    settings = Settings(
        embedding_url=embedding_url,
        qdrant_url=args.qdrant_url,
        gemini_model=args.model,
        timeout_s=args.timeout_s,
    )
    service = GeminiQaService(settings)
    QueryHandler.service = service
    server = ThreadingHTTPServer((args.host, args.port), QueryHandler)
    print(
        f"Gemini qa_api listening on http://{args.host}:{args.port} "
        f"(model={args.model}, qdrant={args.qdrant_url})",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
