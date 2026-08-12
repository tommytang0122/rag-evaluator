# 手動煙霧測試前置作業指南（給執行代理，如 Codex）

本文件是自足的：不需要其他對話上下文即可照做。目標是在本機完成 rag-evaluator 對
`project-nas-rag` 的第一次端到端煙霧測試（2–3 題），驗證「執行器 → 評分 → 報告」整條管線可通。

**這不是正式評測**。正式評測需要 30–50 題人工黃金集（見 `RAG-EVALUATION-DESIGN.md` §7），
煙霧測試只驗證管線本身。

---

## 執行守則（必讀）

1. **gold 答案絕不可自己編**。出題時必須親自打開對應頁面 PNG 看到那個數字；
   若無法讀圖，就把候選題與頁面路徑列出來，**停下來請人類確認 gold 值**再繼續。
2. 任何服務起不來、curl 不通，先回報狀態與錯誤訊息，不要盲目重試或改別人專案的程式碼。
3. 不要 commit 任何含 API key 的檔案（`.env` 已在慣例上不入版控）。
4. 每個步驟都有「驗收」小節——通過才往下走。

---

## 步驟 0：環境健檢

工作目錄：`~/workspace/rag-evaluator`（本 repo）。受測系統在 `~/workspace/project-nas-rag`。

```bash
cd ~/workspace/rag-evaluator
uv run rag-eval --help          # 應列出 corpus/dataset/collect/score/report/run
ls ~/workspace/project-nas-rag/output/images/gavin_test/   # 應列出數個 PDF 資料夾
```

**驗收**：兩個指令都有輸出、無錯誤。

## 步驟 1：把 nas-rag 問答鏈跑起來

qa_api 是一支 FastAPI 服務，檔案在 `~/workspace/project-nas-rag/docs/archive/qa_api.py`，
監聽 **port 8020**，`POST /v1/query` 回 `{answer, sources[]}`。

它的依賴（由 `project-nas-rag/.env` 的環境變數指定，缺一不可）：

| 依賴 | 環境變數 / 預設 | 健檢指令 |
|---|---|---|
| Qdrant | localhost:6333 | `curl -s http://localhost:6333/collections`（回 JSON 且含 `gavin_test`） |
| Embedding API | `EMBEDDING_API_URL` | 對該 URL 打一筆測試 embedding |
| Reranker API | `RERANKER_API_URL` | 同上 |
| Ollama + gemma4:31b | `OLLAMA_URL` | `curl -s $OLLAMA_URL/api/tags`（model 清單含 gemma4） |

Qdrant 若沒起，README 的啟法：

```bash
docker run -d -p 6333:6333 -p 6334:6334 --name nas-qdrant qdrant/qdrant
```

> Qdrant 容器若是新起的，裡面不會有 `gavin_test` collection——那需要先跑過 nas-rag 的
> upload pipeline。如果 `collections` 回傳裡沒有 `gavin_test`，**停下來回報**，
> 不要自己重跑 ingestion。

依賴都通了之後啟動 qa_api（在 project-nas-rag 的環境裡）：

```bash
cd ~/workspace/project-nas-rag
python docs/archive/qa_api.py    # 或以該專案慣用的方式啟動
```

**驗收**：

```bash
curl -s -X POST http://localhost:8020/v1/query -H 'Content-Type: application/json' \
  -d '{"query":"2023年9月的分廠損益狀況如何？","collection_names":["gavin_test"],"top_k":3}'
```

回傳 JSON 含非空 `answer` 與 `sources[]`，且 sources 元素有 `source` 與 `page` 欄位。
**把一筆完整回傳存檔**（後面出題要對照 `source` 的確切檔名格式）：

```bash
curl -s -X POST http://localhost:8020/v1/query -H 'Content-Type: application/json' \
  -d '{"query":"2023年9月的分廠損益狀況如何？","collection_names":["gavin_test"],"top_k":5}' \
  > /tmp/probe-response.json
```

## 步驟 2：建 `system.yaml`

在 rag-evaluator repo 根目錄建立：

```yaml
adapter: nas_rag        # 底線，不是 nas-rag（見 src/rag_evaluator/adapters/nas_rag.py 的 ADAPTERS）
endpoint: http://localhost:8020/v1/query   # 必須含完整路徑！adapter 直接 POST 這個 URL，不會自動補 /v1/query
collection_names: [gavin_test]
top_k: 5
timeout_s: 120          # gemma4:31b 在本機可能很慢，放寬一點
```

**驗收**：`uv run python -c "from rag_evaluator.config import load_system_config; from pathlib import Path; print(load_system_config(Path('system.yaml')))"` 印出 config、無錯誤。

## 步驟 3：出 2–3 題煙霧黃金集 `smoke.jsonl`

格式定義在 `src/rag_evaluator/dataset/models.py`（`DatasetItem`）。規則：

- `answer_type` 只能是 `answerable` 或 `refusal`
- `refusal` 題**不可**帶 `evidence` 或 `gold_value`（validator 會擋）
- `evidence[].document` 必須與步驟 1 存檔的 `sources[].source` **檔名格式一致**
  （先看 probe-response.json 確認是完整路徑還是純檔名）
- `id` 不可重複

出題流程（**人機分工的關鍵步驟**）：

1. 從 `~/workspace/project-nas-rag/output/images/gavin_test/<PDF名>/page_<N>.png` 挑 1–2 頁
2. 讀圖，選一個明確、單頁可答的數字（金額欄位最好），記下「數字、單位、頁碼、檔名」
3. **若無法讀圖**：列出挑選的頁面路徑與候選問題，停下來請人類看圖填 gold 值
4. 另加一題確定不可答的 refusal 題（問語料裡不存在的年份/公司即可）

範本（gold 值以實際頁面為準）：

```jsonl
{"id": "smoke-001", "question": "2023年9月○○廠的營業收入是多少？", "answer_type": "answerable", "gold_answer": "<數字> 千元", "gold_value": {"number": <數字>, "unit": "千元"}, "evidence": [{"document": "<與sources一致的檔名>", "page": <N>, "collection": "gavin_test"}], "tags": ["numeric", "single-page"]}
{"id": "smoke-002", "question": "2030年火星廠區的營業收入是多少？", "answer_type": "refusal", "gold_answer": "", "tags": ["unanswerable"]}
```

**驗收**：`uv run python -c "from rag_evaluator.dataset.models import load_dataset; from pathlib import Path; print(load_dataset(Path('smoke.jsonl')))"` 載入成功。

## 步驟 4：`.env` 設 judge 金鑰

`score` 階段的 judge 用 Gemini。在 rag-evaluator repo 根目錄的 `.env` 放：

```
GEMINI_API_KEY=<key>
```

金鑰請向人類要，或確認 `project-nas-rag/.env` 是否已有可共用的。
沒有金鑰時：**步驟 5 的 collect 仍可做**（零 LLM 依賴），score/report 暫停並回報。

## 步驟 5：分段執行與驗收

```bash
cd ~/workspace/rag-evaluator

# 5a. 收集（只打 nas-rag，不用 LLM judge）
uv run rag-eval collect --system system.yaml --dataset smoke.jsonl --run-id smoke-01
```

**驗收 5a**：`runs/smoke-01/raw.jsonl` 存在；每題有 `kind:"answer"` 的列，
`answer` 非空、`error` 為空、`latency_ms` 合理。**把每題的 answer 摘出來給人類過目**。

```bash
# 5b. 評分（需要 GEMINI_API_KEY）
uv run rag-eval score --dataset smoke.jsonl --run-id smoke-01

# 5c. 報告
uv run rag-eval report --dataset smoke.jsonl --run-id smoke-01
```

**驗收 5b/5c**：`runs/smoke-01/scores.jsonl` 每題一列，欄位含 `correctness`、`retrieval`、
`faithfulness_status`；`runs/smoke-01/report.md` 產出。最後**貼出 report.md 內容**。

### 結果怎麼判讀

- answerable 題：`retrieval.evidence_recall` 是否為 1（gold 頁有被檢回）、
  `correctness` 是否為 2 且 `method` 是 `rule_numeric`（走到規則層代表數值抽取正常）
- refusal 題：`correctness` 2 = 系統正確拒答；0 + `hallucinated_answer: true` = 系統編了答案
  （這是**受測系統**的問題，不是評分器壞掉——如實回報即可）
- `judge_error: true` 或 `faithfulness_status: "judge_error"` → 檢查 GEMINI_API_KEY 與網路

## 已知陷阱

- **dataset 或 system.yaml 改了就要換 `--run-id`**：manifest 會釘住 dataset 的 SHA
  與 system_config（`cli.py`），改動後對舊 run-id 執行會報 manifest mismatch。這是設計行為。
- **adapter 名是 `nas_rag`**（底線）。
- **忠實度的圖片升級在煙霧測試不會發生**：沒給 `--corpus` 時 judge 只有 sources 附帶的
  文字可用，部分論斷會記 `evidence_unavailable`，屬預期。建 corpus 首選
  `uv run rag-eval corpus from-qdrant --collection gavin_test -o corpus.jsonl`
  （直接 scroll Qdrant REST，payload 即最終欄位，無下述 manifest 陷阱），
  score 時加 `--corpus corpus.jsonl`（同一 run-id 第一次帶 corpus 會把
  corpus SHA 也釘進 manifest）。
  備援是 `corpus from-nas-rag --manifest .../output/verification_gavin_test.jsonl`，
  但注意兩個 manifest 格式陷阱（2026-07-27 實測）：verification jsonl 每列
  **沒有 `collection` 欄位**（轉換會 KeyError，需先逐列補上）；且該檔是 append
  模式，重跑 pipeline 後會殘留舊列（同一 (source, page) 取最後一筆去重後再轉）。
- **只拿得到前端網站也能評**：前端是純靜態 SPA，同源打 `/api/v1/query`，由 nginx
  剝掉 `/api` 前綴反代到 backend（`deploy/nginx-nas-rag.conf`），request/response
  契約與 `nas_rag` adapter 完全相同（含 top_k，截斷 probe 也可用）。把
  `system.yaml` 的 `endpoint` 改成 `http://<web-host>/api/v1/query` 即可，
  不需要新 adapter。
- **本機 `nas-qdrant` container 沒掛 volume**：資料存在 container 可寫層，
  Qdrant 啟動時會警告 storage 可能遺失（2026-08-12 實測 gavin_test 已消失）。
  重灌後建議 `docker run -v` 掛實體目錄。
- **gemma4 temperature=1.0**：同題重問答案會變。煙霧測試 `--runs 1` 即可；
  正式評測才需要 `--runs 3`。

## 完成後回報清單

1. 各依賴服務最終狀態（哪些原本沒起、怎麼起的）
2. smoke.jsonl 內容與 gold 值的出處（哪個 PNG 哪一頁）
3. raw.jsonl 每題的 answer 摘要
4. report.md 全文
5. 過程中任何停下來等人類決定的點
