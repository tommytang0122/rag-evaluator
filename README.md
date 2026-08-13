# rag-evaluator

RAG 回答品質評分 CLI:檢索指標(規則)+ 答案正確性(規則優先、LLM 兜底)+ 多模態忠實度(Gemini judge)。原針對 `project-nas-rag`(中文企業財報 PDF → Qdrant 多模態 RAG)設計,但執行器與評分引擎解耦,任何符合 `ask(question) → {answer, sources[]}` 介面的系統都能評。

設計依據與指標定義見 [RAG-EVALUATION-DESIGN.md](RAG-EVALUATION-DESIGN.md)。

```
┌──────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────┐
│ 黃金集     │──▶│ collect      │──▶│ score          │──▶│ report   │
│ golden    │   │ 打受測系統    │   │ A 檢索(規則)   │   │ 兩軌報表  │
│ .jsonl   │   │ n 次 + probe  │   │ B 正確性(規則) │   │ 診斷/迴歸 │
│          │   │ 落盤 raw     │   │ C 忠實度(多模態)│   │          │
└──────────┘   └──────────────┘   └───────────────┘   └──────────┘
```

## 一鍵開跑

```bash
uv sync                      # 或 uv pip install -e ".[dev]"
cp .env.example .env         # 填下表的值
./run-eval.sh                # run-id 省略時用 prod-<日期時間>
```

`run-eval.sh` 依序做:讀 `.env` → 生成 `runs/<run-id>/system.yaml` → **preflight**(打一發真的 query,驗回傳符合 adapter 契約)→ 備妥 corpus → collect → score → report。preflight 是為了讓 endpoint 打錯、token 過期、憑證不對這類問題在**跑整份題庫之前**就明確報出來,而不是收完一堆 `system_error` 才回頭查。

```bash
./run-eval.sh prod-01 --runs 3 --dataset golden.jsonl --baseline prod-00
```

| 選項 | 預設 | 說明 |
|---|---|---|
| `[run-id]` | `prod-<YYYYMMDD-HHMM>` | 改了 `.env` 或黃金集就要換一個 |
| `--runs N` | 3 | 同題重問次數;煙霧測試用 1 |
| `--dataset f` | `golden.jsonl` | 黃金集 |
| `--corpus f` | `corpus.jsonl` | 不存在時會試著從 Qdrant 建 |
| `--baseline id` | 無 | 與基準 run-id 做迴歸比較 |
| `--skip-preflight` | 關 | 跳過契約檢查 |
| `ENV_FILE=path` | `.env` | 換一份環境設定檔 |

### 人類要提供什麼

`.env` 裡只有這幾個是**必填**,其餘留預設即可(完整說明見 `.env.example`):

| 變數 | 說明 |
|---|---|
| `GEMINI_API_KEY` | judge 用。judge 要與受測系統的回答模型不同家,避免同源偏袒 |
| `RAG_EVAL_ENDPOINT` | **必須含完整路徑**,如 `http://host/api/v1/query`——adapter 直接 POST 這個 URL,不會自動補 `/v1/query` |
| `RAG_EVAL_COLLECTION` | 逗號分隔可多個 |

視環境選填:

| 變數 | 預設 | 何時要設 |
|---|---|---|
| `RAG_EVAL_AUTH_TOKEN` | 空(不送) | 端點需要認證。值原樣放進 header,**不會自動補 `Bearer `**,所以 `Bearer xxx` 與純 API key 都能表達 |
| `RAG_EVAL_AUTH_HEADER` | `Authorization` | 改成 `X-API-Key` 之類 |
| `RAG_EVAL_TLS_VERIFY` | `true` | 內網自簽憑證填 `false`,或填 CA bundle 路徑 |
| `RAG_EVAL_TOP_K` / `RAG_EVAL_TIMEOUT_S` | 5 / 300 | 受測系統慢就放寬 timeout |
| `RAG_EVAL_CUTOFF_PROBE_TOP_K` | 20 | 以更大的 top_k 再打一次,歸因「檢索沒中是被截斷還是排名太低」 |
| `RAG_EVAL_QDRANT_URL` | `http://localhost:6333` | 沒有 `corpus.jsonl` 時,script 從這裡建 |

**token 不會落盤**:`system.yaml` 與 run manifest 只記 `auth_env: RAG_EVAL_AUTH_TOKEN` 這個**變數名稱**,值僅在建 HTTP client 時從環境解析。manifest 會把整份 system_config 原樣寫進評測紀錄,所以秘密絕不能以明文進設定檔。

需要逐步 debug 或換受測系統時,走下面的分步流程。

## 分步流程

受測系統的連線設定寫在 `system.yaml`(`run-eval.sh` 生成的那份在 `runs/<run-id>/system.yaml`,repo 根目錄這份留給手動流程):

```yaml
adapter: nas_rag                            # 註冊於 src/rag_evaluator/adapters/
endpoint: http://localhost:8020/v1/query    # 必須含完整路徑
collection_names: [gavin_test]
top_k: 5
timeout_s: 120
auth_env: RAG_EVAL_AUTH_TOKEN               # 可省略;存變數名而非 token 本身
auth_header: Authorization
verify: true                                # false=略過驗證,或填 CA bundle 路徑
```

只拿得到前端網站時,`endpoint` 改成 `http://<web-host>/api/v1/query` 即可(nginx 會剝 `/api` 前綴反代給 backend,契約相同)。

`.env.example` 同時是「換一個受測 RAG 系統」的 checklist:五個區段依序是 judge 金鑰、Qdrant URL、受測系統連線、黃金集怎麼建、換系統要檢查的內建假設。

### 1. 建 corpus(忠實度證據來源;可略過)

```bash
# 首選:直接 scroll Qdrant REST(payload 即最終欄位)
uv run rag-eval corpus from-qdrant --collection gavin_test -o corpus.jsonl

# 備援:從 nas-rag 的 verification manifest 轉
uv run rag-eval corpus from-nas-rag --manifest .../verification_gavin_test.jsonl -o corpus.jsonl
```

沒有 corpus 評測照跑,但忠實度只能用 sources 附帶的文字,部分論斷會記 `evidence_unavailable`;有 corpus 且拿得到頁面 PNG 時,判不出的論斷會升級送原圖給多模態 judge。

### 2. 建黃金集(唯一需要人力的部分)

出題由 `generate-golden-questions` agent skill 負責(agent 親讀頁面 PNG 產生 review.csv),人工填 `approved` 欄後定稿:

```bash
uv run rag-eval dataset finalize --review review.csv -o golden.jsonl
```

建議 30–50 題,必含跨頁題與不可答題。現有題庫的可讀版:`docs/golden-set.html`(由 `docs/build_golden_set_html.py` 生成,可用 `--scores runs/<id>/scores.jsonl` 附上評測結果)。

### 3. collect:打受測系統、落盤原始記錄

```bash
uv run rag-eval collect --system system.yaml --dataset golden.jsonl \
  --run-id golden-03 --runs 3
```

- `--runs 3`:同題重問 n 次(受測系統 temperature 高時必要,報 pass@1/pass@3/一致率);煙霧測試用 1。
- 可中斷續跑:重跑同一 run-id 會跳過已完成的題。
- 若 system.yaml 的 `diagnostics.cutoff_probe_top_k` 有設,另以大 top_k 打一次做截斷歸因。
- 連線失敗會重試 2 次後記為 `system_error: HTTP 401 {...}`,錯誤細節寫進 `raw.jsonl` 的 `error` 欄,不必回頭手動 curl 猜原因。

### 4. score:離線評分

```bash
uv run rag-eval score --dataset golden.jsonl --run-id golden-03 --corpus corpus.jsonl
```

- 需要 `GEMINI_API_KEY`(judge 預設 `gemini-2.5-flash`,`--model` 可換)。
- 改了 judge prompt 之後重評:加 `--rescore-tag v2`,會另寫 `scores-v2.jsonl` 與 sidecar manifest,不覆蓋原分數。

### 5. report:產出 markdown 報表

```bash
uv run rag-eval report --dataset golden.jsonl --run-id golden-03 \
  --baseline golden-02             # 可選:與基準 run-id 做 bootstrap 迴歸比較
```

產出 `runs/<run-id>/report.md`:總覽指標、tag 分桶、截斷/type 診斷、失敗案例。

### 一鍵跑完(collect+score+report)

```bash
uv run rag-eval run --system system.yaml --dataset golden.jsonl \
  --run-id golden-03 --runs 3 --corpus corpus.jsonl
```

## 輸出與溯源

每個 run 落在 `runs/<run-id>/`:

| 檔案 | 內容 |
|---|---|
| `system.yaml` | 由 `.env` 生成的受測系統設定(僅 `run-eval.sh` 產出) |
| `raw.jsonl` | 每題每次的 Q/A/sources/latency 原始記錄 |
| `scores.jsonl` | 逐題評分(含 judge 理由、prompt 版本) |
| `report.md` | 彙總報表 |
| `run_manifest.json` | 溯源:dataset/corpus SHA、system_config、evaluator 版本 |

manifest 釘住 dataset 與 system.yaml:**改了任一個就要換 `--run-id`**,對舊 run-id 執行會報 mismatch(設計行為,exit code 2)。

## 換一個受測 RAG 系統要提供什麼

1. **介面**:API 契約相同 → 只改 system.yaml;不同 → 仿 `adapters/nas_rag.py` 寫 ~50 行 adapter 註冊進 `ADAPTERS`,sources 最低要求每筆有 `(document, page)`。
2. **黃金集**:新語料重新出題。
3. **三個內建假設要檢查**:檢索單位是「頁」(不是的話要動核心)、拒答句式(`eval/refusal.py`,寫死中文)、數值單位字典(`eval/numeric.py`,偏台灣財報)。

完整 checklist 見 `.env.example`。

## 其他文件

- [RAG-EVALUATION-DESIGN.md](RAG-EVALUATION-DESIGN.md) — 指標設計與取捨
- [docs/manual-smoke-test.md](docs/manual-smoke-test.md) — 本地端到端煙霧測試 runbook(含已知陷阱)
- `docs/golden-set.html` — 黃金集可讀版

## 開發

```bash
uv run pytest -q
```
