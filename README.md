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

## 安裝與設定

```bash
uv sync                      # 或 uv pip install -e ".[dev]"
cp .env.example .env         # 填 GEMINI_API_KEY;其餘預設值可直接用
```

`.env.example` 同時是「換一個受測 RAG 系統」的 checklist:五個區段依序是 judge 金鑰、Qdrant URL、system.yaml 要填什麼、黃金集怎麼建、換系統要檢查的內建假設。

受測系統的連線設定寫在 `system.yaml`(不放 .env,因為 run manifest 會釘住它做溯源):

```yaml
adapter: nas_rag                            # 註冊於 src/rag_evaluator/adapters/
endpoint: http://localhost:8020/v1/query    # 必須含完整路徑
collection_names: [gavin_test]
top_k: 5
timeout_s: 120
```

只拿得到前端網站時,`endpoint` 改成 `http://<web-host>/api/v1/query` 即可(nginx 會剝 `/api` 前綴反代給 backend,契約相同)。

## 評測流程

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
