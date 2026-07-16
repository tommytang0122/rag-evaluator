# RAG 問答系統品質評測工具 — 設計文件

**日期**:2026-07-16
**狀態**:Draft
**專案**:rag-evaluator

---

## 1. 概述

### 1.1 目標

建立一個可重用的 RAG 問答系統品質評測 CLI 工具,涵蓋:

1. **黃金測試集生成**:從受測系統的文件語料自動生成 QA 對,經人工審核後定稿
2. **檢索層評測**:Hit@k、MRR、Citation Precision(純程式比對,零 LLM 成本)
3. **生成層評測**:Correctness、Faithfulness、Refusal Accuracy(Gemini LLM-as-judge)
4. **報告輸出**:Markdown 總結報告 + 逐題 JSONL 原始結果,支援 baseline 回歸比較

### 1.2 受測系統假設

- 近期有多個 RAG 系統要測,做法都類似 `project-nas-rag`:HTTP 問答 API + 向量庫,回答附帶引用來源(文件名 + 頁碼)
- 各系統服務**不同的文件庫**,各自有自己的測試集;評測目的是各系統的品質把關與迭代回歸,不做跨系統橫向比較
- 第一個 adapter 對接 nas-rag 的 `POST /v1/query`

### 1.3 關鍵決策

- **通用核心 + adapter**:評測邏輯只依賴 `RAGSystem` 抽象介面;每個受測系統一個 adapter
- **裁判用雲端強模型**:Gemini(專案環境已有 `GEMINI_API_KEY`)。裁判必須比受測系統強,分數才可信
- **不引入 RAGAS/DeepEval 等評測框架**:指標計算本身很簡單(幾十行),且 citation 頁碼比對這種需求框架沒有現成支援;引框架反而被其抽象綁住
- **一次 `ask()` 同時取得兩層評測素材**:nas-rag 類系統的回應中 `sources` 就是 rerank 後的 top-k 檢索結果,檢索指標直接從中計算,不需另連向量庫

---

## 2. 架構

### 2.1 專案結構

```
rag-evaluator/
├── src/rag_evaluator/
│   ├── adapters/
│   │   ├── base.py        # RAGSystem Protocol + RAGAnswer/RetrievedChunk 資料模型
│   │   └── nas_rag.py     # nas-rag adapter(HTTP /v1/query)
│   ├── dataset/
│   │   ├── corpus.py      # 語料 JSONL 讀取與驗證;nas-rag manifest 轉換器
│   │   └── generator.py   # LLM 生成 QA 對、匯出審核 CSV、定稿 dataset.jsonl
│   ├── eval/
│   │   ├── retrieval.py   # Hit@k、MRR、Citation Precision
│   │   ├── generation.py  # Correctness、Faithfulness、Refusal Accuracy
│   │   └── runner.py      # 逐題執行、結果落盤、斷點續跑
│   ├── judge.py           # Gemini 裁判封裝(結構化輸出、重試、節流)
│   ├── report.py          # report.md 產生、baseline 差值比較
│   ├── config.py          # 系統 YAML 設定載入(pydantic)
│   └── cli.py             # CLI 入口(argparse 子命令)
├── tests/                 # 每個模組一個 test_<module>.py
├── runs/                  # 評測輸出(gitignore)
└── pyproject.toml
```

### 2.2 RAGSystem 抽象介面

```python
@dataclass
class SourceRef:
    document: str          # 文件名
    page: int              # 頁碼
    score: float | None    # rerank/相似度分數(可選)

@dataclass
class RAGAnswer:
    answer: str
    sources: list[SourceRef]
    latency_ms: int

class RAGSystem(Protocol):
    def ask(self, question: str) -> RAGAnswer: ...

    # 可選能力:能直查檢索層(不經 LLM 生成)的系統才實作。
    # v1 的指標只用 ask() 的 sources;retrieve() 保留給未來評 rerank 前召回。
    def retrieve(self, question: str, k: int) -> list[SourceRef] | None: ...
```

### 2.3 系統設定 YAML

每個受測系統一份 YAML,由 `--system` 傳入:

```yaml
adapter: nas_rag                              # 對應 adapters/ 下的實作
endpoint: http://localhost:8020/v1/query
collection_names: [hr]
top_k: 5
timeout_s: 90
```

新系統若 API 格式與 nas-rag 相同,只需新增一份 YAML;格式不同則新增一個 adapter class 並在 YAML 指定 `adapter:` 名稱。

### 2.4 nas-rag adapter

- `POST {endpoint}`,body:`{"query": <question>, "collection_names": [...], "top_k": N}`
- 回應:`{"answer": str, "sources": [{"source": str, "page": int, "rerank_score": float, ...}]}`
- 映射:`source` → `SourceRef.document`,`page` → `SourceRef.page`,`rerank_score` → `SourceRef.score`

---

## 3. 黃金測試集生成

### 3.1 語料輸入(通用格式)

生成器不直接讀 PDF,吃標準化語料 JSONL,每行一頁:

```json
{"document": "差旅管理辦法_v3.pdf", "page": 5, "text": "頁面文字或摘要...", "collection": "hr"}
```

`corpus from-nas-rag` 子命令從 nas-rag 的 `output/manifest_<collection>.jsonl` 轉出此格式。其他系統只要能產出此格式即可共用生成器。

### 3.2 生成流程

1. **抽樣**:從語料抽 N 頁(`--sample-pages`,預設涵蓋每份文件至少一頁);`text` 過短(< 80 字)的頁面跳過
2. **LLM 生成**:每頁送 Gemini,產出 1–2 個「使用者真的會問的問題」+ 標準答案;prompt 強制答案必須能從該頁文字推出。題型配比:
   - **單頁事實題**:主力,約 70%
   - **表格/數字題**:該頁含表格特徵(Markdown 表格/多數字)時優先生成,約 15%
   - **不可回答題**:約 15%,對整體語料生成「聽起來相關但文件中沒有答案」的問題,黃金答案標記為拒答
3. **人工審核**:輸出 `review.csv`(欄位:question、gold_answer、answer_type、question_type、evidence 文件+頁碼、來源頁文字節錄、approved)。人工在試算表中修正或把 approved 設為否
4. **定稿**:`dataset finalize` 讀回 CSV,只保留 approved 的題目,輸出 `dataset.jsonl`

### 3.3 dataset.jsonl 格式

```json
{"id": "q001",
 "question": "國內出差住宿補助上限是多少?",
 "gold_answer": "每日 2,500 元",
 "answer_type": "answerable",
 "question_type": "fact",
 "evidence": [{"document": "差旅管理辦法_v3.pdf", "page": 5}]}
```

- `answer_type` ∈ {`answerable`, `refusal`};refusal 題 `evidence` 為空陣列、`gold_answer` 為空
- `question_type` ∈ {`fact`, `table`, `unanswerable`}
- `evidence` 為陣列,格式支援多頁(跨頁題可人工手寫加入;LLM 自動生成 v1 不做)

### 3.4 v1 不做

- 跨頁整合題的自動生成(LLM 生成品質差;格式已支援,人工可補)
- 多輪對話題(受測系統尚無多輪能力)

---

## 4. 評測指標

### 4.1 檢索層(純程式計算)

對每題 answerable 題,比對 `RAGAnswer.sources` 與黃金 `evidence`:

| 指標 | 定義 |
|---|---|
| Hit@k(k=1,3,5) | 任一 evidence 頁出現在前 k 個 sources 即為命中 |
| MRR | 第一個命中 evidence 的 source 排名倒數,全部題目取平均;未命中計 0 |
| Citation Precision | sources 中屬於 evidence 的比例,全部題目取平均 |

比對鍵為 `(document, page)`;document 正規化:去路徑、去副檔名、casefold,避免格式差異誤判。refusal 題不計入檢索指標。

### 4.2 生成層(Gemini 裁判)

| 指標 | 對象 | 方式 |
|---|---|---|
| Correctness(0/1/2) | answerable 題 | 裁判比對系統答案 vs 黃金答案:2=完全正確、1=部分正確或有遺漏、0=錯誤。報告呈現平均分(滿分 2)與各分數佔比 |
| Faithfulness(0–1) | answerable 題 | 裁判先從系統答案抽出事實陳述,逐條檢查是否有**系統實際引用的 sources 頁面**文字支撐,回傳有支撐比例。頁面文字以 `(document, page)` 從語料 JSONL(`run --corpus`)查得;查不到時改用系統回傳 source 中的內容欄位(如有),兩者皆無則該題標記 `faithfulness_skipped` |
| Refusal Accuracy | refusal 題 | 裁判二元判定系統是否正確拒答(表達「文件中找不到相關資訊」意涵) |

### 4.3 裁判設計

- 每次判定回傳結構化 JSON:`{score, reason}`;`reason` 一句話,存入結果供人工抽查
- 裁判 prompt 內含評分準則與 few-shot 範例
- 判 Correctness 時**不給裁判看 context**(只看問題 + 黃金答案 + 系統答案),避免被 context 帶偏;判 Faithfulness 時才提供 context
- Gemini 呼叫帶指數退避重試(3 次)與節流(可設定 QPS);單題判定失敗標記 `judge_error`,不中斷整體評測,不計入平均

---

## 5. 評測執行與報告

### 5.1 執行(runner)

- 逐題:呼叫 `ask()` → 計算檢索指標 → 裁判判定 → 逐題 append 寫入 `runs/<run-id>/results.jsonl`
- 每行 results.jsonl 含:題目 id、系統原始回答與 sources、各指標值、裁判理由、延遲、錯誤標記
- **斷點續跑**:重跑同一 `--run-id` 時,已存在於 results.jsonl 的題目跳過
- 受測系統呼叫失敗(timeout/5xx):重試 2 次,仍失敗標記 `system_error`,繼續下一題

### 5.2 報告(report.md)

- **總覽表**:Hit@1/3/5、MRR、Citation Precision、Correctness 平均、Faithfulness 平均、Refusal Accuracy、平均延遲、`system_error`/`judge_error` 題數
- **分項統計**:按 `question_type` 拆分各指標
- **最差案例清單**:Correctness=0 或檢索全 miss 的題目,列出問題、系統答案、黃金答案、裁判理由
- **baseline 比較**:`--baseline <run-id>` 時每個指標附差值(+/-)

### 5.3 CLI

```bash
rag-eval corpus from-nas-rag --manifest <path> -o corpus.jsonl
rag-eval dataset generate --corpus corpus.jsonl -o review.csv [--sample-pages N]
rag-eval dataset finalize --review review.csv -o dataset.jsonl
rag-eval run --system nas-rag.yaml --dataset dataset.jsonl --corpus corpus.jsonl \
             --run-id <id> [--baseline <run-id>]
# --corpus 供 Faithfulness 查引用頁面文字;省略時 Faithfulness 全部標記 faithfulness_skipped
```

---

## 6. 技術棧

| 項目 | 選型 | 說明 |
|---|---|---|
| 語言 | Python 3.11+ | 與 nas-rag 一致 |
| HTTP client | httpx | 呼叫受測系統 |
| LLM | google-genai(Gemini) | 測試題生成 + 裁判;`GEMINI_API_KEY` 由 `.env` 提供 |
| 資料模型 | pydantic v2 | dataset/corpus/config 驗證 |
| CLI | argparse 子命令 | 無額外依賴 |
| 測試 | pytest | LLM 呼叫以 mock 測;prompt 邏輯以 fixture 驗證 |

模型名稱、QPS、重試次數、抽樣參數等可調值集中於 `.env` 與系統 YAML,不散落在程式碼中(遵循 nas-rag 的設定分離原則)。

---

## 7. 錯誤處理

| 場景 | 處理 |
|---|---|
| 受測系統 timeout / 5xx | 重試 2 次後標記 `system_error`,繼續下一題 |
| 裁判 API 失敗 | 重試 3 次後標記 `judge_error`,不計入平均,報告列出題數 |
| dataset/corpus 格式錯誤 | 啟動時 pydantic 全檔驗證,先報錯不開跑 |
| 語料頁文字缺失(Faithfulness 無 context) | 標記 `faithfulness_skipped`,不計入平均 |
| run-id 已存在 | 續跑(跳過已完成題目);如需重跑請換 run-id 或刪除舊目錄 |

---

## 8. 初版範圍外(未來可擴展)

- `retrieve()` 介面的實際使用(評 rerank 前召回,需直連向量庫)
- 跨頁整合題自動生成
- 多輪對話評測
- HTML 報告 / 圖表
- 跨系統橫向比較報告
