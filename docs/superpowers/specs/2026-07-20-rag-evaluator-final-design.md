# RAG 問答系統品質評測工具 — 最終版設計文件

**日期**:2026-07-20
**狀態**:Approved
**專案**:rag-evaluator
**取代**:本文件合併並取代 `2026-07-16-rag-evaluator-design.md`(通用架構)與根目錄 `RAG-EVALUATION-DESIGN.md`(nas-rag 特化分析)兩份草稿。

---

## 1. 概述

### 1.1 目標

建立一個「**通用核心 + 每受測專案特化層**」的 RAG 問答品質評測 CLI 工具,涵蓋:

1. **黃金測試集生成**:從受測系統的文件語料自動生成 QA 對,經人工審核後定稿
2. **檢索層評測**:Hit@k、MRR、Citation Precision(純程式比對,零 LLM 成本)+ 特化診斷(截斷誤殺、type 分桶)
3. **生成層評測**:Correctness(規則優先、LLM 兜底)、Faithfulness(分層多模態裁判)、Refusal Accuracy
4. **報告輸出**:Markdown 總結報告 + 逐題 JSONL 原始結果,支援 baseline 回歸比較

### 1.2 受測系統假設

- 近期有多個 RAG 系統要測,做法都類似 `project-nas-rag`:HTTP 問答 API + 向量庫,回答附帶引用來源(文件名 + 頁碼)
- 各系統服務**不同的文件庫**,各自有自己的測試集;評測目的是各系統的品質把關與迭代回歸,不做跨系統橫向比較
- 第一個 adapter 對接 nas-rag 的 `POST /v1/query`

### 1.3 關鍵決策

- **通用核心 + adapter + 可選能力(capability)**:評測邏輯只依賴 `RAGSystem` 抽象介面;每個受測系統一個 adapter;特化診斷透過可選能力與可選欄位掛入,adapter 未支援時自動跳過並在報告註明,核心不被特化污染
- **裁判用雲端強模型**:Gemini(`google-genai` SDK,與 nas-rag 同一套;`GEMINI_API_KEY` 由 `.env` 提供)。裁判與受測系統的生成模型(gemma4)不同家,避免同源偏袒
- **規則優先、LLM 兜底**:可用確定性規則算的指標(檢索比對、數值比對、拒答判定)一律用規則——可重現、零成本;LLM 裁判只留給敘述題正確性與忠實度
- **不引入 RAGAS/DeepEval 等評測框架**:指標計算本身很簡單,且 citation 頁碼比對、單位綁定這類需求框架沒有現成支援
- **一次 `ask()` 同時取得兩層評測素材**:nas-rag 類系統回應中的 `sources` 就是 rerank 後的 top-k 檢索結果,檢索指標直接從中計算

---

## 2. 架構

### 2.1 專案結構

```
rag-evaluator/
├── src/rag_evaluator/
│   ├── adapters/
│   │   ├── base.py        # RAGSystem Protocol + RAGAnswer/SourceRef 資料模型
│   │   └── nas_rag.py     # nas-rag adapter(HTTP /v1/query,支援 top_k 覆寫)
│   ├── dataset/
│   │   ├── corpus.py      # 語料 JSONL 讀取與驗證;nas-rag manifest 轉換器
│   │   └── generator.py   # LLM 生成 QA 對、匯出審核 CSV、定稿 dataset.jsonl
│   ├── eval/
│   │   ├── retrieval.py   # Hit@k、MRR、Citation Precision、type 分桶、lost_to_cutoff
│   │   ├── numeric.py     # 數值抽取與正規化比對(千分位、全半形、千元↔元)
│   │   ├── generation.py  # Correctness、Faithfulness、Refusal 判定編排
│   │   └── runner.py      # 逐題執行(含 --runs N)、結果落盤、斷點續跑
│   ├── judge.py           # Gemini 裁判封裝(結構化輸出、重試、節流、多模態)
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
    document: str           # 文件名
    page: int               # 頁碼
    score: float | None = None    # rerank/相似度分數(可選)
    type: str | None = None       # 證據類型,如 table_figure/table_text(可選,分桶用)
    image_path: str | None = None # 頁面圖檔路徑(可選,多模態忠實度裁判用)
    content: str | None = None    # 頁面文字內容(可選,忠實度文字驗證備援)

@dataclass
class RAGAnswer:
    answer: str
    sources: list[SourceRef]
    latency_ms: int

class RAGSystem(Protocol):
    def ask(self, question: str) -> RAGAnswer: ...

class SupportsTopKOverride(Protocol):
    # 可選能力:截斷診斷需要用更大的 top_k 重打同一題。
    # adapter 未實作此介面時,lost_to_cutoff 診斷自動跳過並在報告註明。
    def ask_with_top_k(self, question: str, top_k: int) -> RAGAnswer: ...
```

### 2.3 系統設定 YAML

每個受測系統一份 YAML,由 `--system` 傳入:

```yaml
adapter: nas_rag                              # 對應 adapters/ 下的實作
endpoint: http://localhost:8020/v1/query
collection_names: [hr]
top_k: 5
timeout_s: 90
diagnostics:
  cutoff_probe_top_k: 20    # 啟用截斷診斷,省略或 null 則停用
  type_buckets: true        # 啟用 type 分桶(sources 無 type 欄位時自動不分)
```

新系統若 API 格式與 nas-rag 相同,只需新增一份 YAML;格式不同則新增一個 adapter class 並在 YAML 指定 `adapter:` 名稱。

### 2.4 nas-rag adapter

- `POST {endpoint}`,body:`{"query": <question>, "collection_names": [...], "top_k": N}`
- 回應:`{"answer": str, "sources": [{"source": str, "page": int, "rerank_score": float, "type": str, "image_path": str, "content": str, ...}]}`
- 映射:`source` → `SourceRef.document`,`page` → `SourceRef.page`,`rerank_score` → `SourceRef.score`,`type`/`image_path`/`content` 原樣帶入
- 實作 `SupportsTopKOverride`(同一 endpoint,body 改 `top_k`)

---

## 3. 黃金測試集

### 3.1 語料輸入(通用格式)

生成器不直接讀 PDF,吃標準化語料 JSONL,每行一頁:

```json
{"document": "差旅管理辦法_v3.pdf", "page": 5, "text": "頁面文字或摘要...", "collection": "hr", "image_path": "output/images/hr/差旅管理辦法_v3/page_5.png"}
```

- `image_path` 可選:忠實度裁判升級看圖時,優先以 corpus 的 `image_path` 定位頁面圖,其次用 source 回傳的 `image_path`
- `corpus from-nas-rag` 子命令從 nas-rag 的 `output/manifest_<collection>.jsonl` 轉出此格式。其他系統只要能產出此格式即可共用生成器

### 3.2 生成流程

1. **抽樣**:從語料抽 N 頁(`--sample-pages`,預設涵蓋每份文件至少一頁);`text` 過短(< 80 字)的頁面跳過
2. **LLM 生成**:每頁送 Gemini,產出 1–2 個「使用者真的會問的問題」+ 標準答案;prompt 強制答案必須能從該頁文字推出。v1 自動生成三類:
   - **單頁事實題**(tag `single-page`):主力,約 70%
   - **數值題**(tag `numeric`):該頁含表格特徵(Markdown 表格/多數字)時優先生成,約 15%;生成時同時要求輸出 `gold_value`(數字+單位)
   - **不可回答題**(tag `unanswerable`):約 15%,對整體語料生成「聽起來相關但文件中沒有答案」的問題,黃金答案標記為拒答
3. **人工審核**:輸出 `review.csv`(欄位:question、gold_answer、gold_value、answer_type、tags、evidence 文件+頁碼、來源頁文字節錄、approved)。人工在試算表中修正、補標陷阱題型 tag、或把 approved 設為否
4. **定稿**:`dataset finalize` 讀回 CSV,只保留 approved 的題目,輸出 `dataset.jsonl`

### 3.3 dataset.jsonl 格式

```json
{"id": "q001",
 "question": "2025年1月台塑廠區的營業收入是多少?",
 "gold_answer": "12,415 NTD千元",
 "gold_value": {"number": 12415, "unit": "NTD千元"},
 "answer_type": "answerable",
 "tags": ["numeric", "single-page", "unit-binding"],
 "evidence": [{"document": "營收月報.pdf", "page": 3}]}
```

- `answer_type` ∈ {`answerable`, `refusal`};refusal 題 `evidence` 為空陣列、`gold_answer` 為空、`gold_value` 為 null
- `gold_value` 可選:有值的題目正確性走規則比對,null 則走 LLM 裁判
- `tags` 為陣列,詞彙表:`single-page`、`numeric`、`unit-binding`、`company-match`、`multi-page`、`unanswerable`;報告按 tag 分項統計
- `evidence` 為陣列,支援多頁(跨頁題可人工手寫加入;LLM 自動生成 v1 不做)

### 3.4 陷阱題型的定位

`unit-binding`(同文件千元/元混用)、`company-match`(欄位沒標公司名不能亂用)、`multi-page`(跨頁整合)三類陷阱題 LLM 自動生成品質差,v1 由人工在審核階段補標既有題目或手寫新題;格式已完整支援。

### 3.5 v1 不做

- 陷阱題型與跨頁題的自動生成
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

比對鍵為 `(document, page)`;document 正規化:去路徑、去副檔名、casefold。refusal 題不計入檢索指標。

**特化診斷**(依 YAML `diagnostics:` 啟用):

| 診斷 | 定義 | 前置條件 |
|---|---|---|
| `lost_to_cutoff` | 對 evidence 未全命中的題目,用 `cutoff_probe_top_k`(預設 20)重打一次;gold 頁出現在大 top_k 回傳、但不在正常回傳的 sources 裡 → 記為截斷誤殺。報告呈現誤殺題數與比率 | adapter 實作 `SupportsTopKOverride`,否則跳過並在報告註明 |
| type 分桶 | evidence recall 按命中 source 的 `type` 欄位(如 `table_figure` vs `table_text`)分開統計,量化「reranker 對圖片頁只排到檔名」的影響 | sources 帶 `type` 欄位,否則不分桶 |

### 4.2 正確性(規則優先,LLM 兜底)

依題目屬性分流:

1. **refusal 題(規則)**:判定 answer 是否含拒答句式(「找不到相關資訊」等,句式清單可設定)
   - 不可答題正確拒答 → Refusal Accuracy 命中
   - 不可答題卻給實質答案 → `hallucinated_answer`(最嚴重的失敗模式,獨立統計)
2. **有 `gold_value` 的數值題(規則)**:從 answer 抽「數字+單位」,正規化後比對
   - 正規化涵蓋:千分位逗號、全半形數字、`千元`↔`元` 倍率換算、常見單位別名
   - 數字與單位皆對 → correct(2 分);數字對但單位錯 → `unit_mismatch`(獨立統計,計 0 分);數字錯 → 0 分
   - answer 抽不出數值:含拒答句式 → `false_refusal`(可答題卻拒答,獨立統計);否則降級送 LLM 裁判
3. **無 `gold_value` 的敘述題(LLM 裁判)**:Gemini 比對系統答案 vs 黃金答案,輸出 0/1/2(2=完全正確、1=部分正確或有遺漏、0=錯誤)
   - 判定時**不給裁判看 context**(只看問題 + 黃金答案 + 系統答案),避免被 context 帶偏
   - 可答題卻拒答同樣記 `false_refusal`

報告呈現:Correctness 平均分(滿分 2)與各分數佔比、`unit_mismatch` 率、`false_refusal` 率、`hallucinated_answer` 率。

### 4.3 忠實度(分層裁判,answerable 題)

兩段式,控制多模態成本:

1. **論斷分解**:Gemini 從系統答案抽出原子事實陳述(逐條、二元可驗證)
2. **第一段——文字驗證**:每條論斷配上「系統實際引用的 sources 頁面」的文字(以 `(document, page)` 從 corpus 查 `text`;查不到用 source 的 `content` 欄位),裁判逐條回 `supported / unsupported / insufficient`
3. **第二段——看圖升級**:`insufficient` 的論斷,若該頁有圖(corpus `image_path` 或 source `image_path`),餵頁面 PNG 給 Gemini vision 再判一次 `supported / unsupported`
4. **計分**:Faithfulness = supported 論斷數 / 總論斷數;某論斷文字與圖皆無 → 該論斷不計入,全部論斷皆無證據可查 → 該題標記 `faithfulness_skipped`

忠實度檢查只餵與該論斷相關的引用頁,不餵全部 context(避免裁判漏看)。

### 4.4 裁判設計

- 每次判定回傳結構化 JSON:`{score, reason}`;`reason` 一句話,存入結果供人工抽查
- 裁判 prompt 內含評分準則與 few-shot 範例;**judge prompt 版本號寫進每筆評分記錄**,prompt 一改歷史分數即不可比,版本號讓報告能標註
- Gemini 呼叫帶指數退避重試(3 次)與節流(可設定 QPS);單題判定失敗標記 `judge_error`,不中斷整體評測,不計入平均
- 用 `google-genai` SDK(`genai.Client`),與 nas-rag 相同;測試以 mock `genai` 模組方式驗證(沿用 nas-rag 測試模式)

---

## 5. 評測執行與報告

### 5.1 執行(runner)

- `--runs N`(預設 1):每題執行 N 次,處理受測系統 temperature 1.0 的回答非決定性
- 逐題逐 run:呼叫 `ask()` → 計算檢索指標(含診斷)→ 正確性判定 → 忠實度判定 → append 寫入 `runs/<run-id>/results.jsonl`
- 每行 results.jsonl 含:題目 id、**run 編號**(n=1 時為 0)、系統原始回答與 sources、各指標值、裁判理由、judge prompt 版本、延遲、錯誤標記
- **斷點續跑**:重跑同一 `--run-id` 時,已存在於 results.jsonl 的 (題目, run) 組合跳過
- 受測系統呼叫失敗(timeout/5xx):重試 2 次,仍失敗標記 `system_error`,繼續下一題
- 截斷診斷的 probe 呼叫每題最多一次(不隨 runs 倍增),掛在該題第一個 run

### 5.2 報告(report.md)

- **總覽表**:Hit@1/3/5、MRR、Citation Precision、Correctness 平均、`unit_mismatch` 率、`false_refusal` 率、`hallucinated_answer` 率、Faithfulness 平均、Refusal Accuracy、平均延遲、`system_error`/`judge_error` 題數
- **特化診斷區**(啟用時):`lost_to_cutoff` 誤殺率、type 分桶 recall 表;未啟用或 adapter 不支援時註明
- **多次執行區**(N>1 時):pass@1(平均分)、pass@N(任一 run 正確即算對)、答案一致率(數值題比正規化後數值是否全同,敘述題比裁判分數是否全同)
- **分項統計**:按 `tags` 拆分各指標
- **最差案例清單**:Correctness=0 或檢索全 miss 的題目,列出問題、系統答案、黃金答案、裁判理由、引用頁面(含 image_path)
- **baseline 比較**:`--baseline <run-id>` 時每個指標附差值(+/-)

### 5.3 CLI

```bash
rag-eval corpus from-nas-rag --manifest <path> -o corpus.jsonl
rag-eval dataset generate --corpus corpus.jsonl -o review.csv [--sample-pages N]
rag-eval dataset finalize --review review.csv -o dataset.jsonl
rag-eval run --system nas-rag.yaml --dataset dataset.jsonl --corpus corpus.jsonl \
             --run-id <id> [--runs N] [--baseline <run-id>]
# --corpus 供 Faithfulness 查引用頁面文字/圖檔;省略時 Faithfulness 退回 source 的
# content/image_path 欄位,兩者皆無則標記 faithfulness_skipped
```

---

## 6. 技術棧

| 項目 | 選型 | 說明 |
|---|---|---|
| 語言 | Python 3.11+ | 與 nas-rag 一致 |
| HTTP client | httpx | 呼叫受測系統 |
| LLM | google-genai(Gemini) | 測試題生成 + 裁判(文字與 vision);`GEMINI_API_KEY` 由 `.env` 提供 |
| 資料模型 | pydantic v2 | dataset/corpus/config 驗證 |
| CLI | argparse 子命令 | 無額外依賴 |
| 測試 | pytest | LLM 呼叫以 mock 測(沿用 nas-rag 的 mock genai 模式);數值正規化、檢索比對以純函式單元測試 |

模型名稱、QPS、重試次數、抽樣參數、拒答句式清單等可調值集中於 `.env` 與系統 YAML,不散落在程式碼中。

---

## 7. 錯誤處理

| 場景 | 處理 |
|---|---|
| 受測系統 timeout / 5xx | 重試 2 次後標記 `system_error`,繼續下一題 |
| 裁判 API 失敗 | 重試 3 次後標記 `judge_error`,不計入平均,報告列出題數 |
| dataset/corpus 格式錯誤 | 啟動時 pydantic 全檔驗證,先報錯不開跑 |
| 忠實度無文字也無圖 | 標記 `faithfulness_skipped`,不計入平均 |
| 截斷診斷但 adapter 不支援 top_k 覆寫 | 跳過診斷,報告註明 |
| 圖檔路徑不存在 | 該論斷維持 insufficient,計為無證據可查 |
| run-id 已存在 | 續跑(跳過已完成 (題目, run));如需重跑請換 run-id 或刪除舊目錄 |

---

## 8. 初版範圍外(未來可擴展)

- rerank 前召回評測(需直連向量庫)
- 陷阱題型/跨頁整合題自動生成
- 多輪對話評測
- HTML 報告 / 圖表
- 跨系統橫向比較報告
- `temperature=0` 對照組自動化
