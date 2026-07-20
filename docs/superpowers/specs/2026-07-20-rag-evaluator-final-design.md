# RAG 問答系統品質評測工具 — 最終版設計文件

**日期**:2026-07-20(rev 2,採納外部 review 修訂)
**狀態**:Approved
**專案**:rag-evaluator
**取代**:本文件合併並取代 `2026-07-16-rag-evaluator-design.md`(通用架構)與根目錄 `RAG-EVALUATION-DESIGN.md`(nas-rag 特化分析)兩份草稿。

**rev 2 修訂摘要**:截斷診斷改名次歸因;corpus 文字來源鏈與 vision 出題備援;run manifest 與續跑一致性檢查;分母與兩軌報告定義;陷阱 tag 強制裁判;evidence_recall 與 gold-type 分桶;比對鍵加 collection;忠實度無引用計 0;拒答四態分類;CLI 拆 collect/score/report;baseline 配對 bootstrap。

---

## 1. 概述

### 1.1 目標

建立一個「**通用核心 + 每受測專案特化層**」的 RAG 問答品質評測 CLI 工具,涵蓋:

1. **黃金測試集生成**:從受測系統的文件語料自動生成 QA 對,經人工審核後定稿
2. **檢索層評測**:Hit@k、MRR、Evidence Recall、Citation Precision(純程式比對,零 LLM 成本)+ 特化診斷(截斷誤殺、type 分桶)
3. **生成層評測**:Correctness(規則優先、LLM 兜底)、Faithfulness(分層多模態裁判)、Refusal Accuracy
4. **報告輸出**:Markdown 總結報告 + 逐題 JSONL 原始結果,支援 baseline 配對回歸比較

### 1.2 受測系統假設

- 近期有多個 RAG 系統要測,做法都類似 `project-nas-rag`:HTTP 問答 API + 向量庫,回答附帶引用來源(文件名 + 頁碼)
- 各系統服務**不同的文件庫**,各自有自己的測試集;評測目的是各系統的品質把關與迭代回歸,不做跨系統橫向比較
- 第一個 adapter 對接 nas-rag 的 `POST /v1/query`
- **契約驗證**:`qa_api.py` 僅存在於 nas-rag 的 `docs/archive/`,不可只依 archived 程式碼假定 schema。實作 adapter 前先對真實服務錄製一筆 response 存為 fixture,作為 contract test 依據

### 1.3 關鍵決策

- **通用核心 + adapter + 可選能力(capability)**:評測邏輯只依賴 `RAGSystem` 抽象介面;每個受測系統一個 adapter;特化診斷透過可選能力與可選欄位掛入,adapter 未支援時自動跳過並在報告註明,核心不被特化污染
- **裁判用雲端強模型**:Gemini(`google-genai` SDK,與 nas-rag 同一套;`GEMINI_API_KEY` 由 `.env` 提供)。裁判與受測系統的生成模型(gemma4)不同家,避免同源偏袒
- **規則優先、LLM 兜底**:可用確定性規則算的指標(檢索比對、數值比對、拒答判定)一律用規則——可重現、零成本;LLM 裁判留給敘述題、陷阱題與忠實度
- **collect 與 score 解耦**:問答收集(打受測系統)與評分(規則 + 裁判)是獨立階段,原始回答落盤後可不重打受測系統、用新 judge prompt 重評
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
│   │   └── generator.py   # LLM 生成 QA 對(文字/vision)、匯出審核 CSV、定稿 dataset.jsonl
│   ├── eval/
│   │   ├── retrieval.py   # Hit@k、MRR、Evidence Recall、Citation Precision、type 分桶、lost_to_cutoff
│   │   ├── numeric.py     # 數值抽取與正規化比對(Decimal、千分位、全半形、千元↔元)
│   │   ├── refusal.py     # 拒答四態分類(pure/substantive/mixed/empty)
│   │   ├── generation.py  # Correctness、Faithfulness、Refusal 判定編排
│   │   ├── collector.py   # 問答收集(逐題呼叫 ask(),原始結果落盤、斷點續跑)
│   │   └── scorer.py      # 評分(讀原始結果,規則 + 裁判,scores 落盤、斷點續跑)
│   ├── judge.py           # Gemini 裁判封裝(結構化輸出、重試、節流、多模態)
│   ├── report.py          # report.md 產生、baseline 配對比較(bootstrap CI)
│   ├── run_manifest.py    # run_manifest.json 建立與續跑一致性驗證
│   ├── config.py          # 系統 YAML 設定載入(pydantic)
│   └── cli.py             # CLI 入口(argparse 子命令)
├── tests/                 # 每個模組一個 test_<module>.py;fixtures/ 含真實 API response 錄製
├── runs/                  # 評測輸出(gitignore)
└── pyproject.toml
```

### 2.2 RAGSystem 抽象介面

```python
@dataclass
class SourceRef:
    document: str           # 文件名(nas-rag 為絕對路徑,比對前正規化)
    page: int               # 頁碼(1-based)
    collection: str | None = None   # 所屬 collection(可選,比對鍵用)
    score: float | None = None      # rerank/相似度分數(可選)
    type: str | None = None         # 證據類型,如 table_figure/table_text(可選)
    image_path: str | None = None   # 頁面圖檔路徑(可選,多模態裁判用)
    content: str | None = None      # 頁面文字內容(可選,忠實度文字驗證備援)
    schema_text: str | None = None  # 頁面 LLM 摘要(可選,忠實度文字驗證備援)
    file_hash: str | None = None    # 文件雜湊(可選,比對鍵優先使用)

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
  type_buckets: true        # 啟用 type 分桶(gold 頁無 type 資訊時自動不分)
```

新系統若 API 格式與 nas-rag 相同,只需新增一份 YAML;格式不同則新增一個 adapter class 並在 YAML 指定 `adapter:` 名稱。

### 2.4 nas-rag adapter

- `POST {endpoint}`,body:`{"query": <question>, "collection_names": [...], "top_k": N}`
- 回應:`{"answer": str, "sources": [<Qdrant payload copy + rerank_score>]}`;payload 含 `collection`、`source`(絕對路徑)、`file_hash`、`page`、`type`、`image_path`,table flow 另有 `schema_text`/`content`
- 映射:`source` → `SourceRef.document`,其餘同名欄位原樣帶入,`rerank_score` → `SourceRef.score`
- 實作 `SupportsTopKOverride`(同一 endpoint,body 改 `top_k`)
- **schema 以真實服務錄製的 fixture 為準**(見 1.2),archived 程式碼僅供參考

---

## 3. 黃金測試集

### 3.1 語料輸入(通用格式)

生成器不直接讀 PDF,吃標準化語料 JSONL,每行一頁:

```json
{"collection": "hr", "document": "差旅管理辦法_v3.pdf", "page": 5,
 "text": "頁面文字或摘要...", "text_source": "content",
 "type": "table_text", "file_hash": "sha256:...",
 "image_path": "output/images/hr/差旅管理辦法_v3/page_5.png"}
```

- **`text` 來源鏈**:`content`(原文)> `schema_text`(LLM 摘要)> 無;實際來源記錄於 `text_source` ∈ {`content`, `schema_text`, `none`}。nas-rag 的 image flow 頁**沒有任何文字欄位**,`text_source: none` 是常態而非例外
- `type` 由 manifest 的 `flow` 對映(`image`→`table_figure` 等),供 gold-type 分桶
- `image_path`、`file_hash` 可選;忠實度裁判與比對鍵有值就用
- `corpus from-nas-rag` 子命令從 nas-rag 的 `output/manifest_<collection>.jsonl` 轉出此格式
- **載入時驗證**:文件名正規化後(見 4.1)偵測 `(collection, document, page)` 碰撞,發現即報錯

### 3.2 生成流程

1. **抽樣**:從語料抽 N 頁(`--sample-pages`,預設涵蓋每份文件至少一頁)
2. **LLM 生成**:每頁送 Gemini,產出 1–2 個「使用者真的會問的問題」+ 標準答案。**依頁面文字狀況分流**:
   - `text` 足夠(≥ 80 字)→ 文字 prompt,強制答案必須能從該頁文字推出
   - `text` 不足但有 `image_path` → 改送頁面 PNG 給 Gemini vision 出題(nas-rag 的多數頁面走此路;若跳過它們,測試集會系統性避開圖片頁——正是最需要測的部分)
   - 兩者皆無 → 跳過並記錄
   v1 自動生成三類:
   - **單頁事實題**(tag `single-page`):主力,約 70%
   - **數值題**(tag `numeric`):該頁含表格特徵時優先生成,約 15%;生成時同時要求輸出 `gold_value`(數字+單位)
   - **不可回答題**(tag `unanswerable`):約 15%,生成「聽起來相關但抽樣頁面中沒有答案」的問題
3. **人工審核**:輸出 `review.csv`(欄位:question、gold_answer、gold_value、answer_type、tags、evidence 文件+頁碼、來源頁文字節錄或圖檔路徑、generation_basis、approved)
   - **unanswerable 題 `approved` 預設為否,必須人工主動確認**——生成器只看過抽樣頁,無法證明整個語料庫都沒有答案,錯誤的 negative label 會冤枉受測系統;`generation_basis` 記錄生成依據頁供稽核
   - 人工在此階段修正答案、補標陷阱題型 tag(`unit-binding`、`company-match`、`multi-page`)、或手寫新題
4. **定稿**:`dataset finalize` 讀回 CSV,只保留 approved 的題目,輸出 `dataset.jsonl`
   - **qid 由 question 內容雜湊衍生**(如 `q-<sha256 前 8 碼>`),不依 CSV 列順序;finalize 時偵測 qid 重複即報錯。qid 穩定才能跨 run 配對比較

### 3.3 dataset.jsonl 格式

```json
{"id": "q-3fa9c2d1",
 "question": "2025年1月台塑廠區的營業收入是多少?",
 "gold_answer": "12,415 NTD千元",
 "gold_value": {"number": 12415, "unit": "NTD千元"},
 "answer_type": "answerable",
 "tags": ["numeric", "single-page", "unit-binding"],
 "evidence": [{"collection": "hr", "document": "營收月報.pdf", "page": 3}]}
```

- `answer_type` ∈ {`answerable`, `refusal`};refusal 題 `evidence` 為空陣列、`gold_answer` 為空、`gold_value` 為 null
- `gold_value` 可選:有值且**不帶陷阱 tag** 的題目正確性走規則比對;帶陷阱 tag 或無 `gold_value` 則走 LLM 裁判(見 4.2)
- `tags` 為陣列,詞彙表:`single-page`、`numeric`、`unit-binding`、`company-match`、`multi-page`、`unanswerable`;報告按 tag 分項統計
- `evidence` 為陣列,支援多頁;`collection` 可選(省略時比對不限 collection)
- **v1 語意**:多頁 evidence 視為「皆為必要」(AND),以 Evidence Recall 與 all-evidence-hit 衡量;替代證據組(OR)留待 v2 以 `evidence_groups` 向下相容擴充

### 3.4 陷阱題型的定位

`unit-binding`(同文件千元/元混用)、`company-match`(欄位沒標公司名不能亂用)、`multi-page`(跨頁整合)三類陷阱題 LLM 自動生成品質差,v1 由人工在審核階段補標既有題目或手寫新題;格式已完整支援。

### 3.5 v1 不做

- 陷阱題型與跨頁題的自動生成
- `evidence_groups`(替代證據 OR 語意)
- 多輪對話題(受測系統尚無多輪能力)

---

## 4. 評測指標

### 4.1 檢索層(純程式計算)

**比對鍵**:`(collection, document, page)`。document 正規化:去路徑(同時處理 `\` 與 `/`)、去副檔名、Unicode NFC、casefold;頁碼一律 1-based;evidence 或 source 缺 `collection` 時該欄位不參與比對;雙方都有 `file_hash` 時優先用 `(file_hash, page)`。refusal 題不計入檢索指標。

| 指標 | 定義 |
|---|---|
| Hit@k(k=1,3,5) | 任一 evidence 頁出現在前 k 個 sources 即為命中 |
| Evidence Recall | 該題 evidence 頁被 sources 涵蓋的比例;全部題目取平均 |
| All-evidence Hit | evidence 頁全部出現在 sources 中的題目比例(multi-page 題的關鍵指標) |
| MRR | 第一個命中 evidence 的 source 排名倒數,全部題目取平均;未命中計 0 |
| Citation Precision | sources 中屬於 evidence 的比例,全部題目取平均 |

**特化診斷**(依 YAML `diagnostics:` 啟用):

| 診斷 | 定義 | 前置條件 |
|---|---|---|
| 截斷/排名歸因 | 對 evidence 未全命中的題目,用 `cutoff_probe_top_k`(預設 20)重打一次。gold 頁出現在 probe 回傳中時,依其 **probe 名次**歸因:名次 ≤ 正常 top_k → `lost_to_cutoff`(本應入選卻被最大斷層截掉);名次 > top_k → `ranked_below_top_k`(排名太低,與截斷無關)。probe 回應自身也經過截斷,故 `lost_to_cutoff` 為保守低估,報告註明 | adapter 實作 `SupportsTopKOverride`,否則跳過並在報告註明 |
| type 分桶 | Evidence Recall 按 **gold 頁的 type**(由 corpus 查得,如 `table_figure` vs `table_text`)分開統計——按 gold 分桶才統計得到完全 miss 的圖片頁,量化「reranker 對圖片頁只排到檔名」的影響 | corpus 提供 gold 頁 type,否則不分桶 |

### 4.2 正確性(規則優先,LLM 兜底)

依題目屬性分流:

1. **refusal 題**:先做拒答四態分類(見 4.3)
   - `pure_refusal` → Refusal Accuracy 命中
   - `substantive_answer` 或 `mixed_refusal_answer` → `hallucinated_answer`(最嚴重的失敗模式)
2. **有 `gold_value` 且不帶陷阱 tag 的數值題(規則)**:從 answer 抽「數字+單位」,正規化後比對
   - 正規化:Decimal 運算、千分位逗號、全半形數字、`千元`↔`元` 倍率換算、常見單位別名、負數與百分比;容許誤差預設 0(可設定)
   - 多數字候選:answer 中所有候選數值正規化後,**恰有一個**與 gold 相符且無互相矛盾的候選 → 依單位判定;出現互相矛盾的多個數值 → 降級送 LLM 裁判
   - 數字與單位皆對 → 2 分;數字對但單位錯 → `unit_mismatch`(獨立統計,計 0 分);數字錯 → 0 分
   - answer 抽不出數值:四態分類為 `pure_refusal` → `false_refusal`;否則降級送 LLM 裁判
   - 規則結果同時記錄為獨立訊號 `numeric_match`,與 correctness 分開呈現
3. **帶陷阱 tag(`company-match`、`multi-page`)或無 `gold_value` 的題目(LLM 裁判)**:Gemini 比對系統答案 vs 黃金答案,輸出 0/1/2(2=完全正確、1=部分正確或有遺漏、0=錯誤)
   - 陷阱題即使數值對,主體/期間/廠區指錯即非 2 分——這正是規則比對做不到、必須語意裁判的部分
   - 判定時**不給裁判看 context**(只看問題 + 黃金答案 + 系統答案),避免被 context 帶偏
   - 可答題四態分類為 `pure_refusal` → `false_refusal`,不送裁判

### 4.3 拒答判定(規則,四態互斥)

answer 分類為互斥四態:

- `pure_refusal`:含拒答句式(句式清單可設定)且無實質內容
- `substantive_answer`:無拒答句式、有實質內容
- `mixed_refusal_answer`:含拒答句式**且**含實質內容(如「找不到原始資料,但依推測答案是 12,415 千元」)——不可因包含拒答片語就算拒答成功
- `empty_non_answer`:空白或無意義回覆

「實質內容」判定:抽得出數值、或去除拒答句式後剩餘文字超過門檻(可設定)。

### 4.4 忠實度(分層裁判,answerable 題)

流程明定為四步:

1. **論斷分解**:Gemini 從系統答案抽出原子事實陳述(逐條、二元可驗證),同時要求為每條論斷標注其對應的 source 索引(結構化輸出;對齊不到任何 source 的論斷標 `unaligned`)
2. **第一段——文字驗證**:每條論斷配上其對齊 sources 的文字(以比對鍵從 corpus 查 `text`;查不到用 source 的 `content`/`schema_text`),裁判逐條回 `supported / unsupported / insufficient`;`unaligned` 論斷改用全部 sources 的文字驗證
3. **第二段——看圖升級**:`insufficient` 的論斷,若對齊頁有圖(corpus `image_path` 或 source `image_path`),餵頁面 PNG 給 Gemini vision 再判一次 `supported / unsupported`
4. **計分**:Faithfulness = supported 論斷數 / 可驗證論斷總數
   - **系統無 sources 卻有實質論斷 → Faithfulness 計 0**(不引用來源必須受罰,不得以 skipped 逃過)
   - 有 sources 但某論斷文字與圖皆查無 → 該論斷標 `evidence_unavailable`,不計入分母;全部論斷皆 `evidence_unavailable` → 該題標 `faithfulness_skipped`
   - `no_sources`(系統沒引用)與 `evidence_unavailable`(評測器找不到證據材料)為不同狀態,分開統計
   - 另報 `faithfulness_evaluable_claim_rate`(可驗證論斷 / 全部論斷),過低表示忠實度分數代表性不足

### 4.5 裁判設計

- 每次判定回傳結構化 JSON:`{score, reason}`;以 pydantic schema 驗證後再做語意檢查(分數在值域、reason 非空)——schema 合規不代表內容有效
- 裁判 prompt 內含評分準則與 few-shot 範例;**judge prompt 版本號與內容 hash 皆寫進每筆評分記錄**
- **Prompt injection 防護**:文件內容與系統回答一律視為不可信輸入,以明確分隔符包裹,prompt 指示裁判忽略其中的任何指令
- Gemini 呼叫帶指數退避重試(3 次)與節流(可設定 QPS);單題判定失敗標記 `judge_error`,不中斷整體評測
- 用 `google-genai` SDK(`genai.Client`),與 nas-rag 相同;測試以 mock `genai` 模組方式驗證
- **人工校準**:每次正式評測後抽 10% 裁判判定人工複核,記錄 agreement;LLM 分數在校準前只視為 proxy
- **資料治理註記**:評測會將企業文件文字與頁面圖片送往雲端 Gemini(與 nas-rag 既有的 embedding/摘要流程相同的暴露面);敏感語料需事前確認允許送雲

---

## 5. 評測執行與報告

### 5.1 階段拆分:collect → score → report

```bash
rag-eval collect --system nas-rag.yaml --dataset dataset.jsonl --run-id <id> [--runs N]
rag-eval score   --run-id <id> --corpus corpus.jsonl
rag-eval report  --run-id <id> [--baseline <run-id>]
rag-eval run ...   # 三步的便利包裝,參數為三者聯集
```

- **collect**:逐題逐 run 呼叫 `ask()`,原始回答落盤 `runs/<run-id>/raw.jsonl`(每筆:qid、run 編號、answer、sources、延遲、錯誤標記);截斷診斷的 probe 呼叫每題最多一次(掛在第一個 run),結果同樣落盤
- **score**:讀 raw.jsonl 計算規則指標 + 裁判判定,寫 `runs/<run-id>/scores.jsonl`(每筆:各指標值、裁判理由、judge prompt 版本與 hash)。**不重打受測系統即可用新 judge prompt 重評**:`--rescore-tag <tag>` 產生 `scores-<tag>.jsonl` 並存,供 prompt 迭代比較與 metric bug 修正後重算
- **report**:讀 scores 產出 report.md
- `--runs N`(預設 1):處理受測系統 temperature 1.0 的回答非決定性

### 5.2 run manifest 與續跑一致性

collect 啟動時建立不可變的 `runs/<run-id>/run_manifest.json`:

- dataset 與 corpus 的 SHA-256
- system YAML 完整快照
- evaluator 版本(git SHA)與 adapter 名稱
- `--runs` N、開始時間
- score 階段補記:judge model 名稱、judge prompt 版本與 hash、拒答句式清單 hash、數值正規化規則版本

**續跑**(同 run-id 重跑):逐項驗證現有 manifest 與本次參數,不一致即拒絕執行(明確報錯要求換 run-id);一致則跳過 raw.jsonl / scores.jsonl 中已存在的 (qid, run) 組合。baseline 比較時同樣驗證兩個 run 的 dataset hash 一致,否則拒絕。

### 5.3 錯誤與分母定義

| 標記 | 條件 | 計分處理 |
|---|---|---|
| `system_error` | 受測系統重試 2 次仍失敗 | **端到端指標計 0 分**;另報「有效樣本平均」(排除後),兩軌並陳 |
| `judge_error` | 裁判重試 3 次仍失敗 | 排除於平均,但必報 `judge_coverage`(成功判定題數/應判定題數) |
| `faithfulness_skipped` | 全部論斷 `evidence_unavailable` | 排除於 Faithfulness 平均,報 skipped 題數 |

**各率的分母明定**:

- `unit_mismatch` 率:分母 = 走規則比對的數值題
- `false_refusal` 率:分母 = answerable 題
- `hallucinated_answer` 率:分母 = refusal 題
- Refusal Accuracy:分母 = refusal 題

### 5.4 報告(report.md)

- **總覽表(兩軌)**:每個品質指標同時呈現「端到端」(system_error 計 0)與「有效樣本」(排除 error)兩欄;含 Hit@1/3/5、Evidence Recall、All-evidence Hit、MRR、Citation Precision、mean_correctness、`numeric_match` 率、`unit_mismatch` 率、`false_refusal` 率、`hallucinated_answer` 率、Faithfulness、Refusal Accuracy、平均延遲、`system_error`/`judge_error` 題數、`judge_coverage`、`faithfulness_evaluable_claim_rate`
- **特化診斷區**(啟用時):`lost_to_cutoff` 率、`ranked_below_top_k` 率、gold-type 分桶 recall 表;未啟用或 adapter 不支援時註明
- **多次執行區**(N>1 時):`mean_correctness`(逐 run 平均)、`success@N`(任一 run 得 2 分的題目比例)、`agreement_rate`(數值題:正規化後數值各 run 全同的比例;敘述題:裁判分數全同的比例)
- **分項統計**:按 `tags` 拆分各指標
- **最差案例清單**:correctness=0 或檢索全 miss 的題目,列出問題、系統答案、黃金答案、裁判理由、引用頁面(含 image_path)
- **baseline 比較**:按相同 qid **配對比較**,每個指標附差值與 **bootstrap 95% 信賴區間**(對配對差重抽);CI 不含 0 才標記顯著,避免小測試集把隨機波動當回歸

---

## 6. 技術棧

| 項目 | 選型 | 說明 |
|---|---|---|
| 語言 | Python 3.11+ | 與 nas-rag 一致 |
| HTTP client | httpx | 呼叫受測系統 |
| LLM | google-genai(Gemini) | 測試題生成(文字/vision)+ 裁判(文字/vision);`GEMINI_API_KEY` 由 `.env` 提供 |
| 資料模型 | pydantic v2 | dataset/corpus/config/裁判輸出驗證 |
| CLI | argparse 子命令 | 無額外依賴 |
| 測試 | pytest | LLM 呼叫以 mock 測(沿用 nas-rag 的 mock genai 模式);數值正規化、檢索比對、拒答分類以純函式單元測試;adapter 以真實錄製 fixture 做 contract test |
| 依賴管理 | pyproject 版本上界 + lockfile | 重現性;Gemini 呼叫的重試/節流政策由 evaluator 自行定義,不依賴 SDK 預設 |

模型名稱、QPS、重試次數、抽樣參數、拒答句式清單、數值容許誤差等可調值集中於 `.env` 與系統 YAML,不散落在程式碼中。

---

## 7. 錯誤處理

| 場景 | 處理 |
|---|---|
| 受測系統 timeout / 5xx | 重試 2 次後標記 `system_error`,繼續下一題;計分見 5.3 |
| 裁判 API 失敗 | 重試 3 次後標記 `judge_error`,繼續;報 `judge_coverage` |
| dataset/corpus 格式錯誤 | 啟動時 pydantic 全檔驗證,先報錯不開跑 |
| corpus 文件名正規化後碰撞 | 載入時報錯 |
| 系統無 sources 卻有實質回答 | Faithfulness 計 0(`no_sources` 狀態) |
| 忠實度證據材料查無(文字與圖皆無) | 論斷標 `evidence_unavailable` 不計分母;全查無 → `faithfulness_skipped` |
| 截斷診斷但 adapter 不支援 top_k 覆寫 | 跳過診斷,報告註明 |
| 續跑參數與 run_manifest 不一致 | 拒絕執行,要求換 run-id |
| baseline 與本次 dataset hash 不一致 | 拒絕比較 |
| run-id 已存在且 manifest 一致 | 續跑(跳過已完成 (qid, run)) |

---

## 8. 初版範圍外(未來可擴展)

- rerank 前召回評測(需直連向量庫)
- 陷阱題型/跨頁整合題自動生成
- `evidence_groups`(替代證據 OR 語意)
- unanswerable 題的全語料檢索驗證
- 多輪對話評測
- HTML 報告 / 圖表
- 跨系統橫向比較報告
- `temperature=0` 對照組自動化
