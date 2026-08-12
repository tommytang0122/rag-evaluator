# RAG 回答品質評分系統設計

針對 `project-nas-rag`（中文企業財報 PDF → Qdrant 多模態 RAG）設計的回答品質評分系統。

---

## 0. nas-rag 的實際形態（與評分設計相關的事實）

設計前先確認被評系統的關鍵特性，這些直接決定了評分方法的選擇：

1. **領域是中文企業財報 PDF 問答**：答案高度數值化（金額 + 單位，如「NTD千元」）。後端 prompt 明文要求「單位綁定」，並要求找不到時回「找不到相關資訊」。
2. **檢索單位是「PDF 的某一頁」**：每個 Qdrant point 對應一頁，`sources` 回傳 `{source, page, type, rerank_score, ...}`。**檢索指標可完全用程式精確計算，不需要 LLM**，只要黃金集標到「頁」層級。
3. **上下文是多模態的**：`table_figure` 類型的證據是整頁 PNG（餵給 vision LLM），`table_text` 才是文字。**忠實度 judge 必須是多模態模型**（範例 collection 53 頁中有 47 頁是 image）。
4. **回答端用 gemma4:31b、temperature 1.0**：回答非決定性，同題重問答案會變。評測必須處理這件事。
5. **評測介面**：`POST /v1/query`（port 8020）→ `{answer, sources[]}`，單一 HTTP call，執行器很好寫。
6. **兩個已知的系統性弱點**（評分系統很可能量化到）：
   - reranker 對 `table_figure` 拿到的「文件內容」其實只是 `image_path` 字串（`qa_api.py:138`），等於在排檔名。
   - 「最大斷層法」截斷（`qa_api.py:163-187`）可能把需要的頁切掉。

### 一次 RAG 問答的四個可觀察元素

| 符號 | 內容 |
|---|---|
| Q | 使用者問題 |
| C | 檢索到的上下文（sources：頁面 PNG 或 table_text 內容） |
| A | 系統回答 answer |
| G | 標準答案（黃金集提供，含 gold_evidence 頁碼） |

品質問題永遠出在這四者之間的關係，評分維度沿這些「邊」定義。

---

## 1. 黃金集格式（每題一筆 JSONL）

```json
{
  "qid": "fin-0042",
  "collection_names": ["gavin_test"],
  "question": "2025年1月台塑廠區的營業收入是多少？",
  "answerable": true,
  "gold_answer": "12,415 NTD千元",
  "gold_value": {"number": 12415, "unit": "NTD千元"},
  "gold_evidence": [{"source": "營收月報.pdf", "page": 3}],
  "tags": ["numeric", "single-page", "unit-binding"]
}
```

題型刻意覆蓋五類，對應此系統的已知風險點：

| 題型 tag | 目的 |
|---|---|
| `single-page` 單頁數值題 | 主力，量測基本檢索+讀數能力 |
| `unit-binding` 單位陷阱題 | 同文件內千元/元混用的頁，測 prompt 的單位綁定守則 |
| `company-match` 公司/廠區對應題 | 欄位沒標公司名時不能亂用（prompt 指南第 2 條防守目標） |
| `multi-page` 跨頁整合題 | 測需要多頁的問題 |
| `unanswerable` 不可答題 | `answerable: false`，正解是「找不到相關資訊」 |

> 出題來源：從 `raw/` 下的 PDF 人工出題，標到頁。這是整套系統唯一需要人力的部分。

---

## 2. 指標（按成本分層）

### A. 檢索指標（純程式計算，零 LLM 成本）

拿 `sources` 的 `(source, page)` 對比 `gold_evidence`：

- **`evidence_recall@k`**：gold 頁是否出現在回傳的 sources 裡（最重要的單一檢索指標）
- **`evidence_precision` / `MRR`**：gold 頁排多前面
- **截斷診斷 `lost_to_cutoff`**：同題再用 `top_k=20` 打一次，若 gold 頁在大 top_k 有、小 top_k 沒有，記為誤殺——直接量化「最大斷層法」的誤殺率
- **按 `type` 分桶**（`table_figure` vs `table_text`）分別報 recall，驗證「reranker 只看到 image_path」是否真的拖累圖片頁排名

### B. 答案正確性（規則優先，LLM 兜底）

- **數值題**：從 answer 抽「數字 + 單位」，正規化後比對（千分位、全半形、`千元`↔`元` 換算）。數字對但單位錯 → `unit_mismatch`（prompt 明文防守的失敗模式，獨立統計）
- **敘述題**：LLM judge 對比 gold_answer，輸出 `correct / partially_correct / wrong` 三檔
- **不可答題**：規則判斷 answer 是否含「找不到相關資訊」/「找不到任何相關的資料」
  - 可答題卻回拒答句 → `false_refusal`
  - 不可答題卻給實質答案 → `hallucinated_answer`（最嚴重）

### C. 忠實度（多模態 LLM judge）

- 把 answer 拆成**原子論斷**，逐條驗證「能否從證據推出」，忠實度 = 被支撐論斷 / 總論斷數
- 證據餵法跟著 `type` 走：
  - `table_text` → 餵 `content`
  - `table_figure` → 餵 `output/images/<collection>/<pdf-stem>/page_<N>.png` 原圖（由 sources 的 `image_path` 定位）
- judge 用**多模態模型**：專案已有 `GEMINI_API_KEY`，用 Gemini 當 judge，與被評的 gemma4 不同家，避免同源偏袒
- **便宜的預檢**：先用 `schema_text`（每頁的 LLM 摘要，已在 payload 裡）做文字層驗證，只有判不出來的論斷才升級到看原圖

### 為什麼大部分評分可用確定性規則

此系統的證據可定位到「頁」、答案可正規化成「數字」，所以 A/B 兩層幾乎全是規則計算。LLM judge 只留給忠實度（C）和敘述題——更便宜、更可信、可重複。

---

## 3. 處理 temperature 1.0 的非決定性

每題跑 **n=3 次**，報告三個層次：

- **`pass@1`** 平均分（使用者體感）
- **`pass@3`**（能力上限）
- **答案一致率**：三次抽出的數值是否相同——財報問答同題不同答是嚴重的信任問題，一致率低本身就是重要發現

建議另附一組 `temperature=0` 對照組，量化多少錯誤純粹來自採樣隨機性。

---

## 4. 執行器

```
for q in golden_set:
    for i in range(3):
        t0 = now()
        resp = POST http://<host>:8020/v1/query
               {"query": q.question,
                "collection_names": q.collection_names,
                "top_k": 5}
        record(qid, run=i, answer=resp.answer, sources=resp.sources, latency=now()-t0)
    # 另打一次 top_k=20 供截斷診斷
```

- **執行器與評分引擎解耦**：待測系統只需符合 `POST /v1/query → {answer, sources}` 介面，同一套評分器可評 nas-rag 或其任何版本
- **所有原始記錄落盤成 JSONL**（每筆：Q、C、A、latency），評分階段離線跑、可重跑、可換 judge prompt 版本重評
- **順帶記錄非品質指標**：端到端延遲、檢索耗時——品質提升若以 3 倍延遲換來，需要被看見

---

## 5. 報告與彙總

- **不要只出一個總分**，單一總分掩蓋歸因資訊
- 按 **collection × 題型 tag** 交叉報表，每格報：
  `evidence_recall`、`correctness`、`unit_mismatch率`、`false_refusal率`、`hallucinated_answer率`、`faithfulness`、`答案一致率`、`延遲 P50/P95`
- **失敗案例存檔**：保留 (問題, sources, answer, judge 理由, 對應頁面 PNG 路徑) 完整四元組，這是 debug 的原材料
- **回歸門檻而非絕對門檻**：CI 中比較「本次 vs 上一版基準」，任一維度顯著下降（bootstrap 置信區間不重疊）就標紅，避免 judge 抖動誤報

---

## 6. 系統架構

```
┌──────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────┐
│ 黃金集     │──▶│ RAG 執行器    │──▶│ 評分引擎        │──▶│ 報告產生器 │
│ (JSONL)  │   │ POST /v1/query│   │ A 檢索(規則)   │   │ 維度分數  │
│ Q, G,    │   │ 記錄 A/C/延遲  │   │ B 正確性(規則) │   │ 分佈/迴歸 │
│ evidence │   │ n=3 + topk=20 │   │ C 忠實度(多模態)│   │ 失敗案例  │
└──────────┘   └──────────────┘   └───────────────┘   └──────────┘
```

---

## 7. 落地順序

1. **建 30–50 題黃金集**（人工出題，標到頁）— 唯一需人力的部分
2. **執行器 + A 層檢索指標**（一天內可完成，零 LLM 成本，立即回答「檢索端好不好」）
3. **B 層數值比對**（財報場景下這一項就覆蓋大半正確性判斷）
4. **C 層多模態忠實度 judge**，並人工抽 10% 校準

---

## 8. 已知陷阱

- **LLM judge 偏好長回答與自家文風** → 用論斷分解 + 二元判斷取代整體印象分
- **合成測試題偏簡單**（單頁可答）會高估能力 → 黃金集必含跨頁與不可答題
- **judge prompt 一改歷史分數就不可比** → judge prompt 版本號寫進每筆評分記錄
- **上下文很長時 judge 自己會漏看** → 忠實度檢查只餵與該論斷相關的頁，而非全部上下文
- **多模態證據不可省**：47/53 頁是 image，純文字 judge 會漏看一大半證據
