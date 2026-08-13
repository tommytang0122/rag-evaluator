#!/usr/bin/env python3
"""Render docs/golden-set.html from golden.jsonl.

Usage: python3 docs/build_golden_set_html.py [--out PATH] [--bare]
--bare omits the <!doctype>/<head>/<body> skeleton (for claude.ai Artifact publishing).
"""

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 人工驗證備註(出題時對照頁面 PNG 的位置)
NOTES = {
    "q-4bee55b5": "頁首載明「2023年09月／製表日期 2023/10/01」",
    "q-e2dda906": "稅後利益列 × 2023年1~9月（自結）欄。生成器原稿誤植 150 億元，審核時已更正",
    "q-5ca89663": "每月固定成本 1,560 萬美元、累計虧損 3.70 億美元、台塑美國調價提案爭議均見於本頁",
    "smoke-001": "塑膠一部／嘉義三廠列 × 淨利金額欄",
    "smoke-002": "營業額列 × 2023年1~9月（自結）欄",
    "smoke-003": "刻意虛構的年份與廠區",
    "man-01": "塑膠一部／硬布一廠列 × 淨利金額欄",
    "man-02": "塑膠三部小計列 × 淨利金額欄",
    "man-03": "塑膠三部／嘉義一廠列 × 淨利金額欄",
    "man-04": "塑膠一部／樹林一廠列 × 毛利金額欄（負值）",
    "man-05": "化工一部／麥寮異辛醇廠列 × 淨利金額欄",
    "man-06": "化工二部小計列 × 淨利金額欄（虧損）",
    "man-07": "化工二部／麥寮丙二酚廠列 × 停工損失及備品跌價損失欄",
    "man-08": "電子材料部／銅箔二廠列 × 淨利金額欄",
    "man-09": "纖維部小計列 × 淨利金額欄（虧損）",
    "man-10": "聚酯膜部／離型膜廠列 × 淨利金額欄",
    "man-11": "工務部／錦興公用廠列 × 淨利金額欄",
    "man-12": "全公司合計列 × 淨利金額欄",
    "man-13": "稅前利益列 × 2023年1~9月（自結）欄",
    "man-14": "權益收益／南電列 × 2023年1~9月（自結）欄",
    "man-15": "第二點：截至2023年8月累計虧損3.70億美元",
    "man-16": "第二點：每月固定成本達1,560萬美元（折舊約637萬、利息約636萬）",
    "man-17": "內文：早在1996年就提出「虛擬晶圓廠」概念",
    "man-18": "內文：2000年推出設計服務聯盟（Design Service Alliance）",
    "man-19": "配置圖綠色標籤共 8 項設備",
    "man-20": "第1頁塑膠一部小計 8,084；第4頁工務部小計 75,626",
    "man-21": "公開利益表營業額（自結）91,007,923；損益表第4頁收入合計 13,032,770",
    "man-22": "語料僅有 2023年9月 的分廠損益彙總表",
    "man-23": "德州報告僅載 EG1 供應價（639 美元/噸），無 EG1 營業額",
}

def load_scores(scores_path: Path) -> dict[str, str]:
    """scores.jsonl → {qid: 成績摘要 HTML}。取每題 run 0 的列。"""
    out: dict[str, str] = {}
    for line in scores_path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("run") not in (0, "0") or r["qid"] in out:
            continue

        def _fmt(v):
            if v is None:
                return "n/a"
            s = f"{v:.2f}".rstrip("0").rstrip(".")
            return f"<b>{s}</b>" if v == 1 or v == 2 else s

        c = r.get("correctness")
        cs = f"<b>{c}</b>" if c == 2 else ("?" if c is None else str(c))
        parts = [f"correctness {cs}（{r.get('method')}）"]
        ret = r.get("retrieval")
        if ret:
            parts.append(f"recall {_fmt(ret.get('evidence_recall'))}")
            parts.append(f"MRR {_fmt(ret.get('mrr'))}")
        if r.get("faithfulness") is not None:
            parts.append(f"faith {_fmt(r['faithfulness'])}")
        if r.get("hallucinated_answer"):
            parts.append('<span style="color:var(--seal)">hallucinated</span>')
        out[r["qid"]] = "｜".join(parts)
    return out

SECTIONS = [
    ("一、分廠損益彙總表（2023年09月）", "202309-分廠損益彙總表"),
    ("二、南亞公告利益表（2023年1~9月）", "2023年1-9月公開利益及權益收益"),
    ("三、美國德州公司營運情況報告", "美國德州公司營運情況報告"),
    ("四、台積電數位轉型（文章）", "台積電數位轉型"),
    ("五、膠皮二廠PVC膠皮機簡報", "膠皮二廠"),
]

CSS = """
  :root { --paper:#FAFAF7; --ink:#1C2430; --muted:#5C6672; --rule:#D9DCD4;
          --seal:#B5352C; --well:#F1F2EC; --ok:#2E6B4F; }
  @media (prefers-color-scheme: dark) {
    :root { --paper:#151A21; --ink:#E4E6E1; --muted:#98A0A8; --rule:#2B323C;
            --seal:#E06A5E; --well:#1D242E; --ok:#6FBF97; } }
  :root[data-theme="dark"] { --paper:#151A21; --ink:#E4E6E1; --muted:#98A0A8;
    --rule:#2B323C; --seal:#E06A5E; --well:#1D242E; --ok:#6FBF97; }
  :root[data-theme="light"] { --paper:#FAFAF7; --ink:#1C2430; --muted:#5C6672;
    --rule:#D9DCD4; --seal:#B5352C; --well:#F1F2EC; --ok:#2E6B4F; }
  body { background:var(--paper); color:var(--ink);
    font-family:"Noto Sans TC","PingFang TC","Microsoft JhengHei","Helvetica Neue",sans-serif;
    line-height:1.75; margin:0; padding:3.5rem 1.25rem 5rem; }
  .sheet { max-width:44rem; margin:0 auto; }
  h1, h2, .q { font-family:"Noto Serif TC","Songti TC","PMingLiU",serif; }
  header { border-bottom:2px solid var(--ink); padding-bottom:1.5rem; }
  .eyebrow { font-size:.78rem; letter-spacing:.35em; text-transform:uppercase;
    color:var(--seal); font-weight:600; }
  h1 { font-size:2rem; font-weight:700; margin:.4rem 0 .8rem; letter-spacing:.06em; text-wrap:balance; }
  .meta { display:flex; flex-wrap:wrap; gap:.35rem 2rem; font-size:.82rem;
    color:var(--muted); font-variant-numeric:tabular-nums; }
  .meta b { color:var(--ink); font-weight:600; }
  .note { margin:1.6rem 0 0; padding:.9rem 1.1rem; background:var(--well);
    border-left:3px solid var(--seal); font-size:.88rem; color:var(--muted);
    border-radius:0 4px 4px 0; }
  .note b { color:var(--ink); }
  h2 { font-size:1.12rem; letter-spacing:.1em; margin:3rem 0 .4rem;
    display:flex; align-items:baseline; gap:.75rem; }
  h2 small { font-size:.78rem; color:var(--muted); letter-spacing:.05em;
    font-family:"Noto Sans TC",sans-serif; font-weight:400; }
  .sec-rule { border:0; border-top:1px solid var(--rule); margin:0 0 .5rem; }
  .item { padding:1.6rem 0 1.8rem; border-bottom:1px solid var(--rule); }
  .item:last-child { border-bottom:0; }
  .idrow { display:flex; align-items:center; gap:.6rem; flex-wrap:wrap; }
  .qid { font-family:ui-monospace,"SF Mono",Consolas,monospace; font-size:.74rem;
    color:var(--muted); letter-spacing:.03em; }
  .stamp { font-size:.7rem; font-weight:700; letter-spacing:.18em;
    padding:.1rem .55rem .1rem .7rem; border-radius:3px; border:1.5px solid currentColor; }
  .stamp.ans { color:var(--ok); }
  .stamp.ref { color:var(--seal); }
  .tag { font-size:.72rem; color:var(--muted); }
  .tag::before { content:"#"; opacity:.6; }
  .q { font-size:1.22rem; font-weight:700; margin:.55rem 0 .9rem; line-height:1.55; text-wrap:balance; }
  dl { margin:0; display:grid; grid-template-columns:5.5em 1fr; gap:.45rem 1rem; font-size:.92rem; }
  dt { color:var(--muted); font-size:.8rem; letter-spacing:.1em; padding-top:.12em; }
  dd { margin:0; }
  .gold { background:var(--well); padding:.55rem .8rem; border-radius:4px;
    font-variant-numeric:tabular-nums; }
  .gold .num { font-weight:700; font-size:1.05em; }
  .gold.refusal { color:var(--muted); }
  .gold.refusal em { color:var(--seal); font-style:normal; font-weight:600; }
  .ev { font-size:.88rem; font-variant-numeric:tabular-nums; }
  .ev .doc { font-weight:600; }
  .ev .pg { color:var(--seal); font-weight:600; white-space:nowrap; }
  .verify { font-size:.8rem; color:var(--muted); }
  .score { display:flex; flex-wrap:wrap; gap:.3rem 1.4rem; font-size:.8rem;
    color:var(--muted); font-variant-numeric:tabular-nums; }
  .score b { color:var(--ok); font-weight:700; }
  .pending { font-size:.78rem; color:var(--muted); border:1px dashed var(--rule);
    border-radius:3px; padding:.05rem .5rem; }
  footer { margin-top:3.5rem; padding-top:1rem; border-top:2px solid var(--ink);
    font-size:.78rem; color:var(--muted); line-height:1.9; }
  footer code { font-family:ui-monospace,"SF Mono",Consolas,monospace; font-size:.95em; }
"""


def doc_stem(path: str) -> str:
    return Path(path).name


def render_item(it: dict, scores: dict[str, str], run_label: str) -> str:
    qid = it["id"]
    refusal = it["answer_type"] == "refusal"
    stamp = '<span class="stamp ref">拒 答</span>' if refusal else '<span class="stamp ans">可 答</span>'
    tags = "".join(f'<span class="tag">{html.escape(t)}</span>' for t in it.get("tags", []))
    out = [f'<article class="item"><div class="idrow"><span class="qid">{qid}</span>{stamp}{tags}</div>']
    out.append(f'<p class="q">{html.escape(it["question"])}</p><dl>')

    if refusal:
        note = html.escape(NOTES.get(qid, "語料中不存在此資訊"))
        out.append(f'<dt>標準答案</dt><dd class="gold refusal">{note}——正確行為是<em>拒答</em>。</dd>')
    else:
        gold = html.escape(it["gold_answer"])
        gv = it.get("gold_value")
        gv_note = ""
        if gv:
            gv_note = (f'　<span class="verify">gold_value：{{number: {gv["number"]}, '
                       f'unit: {html.escape(str(gv.get("unit") or ""))}}}</span>')
        out.append(f'<dt>標準答案</dt><dd class="gold"><span class="num">{gold}</span>{gv_note}</dd>')
        evs = it.get("evidence", [])
        if evs:
            docs = {}
            for e in evs:
                docs.setdefault(doc_stem(e["document"]), []).append(e["page"])
            ev_html = "；".join(
                f'<span class="doc">{html.escape(d)}</span> '
                f'<span class="pg">第 {"、".join(map(str, ps))} 頁</span>'
                for d, ps in docs.items()
            )
            verify = NOTES.get(qid)
            vline = f'<br><span class="verify">驗證：{html.escape(verify)}</span>' if verify else ""
            out.append(f'<dt>證據頁</dt><dd class="ev">{ev_html}{vline}</dd>')

    if qid in scores:
        out.append(f'<dt>評測結果</dt><dd class="score"><span>{scores[qid]}</span>'
                   f'<span class="verify">（run: {html.escape(run_label)}）</span></dd>')
    else:
        out.append('<dt>評測結果</dt><dd><span class="pending">尚未評測</span></dd>')
    out.append("</dl></article>")
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "golden-set.html"))
    ap.add_argument("--bare", action="store_true")
    ap.add_argument("--scores", default=None, help="runs/<id>/scores*.jsonl 路徑")
    ap.add_argument("--run-label", default="", help="成績來源標籤,如 golden-02(v2)")
    args = ap.parse_args()

    scores = load_scores(Path(args.scores)) if args.scores else {}
    run_label = args.run_label or (Path(args.scores).parent.name if args.scores else "")

    items = [json.loads(l) for l in (ROOT / "golden.jsonl").open(encoding="utf-8") if l.strip()]
    n_ans = sum(1 for i in items if i["answer_type"] == "answerable")
    n_ref = len(items) - n_ans
    n_cross = sum(1 for i in items if "cross-page" in i.get("tags", []))

    single = [i for i in items if i["answer_type"] == "answerable" and "cross-page" not in i.get("tags", [])]
    cross = [i for i in items if "cross-page" in i.get("tags", [])]
    refusals = [i for i in items if i["answer_type"] == "refusal"]

    sections = []
    used = set()
    for title, key in SECTIONS:
        grp = [i for i in single if key in i["evidence"][0]["document"]]
        for i in grp:
            used.add(i["id"])
        if grp:
            sections.append((title, "", grp))
    leftover = [i for i in single if i["id"] not in used]
    if leftover:
        sections.append(("其他", "", leftover))
    if cross:
        sections.append(("六、跨頁／跨文件題", "檢索需同時命中多頁", cross))
    if refusals:
        sections.append(("七、拒答題", "語料中不存在，正確行為是拒答", refusals))

    body = [f"""<div class="sheet">
  <header>
    <div class="eyebrow">RAG Evaluator · gavin_test</div>
    <h1>黃金問題集 v1</h1>
    <div class="meta">
      <span>共 <b>{len(items)}</b> 題</span>
      <span>可答 <b>{n_ans}</b>／拒答 <b>{n_ref}</b>／跨頁 <b>{n_cross}</b></span>
      <span>語料 <b>51</b> 頁 / 5 份 PDF</span>
      <span>更新 <b>2026-07-27</b></span>
    </div>
    <p class="note">
      每題 gold 均經<b>人工讀頁驗證</b>（逐頁開啟 PNG 核對數字與敘述後出題）。
      本頁由 <b>docs/build_golden_set_html.py</b> 自 golden.jsonl 與評測 scores 生成。
    </p>
  </header>"""]

    for title, small, grp in sections:
        small_html = f" <small>{html.escape(small)}</small>" if small else ""
        body.append(f'<h2>{html.escape(title)}{small_html}</h2><hr class="sec-rule">')
        body.extend(render_item(i, scores, run_label) for i in grp)

    body.append("""<footer>
    檔案：<code>golden.jsonl</code>（30 題題庫）、<code>smoke.jsonl</code>（煙霧子集）、<code>review.csv</code>（生成器審核底稿）<br>
    受測系統：nas-rag（Qdrant gavin_test · Gemini 2048 維檢索 · Sonnet 5 作答）· judge：gemini-2.5-flash (prompt v2)<br>
    30／30 題達標；評測新題請換新的 run-id（dataset SHA 已變更）。
  </footer>
</div>""")

    content = f"<title>RAG 黃金問題集 v1</title>\n<style>{CSS}</style>\n" + "\n".join(body) + "\n"
    if not args.bare:
        content = ('<!doctype html>\n<html lang="zh-Hant">\n<head>\n<meta charset="utf-8">\n'
                   '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                   + content.replace("</style>\n", "</style>\n</head>\n<body>\n", 1)
                   + "</body>\n</html>\n")
    Path(args.out).write_text(content, encoding="utf-8")
    print(f"wrote {args.out} ({len(items)} items)")


if __name__ == "__main__":
    main()
