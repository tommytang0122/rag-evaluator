#!/usr/bin/env bash
#
# 一鍵評測:.env → system.yaml → preflight → collect → score → report
#
#   ./run-eval.sh [run-id] [--runs N] [--dataset f] [--baseline id] [--skip-preflight]
#
# 人類只需要維護 .env(見 .env.example)。受測系統的設定會生成到
# runs/<run-id>/system.yaml——設定與結果放同一個目錄,溯源天然,且不會動到
# repo 根目錄那份手寫的 system.yaml。
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

RUN_ID=""
RUNS=3
DATASET="golden.jsonl"
CORPUS="corpus.jsonl"
BASELINE=""
SKIP_PREFLIGHT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs)           RUNS="$2"; shift 2 ;;
    --dataset)        DATASET="$2"; shift 2 ;;
    --corpus)         CORPUS="$2"; shift 2 ;;
    --baseline)       BASELINE="$2"; shift 2 ;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
    -h|--help)        sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*)               echo "未知選項:$1" >&2; exit 1 ;;
    *)                RUN_ID="$1"; shift ;;
  esac
done

RUN_ID="${RUN_ID:-prod-$(date +%Y%m%d-%H%M)}"
RUN_DIR="runs/$RUN_ID"
SYSTEM_YAML="$RUN_DIR/system.yaml"

say()  { printf '\n\033[1m▸ %s\033[0m\n' "$*"; }
warn() { printf '\033[33m! %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# --- 1. 載入 .env ----------------------------------------------------------
# 用逐行解析而非 `source`,理由有二:.env 的值不該被當 shell 執行;且未加引號
# 的空白值(如 "Bearer sk-x")在 source 下會壞掉,但 python-dotenv 讀得好好的,
# 兩邊行為必須一致。既有的環境變數優先,與 python-dotenv 的預設相同。
ENV_FILE="${ENV_FILE:-.env}"
[[ -f "$ENV_FILE" ]] || die "$ENV_FILE 不存在——先 cp .env.example .env 並填值"
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
  [[ "$line" != *=* ]] && continue
  key="${line%%=*}"; val="${line#*=}"
  key="${key#"${key%%[![:space:]]*}"}"; key="${key%"${key##*[![:space:]]}"}"
  key="${key#export }"
  val="${val#"${val%%[![:space:]]*}"}"; val="${val%"${val##*[![:space:]]}"}"
  [[ "$val" == \"*\" || "$val" == \'*\' ]] && val="${val:1:${#val}-2}"
  [[ -z "${!key:-}" ]] && export "$key=$val"
done < "$ENV_FILE"

missing=()
for v in GEMINI_API_KEY RAG_EVAL_ENDPOINT RAG_EVAL_COLLECTION; do
  [[ -z "${!v:-}" ]] && missing+=("$v")
done
[[ ${#missing[@]} -gt 0 ]] && die ".env 缺少必填項:${missing[*]}(說明見 .env.example)"

[[ -f "$DATASET" ]] || die "找不到黃金集 $DATASET"

# --- 2. 生成 runs/<run-id>/system.yaml -------------------------------------
say "生成 $SYSTEM_YAML"
mkdir -p "$RUN_DIR"
uv run python - "$SYSTEM_YAML" <<'PY'
import os, sys, yaml

cfg = {
    "adapter": os.environ.get("RAG_EVAL_ADAPTER", "nas_rag"),
    "endpoint": os.environ["RAG_EVAL_ENDPOINT"],
    "collection_names": [
        c.strip() for c in os.environ["RAG_EVAL_COLLECTION"].split(",") if c.strip()
    ],
    "top_k": int(os.environ.get("RAG_EVAL_TOP_K") or 5),
    "timeout_s": float(os.environ.get("RAG_EVAL_TIMEOUT_S") or 300),
}

# token 的「值」絕不進 system.yaml——run manifest 會把整份 config 落盤。
# 這裡只釘住「用了哪個環境變數」,adapter 在建 client 時才去環境解析。
if os.environ.get("RAG_EVAL_AUTH_TOKEN"):
    cfg["auth_env"] = "RAG_EVAL_AUTH_TOKEN"
    cfg["auth_header"] = os.environ.get("RAG_EVAL_AUTH_HEADER") or "Authorization"

tls = (os.environ.get("RAG_EVAL_TLS_VERIFY") or "true").strip()
if tls.lower() in ("false", "0", "no"):
    cfg["verify"] = False
elif tls.lower() not in ("true", "1", "yes", ""):
    cfg["verify"] = tls          # CA bundle 路徑

diag = {}
if os.environ.get("RAG_EVAL_CUTOFF_PROBE_TOP_K"):
    diag["cutoff_probe_top_k"] = int(os.environ["RAG_EVAL_CUTOFF_PROBE_TOP_K"])
if (os.environ.get("RAG_EVAL_TYPE_BUCKETS") or "").lower() in ("true", "1", "yes"):
    diag["type_buckets"] = True
if diag:
    cfg["diagnostics"] = diag

with open(sys.argv[1], "w", encoding="utf-8") as fh:
    yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
PY
sed 's/^/    /' "$SYSTEM_YAML"

# --- 3. Preflight:先確認契約通,再花時間跑整份題庫 ------------------------
if [[ "$SKIP_PREFLIGHT" == "0" ]]; then
  say "Preflight:打一發真的 query"
  curl_args=(-sS --max-time "${RAG_EVAL_TIMEOUT_S:-300}" -w '\n%{http_code}'
             -X POST "$RAG_EVAL_ENDPOINT" -H 'Content-Type: application/json')
  [[ -n "${RAG_EVAL_AUTH_TOKEN:-}" ]] &&
    curl_args+=(-H "${RAG_EVAL_AUTH_HEADER:-Authorization}: $RAG_EVAL_AUTH_TOKEN")
  case "$(echo "${RAG_EVAL_TLS_VERIFY:-true}" | tr '[:upper:]' '[:lower:]')" in
    false|0|no) curl_args+=(-k) ;;
    true|1|yes|"") ;;
    *) curl_args+=(--cacert "$RAG_EVAL_TLS_VERIFY") ;;
  esac

  payload=$(uv run python - "$DATASET" <<'PY'
import json, os, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    q = next(json.loads(l) for l in fh if l.strip())["question"]
print(json.dumps({
    "query": q,
    "collection_names": [c.strip() for c in os.environ["RAG_EVAL_COLLECTION"].split(",") if c.strip()],
    "top_k": int(os.environ.get("RAG_EVAL_TOP_K") or 5),
}, ensure_ascii=False))
PY
)
  echo "    query: $(echo "$payload" | head -c 160)"
  response=$(curl "${curl_args[@]}" -d "$payload" 2>&1) || {
    echo "$response" | sed 's/^/    /' >&2
    die "Preflight 連線失敗——endpoint / 認證 / TLS 任一項不對。先修 .env 再重跑。"
  }
  http_code=$(echo "$response" | tail -n1)
  body=$(echo "$response" | sed '$d')
  if [[ "$http_code" != "200" ]]; then
    echo "$body" | head -c 500 | sed 's/^/    /' >&2
    die "Preflight 回 HTTP $http_code(endpoint 要含完整路徑,如 .../api/v1/query;401/403 檢查 RAG_EVAL_AUTH_TOKEN)"
  fi
  # body 走暫存檔而非管線:heredoc 已經佔用了 python 的 stdin。
  body_file=$(mktemp); printf '%s' "$body" > "$body_file"
  trap 'rm -f "$body_file"' EXIT
  uv run python - "$body_file" <<'PY' || die "Preflight 回應不符 adapter 契約,collect 會全題 system_error"
import json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    d = json.load(fh)
answer, sources = d.get("answer"), d.get("sources") or []
if not answer:
    sys.exit("    answer 欄位為空或不存在")
usable = [s for s in sources if "source" in s and "page" in s]
print(f"    answer: {answer[:100]}...")
print(f"    sources: {len(sources)} 筆,其中 {len(usable)} 筆有 source+page")
if not usable:
    sys.exit("    沒有任何 source 同時具備 source 與 page——檢索指標會全 0")
print(f"    範例 source: {usable[0]['source']} 第 {usable[0]['page']} 頁")
PY
fi

# --- 4. corpus:有就沿用,沒有就試著從 Qdrant 建 --------------------------
corpus_args=()
if [[ -f "$CORPUS" ]]; then
  say "corpus:沿用 $CORPUS($(wc -l < "$CORPUS") 頁)"
  corpus_args=(--corpus "$CORPUS")
else
  say "corpus:$CORPUS 不存在,試著從 Qdrant 建"
  collection_args=()
  IFS=',' read -ra cols <<< "$RAG_EVAL_COLLECTION"
  for c in "${cols[@]}"; do collection_args+=(--collection "$(echo "$c" | xargs)"); done
  if uv run rag-eval corpus from-qdrant "${collection_args[@]}" -o "$CORPUS"; then
    corpus_args=(--corpus "$CORPUS")
  else
    rm -f "$CORPUS"
    warn "建不出 corpus(RAG_EVAL_QDRANT_URL=${RAG_EVAL_QDRANT_URL:-未設})。"
    warn "評測照跑,但忠實度只能用 sources 附帶的文字,部分論斷會記 evidence_unavailable。"
  fi
fi

# --- 5. collect → score → report -------------------------------------------
say "collect:$DATASET × $RUNS runs → $RUN_DIR"
if ! uv run rag-eval collect --system "$SYSTEM_YAML" --dataset "$DATASET" \
       --run-id "$RUN_ID" --runs "$RUNS"; then
  die "collect 失敗。若是 manifest mismatch:.env 或黃金集改過了,換一個 run-id 重跑。"
fi

say "score:judge=${RAG_EVAL_JUDGE_MODEL:-gemini-2.5-flash}"
uv run rag-eval score --dataset "$DATASET" --run-id "$RUN_ID" "${corpus_args[@]}" \
  || die "score 失敗。檢查 GEMINI_API_KEY 與網路;judge prompt 改過的話用 --rescore-tag。"

say "report"
report_args=(--dataset "$DATASET" --run-id "$RUN_ID")
[[ -n "$BASELINE" ]] && report_args+=(--baseline "$BASELINE")
uv run rag-eval report "${report_args[@]}"

say "完成"
sed -n '/^## 總覽/,/^## /p' "$RUN_DIR/report.md" | sed '$d'
echo "完整報表:$RUN_DIR/report.md"
