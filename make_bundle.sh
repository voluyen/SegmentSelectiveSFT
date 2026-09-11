#!/usr/bin/env bash
#
# make_bundle.sh - Dong goi ket qua thanh cac phan < 25 MB de tai ve tu server.
#
#   bash make_bundle.sh                              # ket qua attribution + selection
#   bash make_bundle.sh --adapter SelectiveSFT/checkpoints/<run>/checkpoint-90
#   bash make_bundle.sh --add logs/train_lora.log --add Eval/outputs_x/summary.json
#   bash make_bundle.sh --limit 20                   # doi nguong (MB)
#
# Mac dinh KHONG dua vao cac file khong lo va tai tao duoc tu du lieu goc
# (solution_segments.jsonl, IG.jsonl, solutions_selected.jsonl). Phan khong the
# tai tao la lua chon segment + diem tong hop, va ca hai deu rat nho.
# -----------------------------------------------------------------------------

set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATASET="${DATASET:-s1k}"
SELECTED="${SELECTED:-data/${DATASET}/solutions_selected.jsonl}"
IG_COMPACT="${IG_COMPACT:-Attribution/processed_data/${DATASET}/IG_compact.jsonl}"
OUT_DIR="${OUT_DIR:-bundle}"
LIMIT_MB="${LIMIT_MB:-24}"
ADAPTER=""
EXTRA=()

log()  { printf '\n\033[1;34m==>\033[0m \033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --adapter)  ADAPTER="$2"; shift 2 ;;
    --add)      EXTRA+=("$2"); shift 2 ;;
    --limit)    LIMIT_MB="$2"; shift 2 ;;
    --dataset)  DATASET="$2"; SELECTED="data/$2/solutions_selected.jsonl"
                IG_COMPACT="Attribution/processed_data/$2/IG_compact.jsonl"; shift 2 ;;
    --out)      OUT_DIR="$2"; shift 2 ;;
    -h|--help)  sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "Tham so khong hop le: $1 (xem --help)" ;;
  esac
done

rm -rf "$OUT_DIR"; mkdir -p "${OUT_DIR}/payload"

# --- 1. Lua chon segment, da bo phan text (text tai tao duoc tu dataset goc) ---
if [[ -f "$SELECTED" ]]; then
  log "Rut gon $(basename "$SELECTED") - bo text, chi giu lua chon segment"
  python3 - "$SELECTED" "${OUT_DIR}/payload/selected_spans.jsonl" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
n = 0
with open(src) as fin, open(dst, "w") as fout:
    for i, line in enumerate(fin):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        fout.write(json.dumps({
            "idx": i,
            "n_segments": len(r.get("segments", [])),
            "selected_spans_ids": r.get("selected_spans_ids", []),
        }) + "\n")
        n += 1
print("  %d mau" % n)
PY
else
  warn "Khong thay ${SELECTED} - bo qua"
fi

# --- 2. Diem IG tong hop theo segment ---
if [[ -f "$IG_COMPACT" ]]; then
  log "Them $(basename "$IG_COMPACT")"
  cp "$IG_COMPACT" "${OUT_DIR}/payload/"
else
  warn "Khong thay ${IG_COMPACT} - bo qua (chay lai stage ig de sinh ra)"
fi

# --- 3. Thong ke, de doi chieu ma khong can mo file lon ---
log "Sinh stats.json"
python3 - "$SELECTED" "${OUT_DIR}/payload/stats.json" <<'PY'
import json, os, sys
src, dst = sys.argv[1], sys.argv[2]
out = {"source": src}
if os.path.exists(src):
    ratios, empty, nseg = [], 0, []
    for line in open(src):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        s, sel = r.get("segments", []), r.get("selected_spans_ids", [])
        nseg.append(len(s))
        if s:
            ratios.append(len(sel) / len(s))
        if not sel:
            empty += 1
    out.update({
        "n_samples": len(nseg),
        "segments_per_sample_avg": round(sum(nseg) / len(nseg), 1) if nseg else 0,
        "selected_ratio_avg": round(sum(ratios) / len(ratios), 4) if ratios else 0,
        "samples_with_no_selection": empty,
    })
json.dump(out, open(dst, "w"), indent=2)
print("  " + json.dumps(out))
PY

# --- 4. Adapter LoRA (neu co) ---
if [[ -n "$ADAPTER" ]]; then
  [[ -d "$ADAPTER" ]] || die "Khong thay thu muc adapter: ${ADAPTER}"
  log "Them adapter: ${ADAPTER}"
  mkdir -p "${OUT_DIR}/payload/adapter"
  # Chi lay file cua adapter, khong lay optimizer/scheduler state.
  for f in adapter_model.safetensors adapter_config.json README.md \
           tokenizer_config.json tokenizer.json special_tokens_map.json; do
    [[ -f "${ADAPTER}/${f}" ]] && cp "${ADAPTER}/${f}" "${OUT_DIR}/payload/adapter/"
  done
  du -sh "${OUT_DIR}/payload/adapter" | sed 's/^/    /'
fi

for f in ${EXTRA[@]+"${EXTRA[@]}"}; do
  [[ -f "$f" ]] && { log "Them ${f}"; cp "$f" "${OUT_DIR}/payload/"; } || warn "Khong thay ${f}"
done

# --- 5. Nen va cat nho ---
log "Nen va cat thanh phan <= ${LIMIT_MB} MB"
tar -czf "${OUT_DIR}/bundle.tar.gz" -C "${OUT_DIR}" payload
TOTAL=$(du -m "${OUT_DIR}/bundle.tar.gz" | cut -f1)
split -b "${LIMIT_MB}m" "${OUT_DIR}/bundle.tar.gz" "${OUT_DIR}/bundle.tar.gz.part"
rm -f "${OUT_DIR}/bundle.tar.gz"
rm -rf "${OUT_DIR}/payload"

( cd "$OUT_DIR" && (shasum -a 256 bundle.tar.gz.part* > SHA256SUMS 2>/dev/null \
                    || sha256sum bundle.tar.gz.part* > SHA256SUMS) )

cat > "${OUT_DIR}/GHEP_LAI.txt" <<TXT
Tai het cac file bundle.tar.gz.part* roi ghep lai o may cua ban:

    cat bundle.tar.gz.part* > bundle.tar.gz
    tar -xzf bundle.tar.gz

Kiem tra toan ven truoc khi ghep:
    shasum -a 256 -c SHA256SUMS      # hoac: sha256sum -c SHA256SUMS
TXT

echo
log "Xong - ${OUT_DIR}/ (tong ${TOTAL} MB)"
ls -lh "$OUT_DIR" | tail -n +2 | awk '{printf "    %-28s %s\n", $NF, $5}'
echo
echo "  Tai ve tat ca file tren, roi lam theo ${OUT_DIR}/GHEP_LAI.txt"
