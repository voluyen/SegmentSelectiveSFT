#!/usr/bin/env bash
#
# make_bundle.sh - Cat file du lieu thanh nhieu phan nho de tai ve, giu NGUYEN
# noi dung (ke ca phan text). Ghep lai bang 'cat' la ra dung file ban dau.
#
#   bash make_bundle.sh                      # cat cac file du lieu mac dinh
#   bash make_bundle.sh --limit 20           # doi nguong moi phan (MB)
#   bash make_bundle.sh --add logs/train_lora.log
#   bash make_bundle.sh --only data/s1k/solutions_selected.jsonl
#   bash make_bundle.sh --full-ig            # them ca IG.jsonl per-token (rat lon)
#
# File .jsonl duoc cat theo RANH GIOI DONG nen tung phan van la jsonl hop le,
# mo ra xem duoc ngay. File khac cat theo byte. Ca hai deu ghep lai bang 'cat'.
# -----------------------------------------------------------------------------

set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATASET="${DATASET:-s1k}"
OUT_DIR="${OUT_DIR:-bundle}"
LIMIT_MB="${LIMIT_MB:-24}"
FULL_IG=0
ONLY=()
EXTRA=()

log()  { printf '\n\033[1;34m==>\033[0m \033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)   LIMIT_MB="$2"; shift 2 ;;
    --add)     EXTRA+=("$2"); shift 2 ;;
    --only)    ONLY+=("$2"); shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --out)     OUT_DIR="$2"; shift 2 ;;
    --full-ig) FULL_IG=1; shift ;;
    -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "Tham so khong hop le: $1 (xem --help)" ;;
  esac
done

# --- Danh sach file can cat ---
FILES=()
if [[ ${#ONLY[@]} -gt 0 ]]; then
  FILES=("${ONLY[@]}")
else
  # Ket qua chon segment: GIU NGUYEN text.
  FILES+=("data/${DATASET}/solutions_selected.jsonl")
  # Diem IG tong hop theo segment.
  FILES+=("Attribution/processed_data/${DATASET}/IG_compact.jsonl")
  [[ "$FULL_IG" == "1" ]] && FILES+=("Attribution/processed_data/${DATASET}/IG.jsonl")
fi
FILES+=(${EXTRA[@]+"${EXTRA[@]}"})

rm -rf "$OUT_DIR"; mkdir -p "$OUT_DIR"

SPLIT_ANY=0
for f in "${FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    warn "Khong thay ${f} - bo qua"
    continue
  fi
  log "Cat $(basename "$f")"
  python3 - "$f" "$OUT_DIR" "$LIMIT_MB" <<'PY'
import os, sys

src, out_dir, limit_mb = sys.argv[1], sys.argv[2], int(sys.argv[3])
limit = limit_mb * 1024 * 1024
base = os.path.basename(src)
total = os.path.getsize(src)

if total <= limit:
    # Van copy nguyen file, khong cat - tai ve truc tiep duoc.
    dst = os.path.join(out_dir, base)
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while True:
            chunk = fi.read(1 << 20)
            if not chunk:
                break
            fo.write(chunk)
    print("    %.1f MB - khong can cat" % (total / 1e6))
    sys.exit(0)

parts, idx = [], 0

def open_part():
    global idx
    p = os.path.join(out_dir, "%s.part%02d" % (base, idx))
    idx += 1
    parts.append(p)
    return open(p, "wb")

if src.endswith(".jsonl"):
    # Cat theo ranh gioi dong: moi phan van la jsonl hop le.
    fo, size = open_part(), 0
    with open(src, "rb") as fi:
        for line in fi:
            if size and size + len(line) > limit:
                fo.close()
                fo, size = open_part(), 0
            fo.write(line)
            size += len(line)
    fo.close()
else:
    # File khong phai jsonl: cat theo byte.
    with open(src, "rb") as fi:
        while True:
            fo, size = open_part(), 0
            while size < limit:
                chunk = fi.read(min(1 << 20, limit - size))
                if not chunk:
                    break
                fo.write(chunk)
                size += len(chunk)
            fo.close()
            if size == 0:
                os.remove(parts.pop())
                break
            if size < limit:
                break

print("    %.1f MB -> %d phan" % (total / 1e6, len(parts)))
for p in parts:
    print("      %-44s %5.1f MB" % (os.path.basename(p), os.path.getsize(p) / 1e6))
PY
  SPLIT_ANY=1
done

[[ "$SPLIT_ANY" == "1" ]] || die "Khong co file nao de cat."

log "Checksum"
( cd "$OUT_DIR" && (shasum -a 256 ./* > SHA256SUMS 2>/dev/null || sha256sum ./* > SHA256SUMS) ) || true

# --- Huong dan ghep lai ---
{
  echo "Tai het cac file trong thu muc nay ve, roi ghep lai:"
  echo
  for f in "${FILES[@]}"; do
    b="$(basename "$f")"
    if ls "${OUT_DIR}/${b}.part"* >/dev/null 2>&1; then
      echo "    cat ${b}.part* > ${b}"
    elif [[ -f "${OUT_DIR}/${b}" ]]; then
      echo "    # ${b} khong bi cat, dung luon"
    fi
  done
  echo
  echo "Kiem tra toan ven (chay TRUOC khi ghep):"
  echo "    shasum -a 256 -c SHA256SUMS      # hoac: sha256sum -c SHA256SUMS"
  echo
  echo "Kiem tra file jsonl sau khi ghep:"
  echo "    wc -l <ten_file>.jsonl"
  echo "    python3 -c \"import json;[json.loads(l) for l in open('<ten_file>.jsonl')];print('jsonl hop le')\""
} > "${OUT_DIR}/GHEP_LAI.txt"

echo
log "Xong - ${OUT_DIR}/"
ls -lh "$OUT_DIR" | tail -n +2 | awk '{printf "    %-46s %s\n", $NF, $5}'
echo
cat "${OUT_DIR}/GHEP_LAI.txt" | sed 's/^/  /'
