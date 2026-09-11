#!/usr/bin/env bash
#
# setup.sh - Dung moi truong. Tach rieng khoi run_pipeline.sh de pipeline chi
# lo chay, khong lo cai dat.
#
# Hai moi truong KHONG cai chung duoc (torch 2.7.1 + vLLM vs torch 2.9 +
# unsloth), nen phai chon ro cai nao:
#
#   bash setup.sh eval          # attribution + eval: requirements.txt + latex2sympy
#   bash setup.sh train         # train: requirements-sft.txt
#   bash setup.sh check         # chi kiem tra moi truong hien tai, khong cai gi
#
# Tuy chon:
#   bash setup.sh eval  --conda ssft_eval    # tao/dung conda env ten do
#   bash setup.sh train --venv .venv_train   # tao/dung venv o thu muc do
#   bash setup.sh train --lock               # dung requirements-sft-lock.txt
#   bash setup.sh check --for train          # kiem tra theo goi ma train can
#   bash setup.sh eval  --dry-run
#
# Khong dua --conda/--venv thi cai thang vao python dang active.
# -----------------------------------------------------------------------------

set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TARGET=""
CONDA_ENV=""
VENV_DIR=""
USE_LOCK=0
CHECK_FOR=""
DRY_RUN="${DRY_RUN:-0}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

log()  { printf '\n\033[1;34m==>\033[0m \033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[1;32mco\033[0m    %s\n' "$*"; }
miss() { printf '  \033[1;31mTHIEU\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }
run()  { echo "+ $*"; [[ "$DRY_RUN" == "1" ]] && return 0; "$@"; }

[[ $# -gt 0 ]] || { sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0; }
case "$1" in
  eval|train|check) TARGET="$1"; shift ;;
  -h|--help)        sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
  *) die "Tham so dau tien phai la 'eval', 'train' hoac 'check' (xem --help)" ;;
esac

while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda)   CONDA_ENV="$2"; shift 2 ;;
    --venv)    VENV_DIR="$2"; shift 2 ;;
    --lock)    USE_LOCK=1; shift ;;
    --for)     CHECK_FOR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "Tham so khong hop le: $1 (xem --help)" ;;
  esac
done

# =============================================================================
# Kich hoat moi truong dich (neu co yeu cau)
# =============================================================================
activate() {
  if [[ -n "$CONDA_ENV" ]]; then
    command -v conda >/dev/null 2>&1 || die "Khong tim thay lenh 'conda'."
    local base; base="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${base}/etc/profile.d/conda.sh"
    if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
      log "Tao conda env '${CONDA_ENV}' (python ${PYTHON_VERSION})"
      run conda create -y -n "$CONDA_ENV" "python=${PYTHON_VERSION}"
    fi
    [[ "$DRY_RUN" == "1" ]] || conda activate "$CONDA_ENV"
    log "Dang dung conda env: ${CONDA_ENV}"
  elif [[ -n "$VENV_DIR" ]]; then
    if [[ ! -d "$VENV_DIR" ]]; then
      log "Tao venv: ${VENV_DIR}"
      run python3 -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    [[ "$DRY_RUN" == "1" ]] || source "${VENV_DIR}/bin/activate"
    log "Dang dung venv: ${VENV_DIR}"
  else
    log "Dung python dang active: $(command -v python || command -v python3 || echo none)"
  fi
}

# =============================================================================
# check - khong cai gi, chi soi moi truong hien tai
# =============================================================================
# Cap "module_import:ten_hien_thi". Chi liet ke goi that su duoc import o
# top-level luc chay, khong ke goi chi dung trong nhanh tuy chon.
PKGS_COMMON="numpy:numpy tqdm:tqdm transformers:transformers torch:torch datasets:datasets"
PKGS_EVAL="vllm:vllm sympy:sympy mpmath:mpmath pandas:pandas regex:regex pebble:pebble
           multiprocess:multiprocess timeout_decorator:timeout-decorator word2number:word2number
           latex2sympy.latex2sympy2:latex2sympy(cai -e Eval/latex2sympy)"
PKGS_TRAIN="unsloth:unsloth trl:trl peft:peft bitsandbytes:bitsandbytes torchao:torchao"

do_check() {
  local which="${CHECK_FOR:-all}" list="$PKGS_COMMON" missing=0
  case "$which" in
    eval)  list="$PKGS_COMMON $PKGS_EVAL" ;;
    train) list="$PKGS_COMMON $PKGS_TRAIN" ;;
    all)   list="$PKGS_COMMON $PKGS_EVAL $PKGS_TRAIN" ;;
    *) die "--for phai la eval, train hoac all" ;;
  esac

  log "Kiem tra cac goi cho: ${which}"
  # latex2sympy import duoc theo cwd=Eval/, nen kiem tra tu do.
  for pair in $list; do
    local mod="${pair%%:*}" name="${pair#*:}"
    if ( cd "${ROOT_DIR}/Eval" && python -c "import ${mod}" ) >/dev/null 2>&1; then
      ok "$name"
    else
      miss "$name"
      missing=$((missing + 1))
    fi
  done

  # 'import transformers' khong keo theo quantizers, nen khong lo ra duoc loi
  # kieu torchao/torch lech phien ban. Chuoi duoi day moi la chuoi that su chay
  # khi grad_analyze.py goi AutoModelForCausalLM.from_pretrained.
  log "Thu chuoi import that (bat loi torchao / numpy-sklearn lech ABI)"
  if python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.modeling_utils      # keo theo quantizers -> torchao
import transformers.models.qwen2.modeling_qwen2
" >/dev/null 2>&1; then
    ok "transformers nap duoc model class"
  else
    miss "transformers KHONG nap duoc model class - loi that:"
    python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.modeling_utils
import transformers.models.qwen2.modeling_qwen2
" 2>&1 | tail -3 | sed 's/^/        /'
    missing=$((missing + 1))
  fi

  log "Phien ban"
  python - <<'PY' 2>/dev/null || true
mods = ["torch", "transformers", "numpy", "datasets", "trl", "unsloth", "vllm", "peft", "torchao"]
for m in mods:
    try:
        mod = __import__(m)
        print("  %-14s %s" % (m, getattr(mod, "__version__", "?")))
    except Exception:
        pass
try:
    import torch
    print("  %-14s %s" % ("cuda", torch.cuda.is_available()))
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print("    GPU %d: %s, %.0f GB" % (i, p.name, p.total_memory / 1e9))
except Exception:
    pass
PY

  echo
  if [[ "$missing" -eq 0 ]]; then
    log "Day du cho '${which}'."
  else
    die "Thieu ${missing} goi cho '${which}'. Cai bang: bash setup.sh ${which/all/eval}"
  fi
}

# =============================================================================
# install
# =============================================================================
warn_conflict() {
  # torch 2.7.1 (vLLM) va torch 2.9 (unsloth) khong the cung ton tai.
  local other="$1" mod="$2"
  if python -c "import ${mod}" >/dev/null 2>&1; then
    warn "Moi truong nay dang co '${mod}' cua ban ${other}.
      Hai bo dependency ghim torch khac nhau (2.7.1 cho vLLM, 2.9 cho unsloth)
      nen cai chong len nhau se lam hong ca hai. Nen dung env rieng:
        bash setup.sh ${TARGET} --conda ssft_${TARGET}"
  fi
}

do_install_eval() {
  warn_conflict "train" "unsloth"
  log "Cai requirements.txt (attribution + eval)"
  run pip install --upgrade pip
  run pip install -r "${ROOT_DIR}/requirements.txt"
  log "Cai latex2sympy (editable)"
  run pip install -e "${ROOT_DIR}/Eval/latex2sympy" \
    || warn "Cai latex2sympy that bai - van chay duoc vi import theo cwd=Eval/"
  CHECK_FOR="eval"; do_check
}

do_install_train() {
  warn_conflict "eval" "vllm"
  local req="requirements-sft.txt"
  [[ "$USE_LOCK" == "1" ]] && req="requirements-sft-lock.txt"
  log "Cai ${req} (train)"
  run pip install --upgrade pip
  run pip install -r "${ROOT_DIR}/${req}"
  CHECK_FOR="train"; do_check
}

activate
case "$TARGET" in
  check) do_check ;;
  eval)  do_install_eval ;;
  train) do_install_train ;;
esac
