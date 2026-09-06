#!/usr/bin/env bash
#
# eval.sh - Chay MOT LENH DUY NHAT: dung moi truong eval roi eval luon.
#
#   bash eval.sh
#
# Script tu lam tat ca:
#   1. Tao conda env RIENG cho eval - KHONG dung chung env voi train
#      (eval can vllm 0.10 + torch 2.7.1, train can torch 2.9 -> xung dot)
#   2. Cai requirements.txt (goc) + latex2sympy - bo qua neu da cai
#   3. Sinh dap an bang vLLM roi cham diem tung task, log ra logs/eval.log
#   4. In bang tong hop accuracy o cuoi
#
# Mac dinh eval checkpoint moi nhat cua ban selective SFT. Da co metrics.json
# thi task do duoc bo qua (chay lai duoc sau khi dut) - ep lam lai: --overwrite
#
# Tuy chon:
#   bash eval.sh --base                    # eval model goc, chua finetune
#   bash eval.sh --full-sft                # eval checkpoint baseline full-CoT
#   bash eval.sh --model /duong/dan/checkpoint-250
#   bash eval.sh --model /duong/dan/checkpoint-250 --tag sel_ep5
#   bash eval.sh --tasks "aime24 math500"  # chi vai task
#   bash eval.sh --n-sampling 1            # 1 mau/cau cho nhanh (mac dinh 32/6)
#   bash eval.sh --gpu 0,1                 # tensor parallel tren 2 GPU
#   bash eval.sh --max-tokens 16384        # cat ngan sinh cho nhanh
#   bash eval.sh --overwrite               # cham lai tu dau
#   bash eval.sh --skip-setup / --reinstall / --dry-run
# -----------------------------------------------------------------------------

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# =============================================================================
# Cau hinh mac dinh
# =============================================================================
ENV_NAME="${ENV_NAME:-ssft_eval}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
BASE_MODEL="${BASE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
GPU="${GPU:-0}"
LOG_DIR="${LOG_DIR:-logs}"

# Phai khop voi cach train.sh dat ten thu muc checkpoint.
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-4}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-16384}"

# "task so_mau_moi_cau" - lay tu Eval/run_eval.sh goc cua paper.
TASKS_DEFAULT="aime24:32 amc23:32 math500:6 minerva:6 gpqa:6 olympiad:6"
TASKS="${TASKS:-}"                  # rong = dung TASKS_DEFAULT
N_SAMPLING="${N_SAMPLING:-}"        # rong = dung so mau rieng cua tung task

SEED="${SEED:-0}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1}"
PROMPT_TYPE="${PROMPT_TYPE:-deepseek-longcot}"

MODEL=""            # rong = tu suy ra tu WHICH
WHICH="selective"   # selective | fullsft | base
DEFAULT_TAG=""
RUN_TAG="${RUN_TAG:-}"   # rong = tu dat ten, dung de tach output/log giua cac lan
OUTPUT_ROOT=""
OVERWRITE=0
SKIP_SETUP=0
REINSTALL=0
DRY_RUN="${DRY_RUN:-0}"

# =============================================================================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)             ENV_NAME="$2"; shift 2 ;;
    --model)           MODEL="$2"; WHICH="custom"; shift 2 ;;
    --tag)             RUN_TAG="$2"; shift 2 ;;
    --base)            WHICH="base"; shift ;;
    --full-sft)        WHICH="fullsft"; shift ;;
    --selective)       WHICH="selective"; shift ;;
    --epochs)          EPOCHS="$2"; shift 2 ;;
    --lr)              LR="$2"; shift 2 ;;
    --max-seq-length)  MAX_SEQ_LENGTH="$2"; shift 2 ;;
    --tasks)           TASKS="$2"; shift 2 ;;
    --n-sampling)      N_SAMPLING="$2"; shift 2 ;;
    --gpu)             GPU="$2"; shift 2 ;;
    --seed)            SEED="$2"; shift 2 ;;
    --max-tokens)      MAX_TOKENS="$2"; shift 2 ;;
    --temperature)     TEMPERATURE="$2"; shift 2 ;;
    --output-root)     OUTPUT_ROOT="$2"; shift 2 ;;
    --overwrite)       OVERWRITE=1; shift ;;
    --skip-setup)      SKIP_SETUP=1; shift ;;
    --reinstall)       REINSTALL=1; shift ;;
    --dry-run)         DRY_RUN=1; shift ;;
    -h|--help)         sed -n '2,28p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Tham so khong hop le: $1 (xem --help)" >&2; exit 2 ;;
  esac
done

log()  { printf '\n\033[1;34m==>\033[0m \033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }
run()  { echo "+ $*"; [[ "$DRY_RUN" == "1" ]] && return 0; "$@"; }

trap 'die "That bai tai dong $LINENO"' ERR

echo "=============================================================="
echo "  Selective SFT - setup + eval trong mot lenh"
echo "=============================================================="

# =============================================================================
# 1. Chon model can eval
# =============================================================================
# Trainer luu save_strategy=epoch nen thu muc cha chi chua cac checkpoint-<step>.
# Lay checkpoint co step lon nhat.
# In ra duong dan, hoac chuoi rong neu khong tim thay. Khong return 1: bi goi
# trong $(...) nen ERR trap se bat va in them mot dong loi thua.
latest_checkpoint() {
  local d="$1" last=""
  [[ -d "$d" ]] || { echo ""; return 0; }
  if [[ -f "$d/config.json" ]]; then echo "$d"; return 0; fi
  last="$(ls -1d "$d"/checkpoint-* 2>/dev/null \
          | sed 's#.*/checkpoint-##' | sort -n | tail -1 || true)"
  [[ -n "$last" ]] && echo "${d}/checkpoint-${last}" || echo ""
  return 0
}

CKPT_BASE="${ROOT_DIR}/SelectiveSFT/checkpoints/$(basename "$BASE_MODEL")_epoch${EPOCHS}_lr${LR}_len${MAX_SEQ_LENGTH}"

case "$WHICH" in
  base)
    MODEL="$BASE_MODEL"; DEFAULT_TAG="base" ;;
  selective)
    MODEL="$(latest_checkpoint "$CKPT_BASE")"
    [[ -n "$MODEL" ]] || die "Khong thay checkpoint trong ${CKPT_BASE} - train truoc, hoac dung --model / --base"
    DEFAULT_TAG="selective" ;;
  fullsft)
    MODEL="$(latest_checkpoint "${CKPT_BASE}_fullsft")"
    [[ -n "$MODEL" ]] || die "Khong thay checkpoint trong ${CKPT_BASE}_fullsft - chay 'bash train.sh --full-sft' truoc"
    DEFAULT_TAG="fullsft" ;;
  custom)
    [[ -n "$MODEL" ]] || die "--model rong"
    # basename khong du: hai lan train khac nhau deu co checkpoint-250, se
    # dung chung thu muc output va de log len nhau. Ghep them ten thu muc cha.
    DEFAULT_TAG="$(basename "$MODEL")"
    case "$DEFAULT_TAG" in
      checkpoint-*) DEFAULT_TAG="$(basename "$(dirname "$MODEL")")_${DEFAULT_TAG}" ;;
    esac ;;
esac

[[ -n "$RUN_TAG" ]] || RUN_TAG="${DEFAULT_TAG:-$WHICH}"

# Duong dan phai tuyet doi vi lat nua se cd sang Eval/.
[[ -d "$MODEL" ]] && MODEL="$(cd "$MODEL" && pwd)"
[[ -n "$OUTPUT_ROOT" ]] || OUTPUT_ROOT="outputs_${RUN_TAG}"

[[ -n "$TASKS" ]] || TASKS="$TASKS_DEFAULT"

# =============================================================================
# 2. Dung moi truong
# =============================================================================
STAMP=".setup_done_${ENV_NAME}_v1"

setup_env() {
  if command -v conda >/dev/null 2>&1; then
    local conda_base; conda_base="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${conda_base}/etc/profile.d/conda.sh"

    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
      log "Dung conda env san co: ${ENV_NAME}"
    else
      log "Tao conda env: ${ENV_NAME} (python ${PYTHON_VERSION})"
      run conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
      rm -f "$STAMP"
    fi
    [[ "$DRY_RUN" == "1" ]] || conda activate "$ENV_NAME"
  else
    warn "Khong co conda, chuyen sang venv."
    local venv_dir=".venv_${ENV_NAME}"
    if [[ ! -d "$venv_dir" ]]; then
      log "Tao venv: ${venv_dir}"
      run python3 -m venv "$venv_dir"
      rm -f "$STAMP"
    fi
    # shellcheck disable=SC1091
    [[ "$DRY_RUN" == "1" ]] || source "${venv_dir}/bin/activate"
  fi

  if [[ -f "$STAMP" && "$REINSTALL" == "0" ]]; then
    log "Dependency da cai truoc do (xoa ${STAMP} hoac dung --reinstall de cai lai)"
  else
    # requirements.txt o thu muc goc = moi truong vLLM cho eval.
    # KHONG cai chung voi requirements-sft.txt: torch 2.7.1 vs 2.9.0.
    log "Cai dependency eval - lan dau se lau (vllm + torch, vai GB)"
    run pip install --upgrade pip
    run pip install -r "${ROOT_DIR}/requirements.txt"

    # grader.py/parser.py import 'latex2sympy.latex2sympy2'. Import nay da
    # chay duoc vi minh cd sang Eval/, nen cai loi thi chi canh bao.
    log "Cai latex2sympy"
    run pip install -e "${ROOT_DIR}/Eval/latex2sympy" \
      || warn "Cai latex2sympy that bai - van chay duoc vi import theo cwd=Eval/"

    log "Kiem tra import vllm"
    run python -c "import vllm, torch; print(f'  torch={torch.__version__} vllm={vllm.__version__} OK')"

    [[ "$DRY_RUN" == "1" ]] || touch "$STAMP"
  fi
}

if [[ "$SKIP_SETUP" == "1" ]]; then
  log "Bo qua buoc dung moi truong (--skip-setup)"
else
  setup_env
fi

# =============================================================================
# 3. Eval
# =============================================================================
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false

EXTRA_ARGS=()
[[ "$OVERWRITE" == "1" ]] && EXTRA_ARGS+=(--overwrite)

LOG_FILE="${LOG_DIR}/eval_${RUN_TAG}.log"

log "Bat dau eval"
echo "    model      : ${MODEL}"
echo "    tasks      : ${TASKS}"
echo "    gpu        : ${GPU} (tensor_parallel_size=$(echo "$GPU" | tr ',' '\n' | grep -c .))"
echo "    sampling   : t=${TEMPERATURE} top_p=${TOP_P} seed=${SEED} max_tokens=${MAX_TOKENS}"
echo "    n_sampling : ${N_SAMPLING:-theo tung task (aime/amc 32, con lai 6)}"
echo "    output     : Eval/${OUTPUT_ROOT}/<task>/${RUN_TAG}"
echo "    log        : ${LOG_FILE}"
echo

eval_all() {
  for spec in $TASKS; do
    # Cho phep viet "aime24:32" hoac chi "aime24" (dung mac dinh cua task do).
    local task="${spec%%:*}" n="${spec#*:}"
    [[ "$n" == "$task" ]] && n=""
    if [[ -n "$N_SAMPLING" ]]; then
      n="$N_SAMPLING"
    elif [[ -z "$n" ]]; then
      case " $TASKS_DEFAULT " in
        *" ${task}:"*) n="$(echo "$TASKS_DEFAULT" | tr ' ' '\n' | grep "^${task}:" | cut -d: -f2)" ;;
        *) n=1 ;;
      esac
    fi

    echo "=============================================="
    echo "Task: ${task}  |  ${n} mau/cau"
    echo "=============================================="

    run python -u math_eval.py \
      --model_name_or_path "${MODEL}" \
      --data_name "${task}" \
      --output_dir "${OUTPUT_ROOT}/${task}/${RUN_TAG}" \
      --split "test" \
      --prompt_type "${PROMPT_TYPE}" \
      --num_test_sample -1 \
      --max_tokens_per_call "${MAX_TOKENS}" \
      --seed "${SEED}" \
      --temperature "${TEMPERATURE}" \
      --n_sampling "${n}" \
      --top_p "${TOP_P}" \
      --start 0 \
      --end -1 \
      --use_vllm \
      --save_outputs \
      --apply_chat_template \
      ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
  done
}

( cd "${ROOT_DIR}/Eval" && eval_all ) 2>&1 | tee "$LOG_FILE"

# =============================================================================
# 4. Tong hop
# =============================================================================
if [[ "$DRY_RUN" != "1" ]]; then
  log "Ket qua"
  python3 - "$ROOT_DIR/Eval/$OUTPUT_ROOT" <<'PY'
import json, sys, glob, os
root = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(root, "*", "*", "**", "*_metrics.json"), recursive=True)):
    task = os.path.relpath(f, root).split(os.sep)[0]
    try:
        m = json.load(open(f))
    except Exception:
        continue
    rows.append((task, m.get("acc"), m.get("num_samples"), m.get("time_use_in_minite")))
if not rows:
    print("  (chua co metrics.json nao trong %s)" % root)
else:
    print("  %-12s %8s %10s %10s" % ("task", "acc", "n_samples", "time"))
    for t, a, n, tm in rows:
        print("  %-12s %8s %10s %10s" % (t, f"{a:.1f}" if isinstance(a, (int, float)) else a, n, tm))
    accs = [a for _, a, _, _ in rows if isinstance(a, (int, float))]
    if accs:
        print("  %-12s %8.1f" % ("TRUNG BINH", sum(accs) / len(accs)))
PY
fi

log "Xong. Output: Eval/${OUTPUT_ROOT}  |  Log: ${LOG_FILE}"
