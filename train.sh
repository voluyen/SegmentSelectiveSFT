#!/usr/bin/env bash
#
# train.sh - Chay MOT LENH DUY NHAT: dung moi truong roi train luon.
#
#   bash train.sh
#
# Script tu lam tat ca:
#   1. Tao conda env (hoac venv neu khong co conda) - bo qua neu da co
#   2. Cai SelectiveSFT/requirements.txt      - bo qua neu da cai
#   3. Train (wandb da tat, log ra logs/train.log)
#
# Day la FULL finetuning nen rat ton VRAM. Mac dinh: seq 16384, batch 2,
# grad_accum 1 - tu chinh bang --batch-size / --grad-accum / --max-seq-length.
#
# Tuy chon:
#   bash train.sh --epochs 5 --lr 1e-5
#   bash train.sh --gpu 1                  # dung GPU khac
#   bash train.sh --batch-size 1 --grad-accum 2 --max-seq-length 8192
#   bash train.sh --reinstall              # cai lai dependency
#   bash train.sh --skip-setup             # bo qua buoc dung env
#   bash train.sh --dry-run                # chi in lenh
# -----------------------------------------------------------------------------

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# =============================================================================
# Cau hinh mac dinh
# =============================================================================
ENV_NAME="${ENV_NAME:-ssft_train}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
DATA="${DATA:-data/limo/solutions_top70cohe80_lennorm_7B_J50.jsonl}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-3e-5}"
LOG_DIR="${LOG_DIR:-logs}"

MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-16384}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"

SKIP_SETUP=0
REINSTALL=0
DRY_RUN="${DRY_RUN:-0}"

# =============================================================================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)             ENV_NAME="$2"; shift 2 ;;
    --model)           MODEL="$2"; shift 2 ;;
    --data)            DATA="$2"; shift 2 ;;
    --gpu)             GPU="$2"; shift 2 ;;
    --epochs)          EPOCHS="$2"; shift 2 ;;
    --lr)              LR="$2"; shift 2 ;;
    --max-seq-length)  MAX_SEQ_LENGTH="$2"; shift 2 ;;
    --batch-size)      BATCH_SIZE="$2"; shift 2 ;;
    --grad-accum)      GRAD_ACCUM="$2"; shift 2 ;;
    --skip-setup)      SKIP_SETUP=1; shift ;;
    --reinstall)       REINSTALL=1; shift ;;
    --dry-run)         DRY_RUN=1; shift ;;
    -h|--help)         sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Tham so khong hop le: $1 (xem --help)" >&2; exit 2 ;;
  esac
done

log()  { printf '\n\033[1;34m==>\033[0m \033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }
run()  { echo "+ $*"; [[ "$DRY_RUN" == "1" ]] && return 0; "$@"; }

trap 'die "That bai tai dong $LINENO"' ERR

echo "=============================================================="
echo "  Selective SFT - setup + train trong mot lenh"
echo "=============================================================="

# =============================================================================
# 1. Kiem tra so bo
# =============================================================================
[[ -f "$DATA" ]] || die "Khong thay training file: ${DATA}"
log "Training data: ${DATA} ($(wc -l < "$DATA" | tr -d ' ') dong)"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/    GPU /'
else
  warn "Khong thay nvidia-smi. Training can GPU NVIDIA - se that bai neu khong co."
fi

# =============================================================================
# 2. Dung moi truong
# =============================================================================
# Danh dau da cai xong de lan chay sau bo qua buoc pip install.
STAMP=".setup_done_${ENV_NAME}_v2"

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
    log "Cai dependency - lan dau se lau (torch + unsloth, vai GB)"
    run pip install --upgrade pip
    run pip install -r "${ROOT_DIR}/SelectiveSFT/requirements.txt"

    # unsloth thuong tu keo bitsandbytes, nhung khong file requirements nao pin no
    # trong khi train_mask.py dung optim="adamw_8bit" -> bao dam co mat.
    run pip install bitsandbytes

    # unsloth keo torchao ma khong ghim version -> pip lay ban moi nhat (0.18+),
    # ban nay build cho torch > 2.9 nen 'import unsloth' chet voi:
    #   cannot import name 'ScalingType' from 'torch.nn.functional'
    # requirements.txt ghim torch==2.9.0 nen phai ha torchao cho khop.
    log "Ghim torchao<0.18 cho khop torch 2.9.0"
    run pip install --upgrade --force-reinstall --no-cache-dir "torchao<0.18"

    # Kiem tra import ngay tai buoc setup thay vi de chet luc bat dau train.
    log "Kiem tra import unsloth"
    run python -c "import unsloth, torch, torchao; print(f'  torch={torch.__version__} torchao={torchao.__version__} OK')"

    [[ "$DRY_RUN" == "1" ]] || touch "$STAMP"
  fi
}

if [[ "$SKIP_SETUP" == "1" ]]; then
  log "Bo qua buoc dung moi truong (--skip-setup)"
else
  setup_env
fi

# =============================================================================
# 3. Train
# =============================================================================
mkdir -p "$LOG_DIR" "${ROOT_DIR}/SelectiveSFT/checkpoints"

export CUDA_VISIBLE_DEVICES="$GPU"
export REPORT_TO=none            # tat wandb
export WANDB_DISABLED=true WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

CKPT_DIR="SelectiveSFT/checkpoints/$(basename "$MODEL")_epoch${EPOCHS}_lr${LR}_len${MAX_SEQ_LENGTH}"

log "Bat dau training"
echo "    model      : ${MODEL}"
echo "    epochs / lr: ${EPOCHS} / ${LR}"
echo "    seq len    : ${MAX_SEQ_LENGTH}"
echo "    batch      : ${BATCH_SIZE} x ${GRAD_ACCUM} accum (effective $((BATCH_SIZE * GRAD_ACCUM)))"
echo "    checkpoint : ${CKPT_DIR}"
echo "    log        : ${LOG_DIR}/train.log"
echo

( cd "${ROOT_DIR}/SelectiveSFT" && run python -u train_mask.py \
    --model_name_or_path "${MODEL}" \
    --data_names "${ROOT_DIR}/${DATA}" \
    --epochs "${EPOCHS}" \
    --learning_rate "${LR}" \
    --max_seq_length "${MAX_SEQ_LENGTH}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --deepseek \
    --mask \
    --apply_all ) 2>&1 | tee "${LOG_DIR}/train.log"

log "Xong. Checkpoint: ${CKPT_DIR}"
echo "De eval: sua MODEL_PATH trong Eval/run_eval.sh tro vao checkpoint tren, roi 'cd Eval && bash run_eval.sh'"
