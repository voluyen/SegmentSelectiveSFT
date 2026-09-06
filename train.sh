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
#   bash train.sh --grad-checkpoint                # bat lai gradient checkpointing neu OOM
#   bash train.sh --group-by-length                # gom mau cung do dai (khong can khi batch=1)
#   bash train.sh --full-sft               # baseline: SFT tren TOAN BO long CoT (khong mask)
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
LR="${LR:-1e-4}"
LOG_DIR="${LOG_DIR:-logs}"

MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-16384}"
# Micro-batch 1 = khong co padding nao (khong co mau khac de pad theo).
# Effective batch van la 1 x 32 = 32, gradient khong doi.
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
GROUP_BY_LENGTH="${GROUP_BY_LENGTH:-0}"   # 1 = gom mau cung do dai, bo padding thua
NO_GRAD_CKPT="${NO_GRAD_CKPT:-1}"         # 1 = tat gradient checkpointing (ton VRAM, nhanh hon)
                                          #     OOM thi bat lai bang --grad-checkpoint
# 0 = selective SFT (chi hoc segment duoc chon) - mac dinh, dung cua paper.
# 1 = long-CoT SFT thuong: hoc toan bo response. Checkpoint/log rieng,
#     khong de len ban selective.
FULL_SFT="${FULL_SFT:-0}"

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
    --group-by-length) GROUP_BY_LENGTH=1; shift ;;
    --no-grad-checkpoint) NO_GRAD_CKPT=1; shift ;;
    --grad-checkpoint) NO_GRAD_CKPT=0; shift ;;   # bat lai neu OOM
    --full-sft)        FULL_SFT=1; shift ;;
    --selective)       FULL_SFT=0; shift ;;
    --skip-setup)      SKIP_SETUP=1; shift ;;
    --reinstall)       REINSTALL=1; shift ;;
    --dry-run)         DRY_RUN=1; shift ;;
    -h|--help)         sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0 ;;
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
    # requirements-sft.txt = SelectiveSFT/requirements.txt + torchao<0.18
    # + bitsandbytes. KHONG dung requirements.txt o thu muc goc: do la moi
    # truong vLLM cho eval/attribution va se xung dot torch/transformers.
    log "Cai dependency - lan dau se lau (torch + unsloth, vai GB)"
    run pip install --upgrade pip
    run pip install -r "${ROOT_DIR}/requirements-sft.txt"

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
export WANDB_MODE=disabled     # khong dung WANDB_DISABLED: da deprecated, gay spam canh bao
export TOKENIZERS_PARALLELISM=false

EXTRA_ARGS=()
[[ "$GROUP_BY_LENGTH" == "1" ]] && EXTRA_ARGS+=(--group_by_length)
[[ "$NO_GRAD_CKPT"    == "1" ]] && EXTRA_ARGS+=(--no_gradient_checkpointing)

# Khong co --mask thi train_mask.py supervise toan bo response = long-CoT SFT
# thuong. Suffix _fullsft do train_mask.py tu them vao output_dir.
if [[ "$FULL_SFT" == "1" ]]; then
  MASK_ARGS=()
  MODE_NAME="full-CoT SFT (baseline, khong mask)"
  CKPT_SUFFIX="_fullsft"
  LOG_FILE="${LOG_DIR}/train_fullsft.log"
else
  MASK_ARGS=(--mask --apply_all)
  MODE_NAME="selective SFT (chi segment duoc chon)"
  CKPT_SUFFIX=""
  LOG_FILE="${LOG_DIR}/train.log"
fi

CKPT_DIR="SelectiveSFT/checkpoints/$(basename "$MODEL")_epoch${EPOCHS}_lr${LR}_len${MAX_SEQ_LENGTH}${CKPT_SUFFIX}"

log "Bat dau training"
echo "    mode       : ${MODE_NAME}"
echo "    model      : ${MODEL}"
echo "    epochs / lr: ${EPOCHS} / ${LR}"
echo "    seq len    : ${MAX_SEQ_LENGTH}"
echo "    batch      : ${BATCH_SIZE} x ${GRAD_ACCUM} accum (effective $((BATCH_SIZE * GRAD_ACCUM)))"
echo "    group_by_len : $([[ "$GROUP_BY_LENGTH" == 1 ]] && echo on || echo off)"
echo "    grad_ckpt    : $([[ "$NO_GRAD_CKPT" == 1 ]] && echo off || echo on)"
echo "    checkpoint : ${CKPT_DIR}"
echo "    log        : ${LOG_FILE}"
echo

( cd "${ROOT_DIR}/SelectiveSFT" && run python -u train_mask.py \
    --model_name_or_path "${MODEL}" \
    --data_names "${ROOT_DIR}/${DATA}" \
    --epochs "${EPOCHS}" \
    --learning_rate "${LR}" \
    --max_seq_length "${MAX_SEQ_LENGTH}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    --deepseek \
    ${MASK_ARGS[@]+"${MASK_ARGS[@]}"} ) 2>&1 | tee "${LOG_FILE}"

log "Xong. Checkpoint: ${CKPT_DIR}"
echo "De eval: sua MODEL_PATH trong Eval/run_eval.sh tro vao checkpoint tren, roi 'cd Eval && bash run_eval.sh'"
