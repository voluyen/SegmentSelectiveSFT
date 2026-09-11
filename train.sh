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
# Mac dinh (LoRA tren Qwen2.5-7B-Instruct, du lieu s1K-1.1):
#   r=16 alpha=16 dropout=0.05 tren q/k/v/o/gate/up/down_proj
#   lr 5e-5, 3 epoch, seq 32768, batch 1 x accum 32 = 32 mau/step
#   AdamW betas (0.9, 0.999) eps 1e-8 wd 0.0, cosine + warmup_ratio 0.1
#
# Tuy chon:
#   bash train.sh --epochs 5 --lr 1e-5
#   bash train.sh --gpu 1                  # dung GPU khac
#   bash train.sh --batch-size 2 --grad-accum 16   # van la 32 mau/step
#   bash train.sh --lora-r 32 --lora-alpha 64 --lora-dropout 0
#   bash train.sh --4bit                   # QLoRA: it VRAM hon, cham hon mot chut
#   bash train.sh --target-modules "q_proj,v_proj"
#   bash train.sh --full-finetune          # bo LoRA, finetune toan bo (rat ton VRAM)
#   bash train.sh --no-grad-checkpoint     # nhanh hon, ton VRAM hon
#   bash train.sh --segment-mode cue       # chia segment kieu paper thay vi theo "\n\n"
#   bash train.sh --full-sft               # baseline: SFT tren TOAN BO long CoT (khong mask)
#   bash train.sh --optim adamw_8bit       # tiet kiem VRAM optimizer state
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
# Mac dinh KHONG tu dung moi truong: dung thang python dang active. Nhieu cloud
# studio da co san env day du va khong co lenh 'conda'. Dat USE_CONDA=1 (hoac
# --use-conda) de quay lai kieu tao conda env / venv rieng.
USE_CONDA="${USE_CONDA:-0}"
ENV_NAME="${ENV_NAME:-ssft_train}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DATA="${DATA:-data/s1k/solutions_selected.jsonl}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-5e-5}"
LOG_DIR="${LOG_DIR:-logs}"

# 32768 = dung tran max_position_embeddings cua Qwen2.5-7B-Instruct.
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-32768}"
# Micro-batch 1 = khong co padding nao (khong co mau khac de pad theo).
# Effective batch = 1 x 32 = 32 mau/step.
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
GROUP_BY_LENGTH="${GROUP_BY_LENGTH:-0}"   # 1 = gom mau cung do dai, bo padding thua
# Seq 32768 tren 7B thi gradient checkpointing la bat buoc -> mac dinh BAT.
NO_GRAD_CKPT="${NO_GRAD_CKPT:-0}"         # 1 = tat (ton VRAM, nhanh hon)

# --- LoRA ---
USE_LORA="${USE_LORA:-1}"                 # 0 = full finetuning
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LOAD_4BIT="${LOAD_4BIT:-0}"               # 1 = QLoRA, weight goc nap o 4-bit (it VRAM hon nhieu)
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

# --- Optimizer / scheduler ---
OPTIM="${OPTIM:-adamw_torch}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"
ADAM_EPSILON="${ADAM_EPSILON:-1e-8}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"

# --- Segmentation / prompt ---
SEGMENT_MODE="${SEGMENT_MODE:-paragraph}"  # paragraph = chia theo "\n\n"
THINK_PREFIX="${THINK_PREFIX:-none}"       # Qwen khong co token <think>; xem train_mask.py
# 0 = selective SFT (chi hoc segment duoc chon) - mac dinh, dung cua paper.
# 1 = long-CoT SFT thuong: hoc toan bo response. Checkpoint/log rieng,
#     khong de len ban selective.
FULL_SFT="${FULL_SFT:-0}"

SKIP_SETUP=0
REINSTALL=0
DRY_RUN="${DRY_RUN:-0}"
HF_OFFLINE="${HF_OFFLINE:-0}"   # 1 = cam moi ket noi ra HuggingFace Hub

# =============================================================================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)             ENV_NAME="$2"; USE_CONDA=1; shift 2 ;;
    --use-conda)       USE_CONDA=1; shift ;;
    --no-conda)        USE_CONDA=0; shift ;;
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
    --grad-checkpoint) NO_GRAD_CKPT=0; shift ;;
    --lora)            USE_LORA=1; shift ;;
    --full-finetune)   USE_LORA=0; shift ;;
    --lora-r)          LORA_R="$2"; shift 2 ;;
    --lora-alpha)      LORA_ALPHA="$2"; shift 2 ;;
    --lora-dropout)    LORA_DROPOUT="$2"; shift 2 ;;
    --4bit)            LOAD_4BIT=1; shift ;;
    --target-modules)  TARGET_MODULES="$2"; shift 2 ;;
    --optim)           OPTIM="$2"; shift 2 ;;
    --weight-decay)    WEIGHT_DECAY="$2"; shift 2 ;;
    --warmup-ratio)    WARMUP_RATIO="$2"; shift 2 ;;
    --lr-scheduler)    LR_SCHEDULER="$2"; shift 2 ;;
    --segment-mode)    SEGMENT_MODE="$2"; shift 2 ;;
    --think-prefix)    THINK_PREFIX="$2"; shift 2 ;;
    --full-sft)        FULL_SFT=1; shift ;;
    --selective)       FULL_SFT=0; shift ;;
    --skip-setup)      SKIP_SETUP=1; shift ;;
    --reinstall)       REINSTALL=1; shift ;;
    --offline)         HF_OFFLINE=1; shift ;;
    --dry-run)         DRY_RUN=1; shift ;;
    -h|--help)         sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
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
  if [[ "$USE_CONDA" != "1" ]]; then
    log "Dung python dang active: $(command -v python || command -v python3 || echo none)"
    if [[ "$REINSTALL" == "1" ]]; then
      log "Cai lai dependency vao chinh python dang active (--reinstall)"
      run pip install -r "${ROOT_DIR}/requirements-sft.txt"
    else
      log "Gia dinh moi truong da du goi. Them --reinstall de cai lai, hoac --use-conda de dung env rieng."
    fi
    return 0
  fi

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
if [[ "$HF_OFFLINE" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  log "Che do offline: cam ket noi ra HuggingFace Hub"
fi

export TOKENIZERS_PARALLELISM=false

EXTRA_ARGS=()
[[ "$GROUP_BY_LENGTH" == "1" ]] && EXTRA_ARGS+=(--group_by_length)
[[ "$NO_GRAD_CKPT"    == "1" ]] && EXTRA_ARGS+=(--no_gradient_checkpointing)

# Khong co --mask thi train_mask.py supervise toan bo response = long-CoT SFT
# thuong. Cac suffix duoi day do train_mask.py tu ghep vao output_dir; eval.sh
# dung lai dung quy uoc nay de tim checkpoint.
CKPT_SUFFIX=""
if [[ "$FULL_SFT" == "1" ]]; then
  MASK_ARGS=()
  MODE_NAME="full-CoT SFT (baseline, khong mask)"
  CKPT_SUFFIX="${CKPT_SUFFIX}_fullsft"
  LOG_NAME="train_fullsft"
else
  MASK_ARGS=(--mask --apply_all)
  MODE_NAME="selective SFT (chi segment duoc chon)"
  LOG_NAME="train"
fi

if [[ "$USE_LORA" == "1" ]]; then
  TUNE_ARGS=(
    --lora_r "${LORA_R}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_dropout "${LORA_DROPOUT}"
    --target_modules "${TARGET_MODULES}"
  )
  [[ "$LOAD_4BIT" == "1" ]] && TUNE_ARGS+=(--load_in_4bit)
  TUNE_NAME="LoRA r=${LORA_R} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT}$([[ "$LOAD_4BIT" == 1 ]] && echo ' (4-bit/QLoRA)')"
  CKPT_SUFFIX="${CKPT_SUFFIX}_lora"
  LOG_NAME="${LOG_NAME}_lora"
else
  TUNE_ARGS=(--full_finetune)
  TUNE_NAME="full finetuning"
fi
LOG_FILE="${LOG_DIR}/${LOG_NAME}.log"

# Ten thu muc do bash quyet dinh roi truyen thang bang --output_dir. Truoc day
# train_mask.py tu ghep ten bang f-string tu float: LR=1e-4 thanh "_lr0.0001",
# 5e-5 thanh "_lr5e-05", khong bao gio khop chuoi ma bash/eval.sh dung.
CKPT_DIR="SelectiveSFT/checkpoints/$(basename "$MODEL")_epoch${EPOCHS}_lr${LR}_len${MAX_SEQ_LENGTH}${CKPT_SUFFIX}"

log "Bat dau training"
echo "    mode       : ${MODE_NAME}"
echo "    tuning     : ${TUNE_NAME}"
[[ "$USE_LORA" == "1" ]] && echo "    modules    : ${TARGET_MODULES}"
echo "    model      : ${MODEL}"
echo "    epochs / lr: ${EPOCHS} / ${LR}"
echo "    seq len    : ${MAX_SEQ_LENGTH}"
echo "    batch      : ${BATCH_SIZE} x ${GRAD_ACCUM} accum (effective $((BATCH_SIZE * GRAD_ACCUM)))"
echo "    optim      : ${OPTIM} betas=(${ADAM_BETA1}, ${ADAM_BETA2}) eps=${ADAM_EPSILON} wd=${WEIGHT_DECAY}"
echo "    scheduler  : ${LR_SCHEDULER} warmup_ratio=${WARMUP_RATIO}"
echo "    segment    : ${SEGMENT_MODE} (think_prefix=${THINK_PREFIX})"
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
    --output_dir "${ROOT_DIR}/${CKPT_DIR}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --optim "${OPTIM}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --adam_beta1 "${ADAM_BETA1}" \
    --adam_beta2 "${ADAM_BETA2}" \
    --adam_epsilon "${ADAM_EPSILON}" \
    --lr_scheduler_type "${LR_SCHEDULER}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --segment_mode "${SEGMENT_MODE}" \
    --think_prefix "${THINK_PREFIX}" \
    ${TUNE_ARGS[@]+"${TUNE_ARGS[@]}"} \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    ${MASK_ARGS[@]+"${MASK_ARGS[@]}"} ) 2>&1 | tee "${LOG_FILE}"

log "Xong. Checkpoint: ${CKPT_DIR}"
if [[ "$USE_LORA" == "1" ]]; then
  echo "Checkpoint la adapter LoRA. Gop vao weight goc truoc khi eval:"
  echo "    cd SelectiveSFT && python merge_lora.py --adapter <ckpt_dir>/checkpoint-<step>"
  echo "    bash eval.sh --model <ckpt_dir>/checkpoint-<step>-merged"
else
  echo "De eval: bash eval.sh --model ${CKPT_DIR}/checkpoint-<step>"
fi
