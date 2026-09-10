#!/usr/bin/env bash
#
# run_pipeline.sh - Chay toan bo pipeline cua Segment-Level Attribution for
# Selective Learning of Long Reasoning Traces, TRU phan Eval.
#
# Cac stage:
#   setup     Tao/kiem tra conda env, cai requirements + latex2sympy + unsloth deps
#   prep      Tai s1K-1.1 tu HuggingFace, doi ve format question/solution/answer
#   cot       (tuy chon) Tu sinh long-CoT traces bang vLLM
#   split     Chia solution thanh cac segment  (Attribution/segment_split.py)
#   ig        Tinh Integrated-Gradients attribution (Attribution/grad_analyze.py)
#   segments  Gop attribution -> chon important segments (get_important_segments.py)
#   train     Selective SFT co masking (SelectiveSFT/train_mask.py)
#
# Vi du:
#   bash run_pipeline.sh                        # chay prep,split,ig,segments,train
#   bash run_pipeline.sh --stages setup,split,ig,segments,train
#   bash run_pipeline.sh --stages train --epochs 5 --lr 1e-5
#   bash run_pipeline.sh --stages cot           # tu sinh CoT truoc khi split
#   DRY_RUN=1 bash run_pipeline.sh              # chi in lenh, khong chay
# -----------------------------------------------------------------------------

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# =============================================================================
# Cau hinh (co the override bang bien moi truong hoac co dong lenh)
# =============================================================================
STAGES="${STAGES:-prep,split,ig,segments,train}"

# --- Moi truong ---
CONDA_ENV="${CONDA_ENV:-selective_sft}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

# --- Model ---
ATTR_MODEL="${ATTR_MODEL:-Qwen/Qwen2.5-7B-Instruct}"   # model tinh IG
TRAIN_MODEL="${TRAIN_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # model SFT
COT_MODEL="${COT_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"     # model sinh CoT

# --- GPU ---
GPU_ATTR="${GPU_ATTR:-0,1}"
GPU_TRAIN="${GPU_TRAIN:-0}"
GPU_COT="${GPU_COT:-0,1,2,3}"

# --- Duong dan du lieu ---
HF_DATASET="${HF_DATASET:-simplescaling/s1K-1.1}"
RAW_DATA="${RAW_DATA:-data/s1k/train.jsonl}"
SEGMENT_FILE="${SEGMENT_FILE:-data/s1k/solution_segments.jsonl}"
IG_RAW_FILE="${IG_RAW_FILE:-Attribution/processed_data/s1k/solution_segments_attn.jsonl}"
IG_FILE="${IG_FILE:-Attribution/processed_data/s1k/IG.jsonl}"
TRAINING_FILE="${TRAINING_FILE:-data/s1k/solutions_selected.jsonl}"
SEGMENT_MODE="${SEGMENT_MODE:-paragraph}"

# --- Sieu tham so ---
IG_STEPS="${IG_STEPS:-50}"
IG_MAX_TOKENS="${IG_MAX_TOKENS:-0}"      # 0 = khong gioi han; >0 = mau dai hon thi gan diem 0
IG_GRAD_CKPT="${IG_GRAD_CKPT:-1}"        # 1 = bat gradient checkpointing (it VRAM hon nhieu)
EPOCHS="${EPOCHS:-3}"
LR="${LR:-5e-5}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-32768}"

# --- Tracking / logging ---
# "none" = tat hoan toan (mac dinh). Dat "wandb" + WANDB_PROJECT de bat lai.
REPORT_TO="${REPORT_TO:-none}"
WANDB_PROJECT="${WANDB_PROJECT:-selective_sft}"

# --- CoT generation (stage cot) ---
COT_OUTPUT_DIR="${COT_OUTPUT_DIR:-outputs/limo/r1_qwen_7b}"
COT_SEED="${COT_SEED:-0}"
COT_TEMPERATURE="${COT_TEMPERATURE:-0.6}"
COT_N_SAMPLING="${COT_N_SAMPLING:-32}"
COT_MAX_TOKENS="${COT_MAX_TOKENS:-32768}"

# --- Co khac ---
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"            # 1 = xoa file trung gian cu roi tinh lai
HF_OFFLINE="${HF_OFFLINE:-0}"  # 1 = export HF_HUB_OFFLINE=1 khi train
LOG_DIR="${LOG_DIR:-logs}"

# =============================================================================
# Parse tham so dong lenh
# =============================================================================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stages)          STAGES="$2"; shift 2 ;;
    --env)             CONDA_ENV="$2"; shift 2 ;;
    --attr-model)      ATTR_MODEL="$2"; shift 2 ;;
    --train-model)     TRAIN_MODEL="$2"; shift 2 ;;
    --cot-model)       COT_MODEL="$2"; shift 2 ;;
    --gpu-attr)        GPU_ATTR="$2"; shift 2 ;;
    --gpu-train)       GPU_TRAIN="$2"; shift 2 ;;
    --gpu-cot)         GPU_COT="$2"; shift 2 ;;
    --ig-steps)        IG_STEPS="$2"; shift 2 ;;
    --ig-max-tokens)   IG_MAX_TOKENS="$2"; shift 2 ;;
    --ig-no-grad-checkpoint) IG_GRAD_CKPT=0; shift ;;
    --epochs)          EPOCHS="$2"; shift 2 ;;
    --lr)              LR="$2"; shift 2 ;;
    --max-seq-length)  MAX_SEQ_LENGTH="$2"; shift 2 ;;
    --report-to)       REPORT_TO="$2"; shift 2 ;;
    --training-file)   TRAINING_FILE="$2"; shift 2 ;;
    --dataset)         HF_DATASET="$2"; shift 2 ;;
    --segment-mode)    SEGMENT_MODE="$2"; shift 2 ;;
    --force)           FORCE=1; shift ;;
    --offline)         HF_OFFLINE=1; shift ;;
    --dry-run)         DRY_RUN=1; shift ;;
    -h|--help)         sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Tham so khong hop le: $1 (xem --help)" >&2; exit 2 ;;
  esac
done

# =============================================================================
# Tien ich
# =============================================================================
log()  { printf '\n\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

banner() {
  echo "=============================================================="
  echo "  $*"
  echo "=============================================================="
}

run() {
  echo "+ $*"
  [[ "$DRY_RUN" == "1" ]] && return 0
  "$@"
}

has_stage() {
  [[ ",${STAGES}," == *",$1,"* ]]
}

need_file() {
  [[ "$DRY_RUN" == "1" ]] && return 0
  [[ -f "$1" ]] || die "Thieu file: $1 (chay stage truoc do da chua?)"
}

# Moi stage chay trong subshell, ma set -E cho subshell thua ke ERR trap, nen
# truoc day mot loi in ra ba dong [ERROR] chong len nhau che mat loi that.
# BASHPID chi co tu bash 4.0; bash 3.2 (mac dinh cua macOS) khong co, va set -u
# bien viec doc no thanh loi chi mang lam trap nuot mat thong bao. Dang :- nay
# thoai lui ve hanh vi cu tren bash 3.2 thay vi hong.
TOP_PID=$$
trap 'if [[ "${BASHPID:-$TOP_PID}" == "$TOP_PID" ]]; then die "Pipeline dung tai dong $LINENO"; fi' ERR

mkdir -p "$LOG_DIR"

# =============================================================================
# STAGE: setup
# =============================================================================
activate_env() {
  # Kich hoat conda env o cap top-level (moi stage chay trong subshell nen
  # activate ben trong stage se khong con hieu luc o stage sau).
  if ! command -v conda >/dev/null 2>&1; then
    [[ "${ALLOW_SYSTEM_PYTHON:-0}" == "1" ]] || die "Khong tim thay 'conda'.
      Dat ALLOW_SYSTEM_PYTHON=1 neu co y dung python hien tai: $(command -v python || echo none)"
    warn "Dung python hien tai: $(command -v python || echo none)"
    return 0
  fi
  local conda_base; conda_base="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${conda_base}/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    log "Kich hoat conda env '${CONDA_ENV}'"
    conda activate "$CONDA_ENV"
  else
    # Truoc day chi warn roi chay tiep bang python dang co - dan den loi kho
    # hieu tan sau trong thu vien (vi du numpy/sklearn ABI cua env he thong).
    [[ "${ALLOW_SYSTEM_PYTHON:-0}" == "1" ]] || die "Chua co conda env '${CONDA_ENV}'.
      Tao no:              bash run_pipeline.sh --stages setup
      Hoac dung env khac:  bash run_pipeline.sh --env <ten_env> ...
      Hoac co y dung python hien tai: ALLOW_SYSTEM_PYTHON=1 bash run_pipeline.sh ..."
    warn "Chua co env '${CONDA_ENV}', dung python hien tai (ALLOW_SYSTEM_PYTHON=1)."
  fi
}

stage_setup() {
  banner "STAGE 0/6 - Environment setup"

  if command -v conda >/dev/null 2>&1; then
    if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
      log "Conda env '${CONDA_ENV}' da ton tai."
    else
      log "Tao conda env '${CONDA_ENV}' (python ${PYTHON_VERSION})"
      run conda create -y -n "$CONDA_ENV" "python=${PYTHON_VERSION}"
    fi
  else
    warn "Khong tim thay 'conda'. Cai dat truc tiep vao python hien tai."
  fi

  activate_env

  log "Cai dependency chinh (requirements.txt)"
  run pip install -r "${ROOT_DIR}/requirements.txt"

  log "Cai latex2sympy (editable)"
  ( cd "${ROOT_DIR}/Eval/latex2sympy" && run pip install -e . )

  warn "Env '${CONDA_ENV}' chi dung cho prep/split/ig/segments (torch 2.7.1 + vLLM).
      KHONG cai SelectiveSFT/requirements.txt vao day: unsloth can torch 2.9.
      Stage 'train' tu goi train.sh, script do dung env rieng 'ssft_train'."

  log "Setup hoan tat."
}

# =============================================================================
# STAGE: cot (tuy chon) - tu sinh long-CoT traces cho LIMO
# =============================================================================
stage_cot() {
  banner "STAGE 6/6 (tuy chon) - Sinh CoT traces bang vLLM"

  export CUDA_VISIBLE_DEVICES="$GPU_COT"
  export TOKENIZERS_PARALLELISM=false

  ( cd "${ROOT_DIR}/Eval" && run python -u math_eval.py \
      --model_name_or_path "${COT_MODEL}" \
      --data_name "limo" \
      --output_dir "${COT_OUTPUT_DIR}" \
      --split "test" \
      --prompt_type "deepseek-longcot" \
      --num_test_sample -1 \
      --max_tokens_per_call "${COT_MAX_TOKENS}" \
      --seed "${COT_SEED}" \
      --temperature "${COT_TEMPERATURE}" \
      --n_sampling "${COT_N_SAMPLING}" \
      --top_p 1 \
      --start 0 \
      --end -1 \
      --use_vllm \
      --save_outputs \
      --apply_chat_template )

  log "CoT traces nam trong Eval/${COT_OUTPUT_DIR}."
  warn "Buoc chon trace ngan nhat va ghi ve ${LIMO_TEST} can lam thu cong truoc khi chay stage 'split'."
}

# =============================================================================
# STAGE: prep - tai dataset tu HuggingFace ve format cua pipeline
# =============================================================================
stage_prep() {
  banner "STAGE 1/6 - Tai ${HF_DATASET}"

  if [[ -s "${ROOT_DIR}/${RAW_DATA}" && "$FORCE" != "1" ]]; then
    log "${RAW_DATA} da co, bo qua (dung --force de tai lai)."
    return 0
  fi

  run python -u "${ROOT_DIR}/prepare_s1k.py" \
      --dataset "${HF_DATASET}" \
      --output_data_file "${ROOT_DIR}/${RAW_DATA}"

  log "Da tao ${RAW_DATA}"
}

# =============================================================================
# STAGE: split - chia solution thanh segment
# =============================================================================
stage_split() {
  banner "STAGE 2/6 - Chia solution thanh segments (mode=${SEGMENT_MODE})"

  need_file "${ROOT_DIR}/${RAW_DATA}"
  mkdir -p "$(dirname "${ROOT_DIR}/${SEGMENT_FILE}")"

  ( cd "${ROOT_DIR}/Attribution" && run python -u segment_split.py \
      --input_data_file "${ROOT_DIR}/${RAW_DATA}" \
      --output_data_file "${ROOT_DIR}/${SEGMENT_FILE}" \
      --tokenizer "${ATTR_MODEL}" \
      --segment_mode "${SEGMENT_MODE}" )

  [[ "$DRY_RUN" == "1" ]] || need_file "${ROOT_DIR}/${SEGMENT_FILE}"
  log "Da tao ${SEGMENT_FILE}"
}

# =============================================================================
# STAGE: ig - tinh Integrated Gradients attribution
# =============================================================================
stage_ig() {
  banner "STAGE 3/6 - Tinh token attribution (Integrated Gradients)"

  need_file "${ROOT_DIR}/${SEGMENT_FILE}"
  mkdir -p "$(dirname "${ROOT_DIR}/${IG_RAW_FILE}")" "$(dirname "${ROOT_DIR}/${IG_FILE}")"

  # grad_analyze.py mo output_data_file o che do append -> phai don file cu,
  # neu khong ket qua se bi nhan doi.
  if [[ -s "${ROOT_DIR}/${IG_RAW_FILE}" ]]; then
    if [[ "$FORCE" == "1" ]]; then
      log "Xoa file attribution cu: ${IG_RAW_FILE}"
      run rm -f "${ROOT_DIR}/${IG_RAW_FILE}"
    else
      die "${IG_RAW_FILE} da ton tai va script ghi o che do append.
      Chay lai voi --force de xoa va tinh lai, hoac doi IG_RAW_FILE."
    fi
  fi

  export CUDA_VISIBLE_DEVICES="$GPU_ATTR"

  IG_ARGS=()
  [[ "$IG_MAX_TOKENS" != "0" ]] && IG_ARGS+=(--max_input_tokens "$IG_MAX_TOKENS")
  [[ "$IG_GRAD_CKPT" == "0" ]]  && IG_ARGS+=(--no_gradient_checkpointing)

  ( cd "${ROOT_DIR}/Attribution" && run python -u grad_analyze.py \
      --model_name "${ATTR_MODEL}" \
      --input_data "${ROOT_DIR}/${SEGMENT_FILE}" \
      --output_data_file "${ROOT_DIR}/${IG_RAW_FILE}" \
      --output_ig_file "${ROOT_DIR}/${IG_FILE}" \
      --ig_steps "${IG_STEPS}" \
      ${IG_ARGS[@]+"${IG_ARGS[@]}"} )

  log "Da tao ${IG_FILE}"
}

# =============================================================================
# STAGE: segments - gop attribution, chon important segments
# =============================================================================
stage_segments() {
  banner "STAGE 4/6 - Xac dinh important segments"

  need_file "${ROOT_DIR}/${SEGMENT_FILE}"
  need_file "${ROOT_DIR}/${IG_FILE}"
  mkdir -p "$(dirname "${ROOT_DIR}/${TRAINING_FILE}")"

  if [[ -s "${ROOT_DIR}/${TRAINING_FILE}" && "$FORCE" == "1" ]]; then
    log "Xoa training file cu: ${TRAINING_FILE}"
    run rm -f "${ROOT_DIR}/${TRAINING_FILE}"
  fi

  ( cd "${ROOT_DIR}/Attribution" && run python -u get_important_segments.py \
      --input_data_file "${ROOT_DIR}/${SEGMENT_FILE}" \
      --IG_score_data_file "${ROOT_DIR}/${IG_FILE}" \
      --output_data_file "${ROOT_DIR}/${TRAINING_FILE}" )

  log "Da tao training file: ${TRAINING_FILE}"
}

# =============================================================================
# STAGE: train - Selective SFT voi masking
# =============================================================================
stage_train() {
  banner "STAGE 5/6 - Selective SFT"

  need_file "${ROOT_DIR}/${TRAINING_FILE}"
  mkdir -p "${ROOT_DIR}/SelectiveSFT/checkpoints"

  export CUDA_VISIBLE_DEVICES="$GPU_TRAIN"
  [[ "$HF_OFFLINE" == "1" ]] && export HF_HUB_OFFLINE=1

  export REPORT_TO="$REPORT_TO"
  if [[ "$REPORT_TO" == "wandb" ]]; then
    export WANDB_PROJECT="$WANDB_PROJECT"
    log "Tracking: wandb (project=${WANDB_PROJECT})"
  else
    # WANDB_DISABLED da deprecated (spam canh bao moi lan khoi tao Trainer),
    # report_to=none la du.
    export WANDB_MODE=disabled
    log "Tracking: tat (report_to=${REPORT_TO}). Loss van in ra stdout + ${LOG_DIR}/train.log"
  fi

  # Goi train.sh de chi co MOT noi dinh nghia hyperparameter mac dinh.
  # Khong truyen --skip-setup: train.sh phai tu kich hoat env ssft_train cua no,
  # khong dung nho env dang active cua pipeline (torch/vllm khac phien ban).
  ( run bash "${ROOT_DIR}/train.sh" \
      --model "${TRAIN_MODEL}" \
      --data "${TRAINING_FILE}" \
      --gpu "${GPU_TRAIN}" \
      --epochs "${EPOCHS}" \
      --lr "${LR}" \
      --max-seq-length "${MAX_SEQ_LENGTH}" \
      --segment-mode "${SEGMENT_MODE}" )

  log "Checkpoint nam trong SelectiveSFT/checkpoints/"
}

# =============================================================================
# Main
# =============================================================================
banner "Segment-Selective SFT pipeline (khong bao gom Eval)"
cat <<EOF
  Stages       : ${STAGES}
  Dataset      : ${HF_DATASET}  (segment mode: ${SEGMENT_MODE})
  Attr model   : ${ATTR_MODEL}   (GPU ${GPU_ATTR})
  Train model  : ${TRAIN_MODEL}  (GPU ${GPU_TRAIN})
  IG steps     : ${IG_STEPS}
  Epochs / LR  : ${EPOCHS} / ${LR}
  Training file: ${TRAINING_FILE}
  Tracking     : ${REPORT_TO}
  Dry run      : ${DRY_RUN}
EOF

START_TS=$SECONDS

# Kich hoat env mot lan o top-level, tru khi dang o stage setup (setup tu activate).
if ! has_stage setup; then
  activate_env
fi

for stage in setup prep cot split ig segments train; do
  if has_stage "$stage"; then
    "stage_${stage}" 2>&1 | tee "${LOG_DIR}/${stage}.log"
  fi
done

log "Hoan tat sau $(( (SECONDS - START_TS) / 60 )) phut. Log: ${LOG_DIR}/"
echo "De danh gia model, chay rieng: bash eval.sh  (LoRA thi merge truoc, xem cuoi train.sh)"
