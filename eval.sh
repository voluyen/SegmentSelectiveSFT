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
#   bash eval.sh --full-finetune           # checkpoint train khong dung LoRA
#   bash eval.sh --full-sft                # eval checkpoint baseline full-CoT
#   bash eval.sh --model /duong/dan/checkpoint-250
#   bash eval.sh --model /duong/dan/checkpoint-250 --tag sel_ep5
#   bash eval.sh --quick                   # kiem nhanh: math500, 100 cau, 1 mau/cau
#   bash eval.sh --tasks "aime24 math500"  # chi vai task
#   bash eval.sh --n-sampling 1            # 1 mau/cau cho nhanh (mac dinh 32/6)
#   bash eval.sh --num-test-sample 100     # chi lay 100 cau dau moi task
#   bash eval.sh --gpu 0,1,2,3 --data-parallel   # 1 task/GPU chay song song
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
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
GPU="${GPU:-0}"
LOG_DIR="${LOG_DIR:-logs}"

# Phai khop voi cach train.sh dat ten thu muc checkpoint.
EPOCHS="${EPOCHS:-3}"
LR="${LR:-5e-5}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-32768}"
USE_LORA="${USE_LORA:-1}"   # train.sh mac dinh LoRA -> ten thu muc co hau to _lora

# "task so_mau_moi_cau" - lay tu Eval/run_eval.sh goc cua paper.
TASKS_DEFAULT="aime24:32 amc23:32 math500:6 minerva:6 gpqa:6 olympiad:6"
TASKS="${TASKS:-}"                  # rong = dung TASKS_DEFAULT
N_SAMPLING="${N_SAMPLING:-}"        # rong = dung so mau rieng cua tung task
NUM_TEST_SAMPLE="${NUM_TEST_SAMPLE:-}"   # rong = -1 = ca test set
QUICK=0

SEED="${SEED:-0}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1}"
PROMPT_TYPE="${PROMPT_TYPE:-deepseek-longcot}"

DATA_PARALLEL=0                          # 1 = chia task ra tung GPU chay song song
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"     # KV cache lon hon = nhieu seq dong thoi hon
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"       # rong = de vLLM tu lay tu config model
PREFIX_CACHING="${PREFIX_CACHING:-1}"    # n mau/cau dung chung prompt -> bo prefill lap
LOGPROBS="${LOGPROBS:-0}"                # repo khong dung logprobs, bat chi ton them

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
    --lora)            USE_LORA=1; shift ;;
    --full-finetune)   USE_LORA=0; shift ;;
    --epochs)          EPOCHS="$2"; shift 2 ;;
    --lr)              LR="$2"; shift 2 ;;
    --max-seq-length)  MAX_SEQ_LENGTH="$2"; shift 2 ;;
    --tasks)           TASKS="$2"; shift 2 ;;
    --n-sampling)      N_SAMPLING="$2"; shift 2 ;;
    --num-test-sample) NUM_TEST_SAMPLE="$2"; shift 2 ;;
    --quick)           QUICK=1; shift ;;
    --gpu)             GPU="$2"; shift 2 ;;
    --data-parallel|--dp) DATA_PARALLEL=1; shift ;;
    --gpu-mem-util)    GPU_MEM_UTIL="$2"; shift 2 ;;
    --max-model-len)   MAX_MODEL_LEN="$2"; shift 2 ;;
    --no-prefix-caching) PREFIX_CACHING=0; shift ;;
    --logprobs)        LOGPROBS=1; shift ;;
    --seed)            SEED="$2"; shift 2 ;;
    --max-tokens)      MAX_TOKENS="$2"; shift 2 ;;
    --temperature)     TEMPERATURE="$2"; shift 2 ;;
    --output-root)     OUTPUT_ROOT="$2"; shift 2 ;;
    --overwrite)       OVERWRITE=1; shift ;;
    --skip-setup)      SKIP_SETUP=1; shift ;;
    --reinstall)       REINSTALL=1; shift ;;
    --dry-run)         DRY_RUN=1; shift ;;
    -h|--help)         sed -n '2,31p' "${BASH_SOURCE[0]}"; exit 0 ;;
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

# train.sh ghep hau to theo thu tu: [_fullsft][_lora]
LORA_SUFFIX=""
[[ "$USE_LORA" == "1" ]] && LORA_SUFFIX="_lora"
CKPT_BASE="${ROOT_DIR}/SelectiveSFT/checkpoints/$(basename "$BASE_MODEL")_epoch${EPOCHS}_lr${LR}_len${MAX_SEQ_LENGTH}"

case "$WHICH" in
  base)
    MODEL="$BASE_MODEL"; DEFAULT_TAG="base" ;;
  selective)
    MODEL="$(latest_checkpoint "${CKPT_BASE}${LORA_SUFFIX}")"
    [[ -n "$MODEL" ]] || die "Khong thay checkpoint trong ${CKPT_BASE}${LORA_SUFFIX} - train truoc, hoac dung --model / --base
      (them --full-finetune neu ban train khong dung LoRA)"
    DEFAULT_TAG="selective" ;;
  fullsft)
    MODEL="$(latest_checkpoint "${CKPT_BASE}_fullsft${LORA_SUFFIX}")"
    [[ -n "$MODEL" ]] || die "Khong thay checkpoint trong ${CKPT_BASE}_fullsft${LORA_SUFFIX} - chay 'bash train.sh --full-sft' truoc"
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
[[ "$QUICK" == "1" && "$RUN_TAG" != *_quick ]] && RUN_TAG="${RUN_TAG}_quick"

# Duong dan phai tuyet doi vi lat nua se cd sang Eval/.
[[ -d "$MODEL" ]] && MODEL="$(cd "$MODEL" && pwd)"

# vLLM nap model day du, khong hieu adapter LoRA. Neu tro vao thu muc adapter
# thi dung lai va chi ra lenh merge, thay vi de vLLM bao loi kho hieu.
if [[ -d "$MODEL" && -f "$MODEL/adapter_config.json" && ! -f "$MODEL/config.json" ]]; then
  if [[ -d "${MODEL}-merged" ]]; then
    log "Phat hien adapter LoRA, dung ban da merge: ${MODEL}-merged"
    MODEL="${MODEL}-merged"
  else
    die "${MODEL} la adapter LoRA, vLLM khong nap truc tiep duoc. Merge truoc (trong env train):
      conda activate ssft_train
      cd SelectiveSFT && python merge_lora.py --adapter '${MODEL}'
    roi chay lai lenh nay."
  fi
fi
[[ -n "$OUTPUT_ROOT" ]] || OUTPUT_ROOT="outputs_${RUN_TAG}"

# --quick chi dat mac dinh, khong de len cai ban da go tay.
# Rut so cau + so mau chu KHONG rut --max-tokens: model reasoning bi cat ngan
# se mat dap an va accuracy tut gia tao.
if [[ "$QUICK" == "1" ]]; then
  [[ -n "$TASKS" ]]           || TASKS="math500"
  [[ -n "$N_SAMPLING" ]]      || N_SAMPLING=1
  [[ -n "$NUM_TEST_SAMPLE" ]] || NUM_TEST_SAMPLE=100
fi
[[ -n "$TASKS" ]]           || TASKS="$TASKS_DEFAULT"
[[ -n "$NUM_TEST_SAMPLE" ]] || NUM_TEST_SAMPLE=-1

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
    warn "Khong co conda, chuyen sang venv voi python he thong."
    # Cac pin trong requirements.txt (vllm 0.10, triton 3.3.1, xformers 0.0.31)
    # chi co wheel den cp312. Python moi hon se phai build tu source va hong.
    local pyver; pyver="$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
    case "$pyver" in
      3.9|3.10|3.11|3.12) ;;
      *) warn "Python he thong la ${pyver} - requirements.txt chi co wheel den 3.12. Nen cai conda hoac dung python 3.11." ;;
    esac
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

EXTRA_ARGS=(--gpu_memory_utilization "$GPU_MEM_UTIL")
[[ "$OVERWRITE" == "1" ]]       && EXTRA_ARGS+=(--overwrite)
[[ "$PREFIX_CACHING" == "1" ]]  && EXTRA_ARGS+=(--enable_prefix_caching)
[[ "$LOGPROBS" == "1" ]]        && EXTRA_ARGS+=(--return_logprobs)
[[ -n "$MAX_MODEL_LEN" ]]       && EXTRA_ARGS+=(--max_model_len "$MAX_MODEL_LEN")

IFS=',' read -ra GPU_LIST <<< "$GPU"
NGPU=${#GPU_LIST[@]}
if [[ "$DATA_PARALLEL" == "1" && "$NGPU" -lt 2 ]]; then
  warn "--data-parallel can tu 2 GPU tro len, bo qua."
  DATA_PARALLEL=0
fi

LOG_FILE="${LOG_DIR}/eval_${RUN_TAG}.log"

# "aime24:32" hoac chi "aime24" (lay so mau mac dinh cua task do) -> "task:n"
resolve_tasks() {
  local spec task n out=""
  for spec in $TASKS; do
    task="${spec%%:*}"; n="${spec#*:}"
    [[ "$n" == "$task" ]] && n=""
    if [[ -n "$N_SAMPLING" ]]; then
      n="$N_SAMPLING"
    elif [[ -z "$n" ]]; then
      case " $TASKS_DEFAULT " in
        *" ${task}:"*) n="$(echo "$TASKS_DEFAULT" | tr ' ' '\n' | grep "^${task}:" | cut -d: -f2)" ;;
        *) n=1 ;;
      esac
    fi
    out="${out}${task}:${n} "
  done
  echo "$out"
}
RESOLVED_TASKS="$(resolve_tasks)"

log "Bat dau eval"
echo "    model      : ${MODEL}"
echo "    tasks      : ${RESOLVED_TASKS}"
if [[ "$DATA_PARALLEL" == "1" ]]; then
  echo "    gpu        : ${GPU} (data parallel, ${NGPU} tien trinh, moi tien trinh 1 GPU)"
else
  echo "    gpu        : ${GPU} (tensor_parallel_size=${NGPU})"
fi
echo "    sampling   : t=${TEMPERATURE} top_p=${TOP_P} seed=${SEED} max_tokens=${MAX_TOKENS}"
echo "    so cau     : $([[ "$NUM_TEST_SAMPLE" == "-1" ]] && echo "ca test set" || echo "${NUM_TEST_SAMPLE} cau dau")"
echo "    vllm       : gpu_mem=${GPU_MEM_UTIL} prefix_cache=$([[ "$PREFIX_CACHING" == 1 ]] && echo on || echo off) logprobs=$([[ "$LOGPROBS" == 1 ]] && echo on || echo off) max_model_len=${MAX_MODEL_LEN:-auto}"
echo "    output     : Eval/${OUTPUT_ROOT}/<task>/${RUN_TAG}"
echo "    log        : ${LOG_FILE%.log}*.log"
echo

# Chay tuan tu danh sach "task:n" truyen vao, trong CUDA_VISIBLE_DEVICES hien tai.
run_tasks() {
  local spec task n
  for spec in $1; do
    task="${spec%%:*}"; n="${spec##*:}"
    echo "=============================================="
    echo "Task: ${task}  |  ${n} mau/cau  |  GPU ${CUDA_VISIBLE_DEVICES}"
    echo "=============================================="
    run python -u math_eval.py \
      --model_name_or_path "${MODEL}" \
      --data_name "${task}" \
      --output_dir "${OUTPUT_ROOT}/${task}/${RUN_TAG}" \
      --split "test" \
      --prompt_type "${PROMPT_TYPE}" \
      --num_test_sample "${NUM_TEST_SAMPLE}" \
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

export TOKENIZERS_PARALLELISM=false

if [[ "$DATA_PARALLEL" == "1" ]]; then
  # Model 1.5B thua suc nam gon 1 GPU, nen chia task ra chay song song lai
  # nhanh hon tensor parallel (khong ton chi phi giao tiep giua cac GPU).
  # Chia kieu longest-processing-time: task nang nhat vao GPU dang ranh nhat.
  ASSIGN="$(python3 - "$ROOT_DIR/data" "$NGPU" $RESOLVED_TASKS <<'PYSPLIT'
import os, sys
data_dir, ngpu = sys.argv[1], int(sys.argv[2])
items = []
for spec in sys.argv[3:]:
    task, n = spec.rsplit(":", 1)
    f = os.path.join(data_dir, task, "test.jsonl")
    q = sum(1 for _ in open(f)) if os.path.exists(f) else 100
    items.append((q * int(n), spec))
items.sort(reverse=True)
buckets = [[0, []] for _ in range(ngpu)]
for cost, spec in items:
    b = min(buckets, key=lambda x: x[0])
    b[0] += cost
    b[1].append(spec)
for load, specs in buckets:
    print(" ".join(specs))
PYSPLIT
)"

  PIDS=(); IDX=0
  while IFS= read -r line; do
    g="${GPU_LIST[$IDX]}"; IDX=$((IDX + 1))
    if [[ -z "$line" ]]; then
      warn "GPU ${g}: khong duoc chia task nao"
      continue
    fi
    glog="${LOG_FILE%.log}_gpu${g}.log"
    echo "  GPU ${g} <- ${line}   (log: ${glog})"
    ( export CUDA_VISIBLE_DEVICES="$g"
      cd "${ROOT_DIR}/Eval" && run_tasks "$line" ) > "$glog" 2>&1 &
    PIDS+=($!)
  done <<< "$ASSIGN"
  echo

  FAILED=0
  for pid in ${PIDS[@]+"${PIDS[@]}"}; do
    wait "$pid" || FAILED=1
  done
  cat "${LOG_FILE%.log}"_gpu*.log > "$LOG_FILE" 2>/dev/null || true
  [[ "$FAILED" == "0" ]] || die "Co tien trinh eval that bai - xem ${LOG_FILE%.log}_gpu*.log"
else
  export CUDA_VISIBLE_DEVICES="$GPU"
  ( cd "${ROOT_DIR}/Eval" && run_tasks "$RESOLVED_TASKS" ) 2>&1 | tee "$LOG_FILE"
fi

# =============================================================================
# 4. Tong hop
# =============================================================================
SUMMARY_JSON="${ROOT_DIR}/Eval/${OUTPUT_ROOT}/summary.json"

if [[ "$DRY_RUN" != "1" ]]; then
  log "Ket qua"
  META="$(printf '{"tag":"%s","model":"%s","decoding":"%s","temperature":%s,"top_p":%s,"seed":%s,"max_tokens":%s,"num_test_sample":%s,"prompt_type":"%s"}' \
    "$RUN_TAG" "$MODEL" "$([[ "$TEMPERATURE" == "0" ]] && echo greedy || echo sampling)" \
    "$TEMPERATURE" "$TOP_P" "$SEED" "$MAX_TOKENS" "$NUM_TEST_SAMPLE" "$PROMPT_TYPE")"

  python3 - "$ROOT_DIR/Eval/$OUTPUT_ROOT" "$SUMMARY_JSON" "$META" <<'PYSUM'
import json, sys, glob, os, re, datetime

root, out_path, meta = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])

# Ten thu muc chua so cau va so mau/cau, nen mot lan chay nhanh va mot lan
# chay day du nam canh nhau van phan biet duoc.
PAT = re.compile(r"_(-?\d+)_seed(\d+)_t([\d.]+)_n(\d+)_topp")

rows = []
for f in sorted(glob.glob(os.path.join(root, "*", "*", "**", "*_metrics.json"), recursive=True)):
    task = os.path.relpath(f, root).split(os.sep)[0]
    mo = PAT.search(f)
    try:
        m = json.load(open(f))
    except Exception:
        continue
    rows.append({
        "task": task,
        "num_test_sample": (int(mo.group(1)) if mo else None),
        "seed": (int(mo.group(2)) if mo else None),
        "temperature": (float(mo.group(3)) if mo else None),
        "n_sampling": (int(mo.group(4)) if mo else None),
        "acc": m.get("acc"),
        "num_samples": m.get("num_samples"),
        "empty_samples": m.get("empty_samples"),
        "timeout_samples": m.get("timeout_samples"),
        "time_use_in_second": m.get("time_use_in_second"),
        "metrics_file": os.path.relpath(f, root),
    })

accs = [r["acc"] for r in rows if isinstance(r["acc"], (int, float))]
summary = {
    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    **meta,
    "tasks": {r["task"]: r["acc"] for r in rows},
    "average_acc": (round(sum(accs) / len(accs), 2) if accs else None),
    "results": rows,
}

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as fh:
    json.dump(summary, fh, indent=2, ensure_ascii=False)

if not rows:
    print("  (chua co metrics.json nao trong %s)" % root)
else:
    hdr = "  %-12s %7s %4s %8s %10s %8s"
    print(hdr % ("task", "subset", "n", "acc", "num_samples", "time"))
    for r in rows:
        sub = "full" if r["num_test_sample"] == -1 else r["num_test_sample"]
        t = r["time_use_in_second"]
        print(hdr % (r["task"], sub, r["n_sampling"],
                     "%.1f" % r["acc"] if isinstance(r["acc"], (int, float)) else r["acc"],
                     r["num_samples"],
                     "%d:%02d" % (t // 60, t % 60) if isinstance(t, (int, float)) else "-"))
    if accs:
        print(hdr % ("TRUNG BINH", "", "", "%.1f" % (sum(accs) / len(accs)), "", ""))
print()
print("  JSON: %s" % out_path)
PYSUM
fi

log "Xong. Output: Eval/${OUTPUT_ROOT}  |  JSON: ${SUMMARY_JSON}  |  Log: ${LOG_FILE}"
