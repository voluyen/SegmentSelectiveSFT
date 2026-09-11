# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Fork of the ICLR 2026 paper "Segment-Level Attribution for Selective Learning of Long Reasoning Traces"
(see `README.md`, `2602.00425v1.pdf`). Upstream provides three stages — `Attribution/`, `SelectiveSFT/`,
`Eval/` — each driven by a shell script with hard-coded config. This fork adds parameterized wrappers at
the repo root (`run_pipeline.sh`, `train.sh`, `eval.sh`); prefer editing/extending those over the
upstream scripts (`Eval/run_eval.sh`, `SelectiveSFT/run_train.sh`, `Attribution/cal_attribution.sh`),
which are kept close to as-published and still carry the paper's original LIMO/DeepSeek config.

**This fork's current configuration diverges from the paper**: LoRA (not full finetuning) on
`Qwen/Qwen2.5-7B-Instruct`, trained on `simplescaling/s1K-1.1` (not LIMO), with segments split on
`\n\n` (not the paper's backtracking-cue regex). See "Current training configuration" below.

Comments and commit messages in this repo are written in **Vietnamese without diacritics**. Match that
style when editing shell scripts, adding comments, or writing commits.

## Two incompatible Python environments

This is the single most important constraint. Never install both requirement sets into one env:

| Env | Requirements | Pins | Used by |
|---|---|---|---|
| `ssft_train` | `requirements-sft.txt` (superset of `SelectiveSFT/requirements.txt`) | torch 2.9.0, transformers 4.57.1, unsloth, peft, `torchao<0.18` | `train.sh`, `SelectiveSFT/merge_lora.py` |
| `ssft_eval` | `requirements.txt` (root) + `Eval/latex2sympy` editable | torch 2.7.1, transformers 4.56.0, vllm 0.10.0 | `eval.sh`, attribution stages, CoT generation |

`peft` lives only in the train env, so **LoRA merging must run there**, not in the eval env.

Environment setup lives in **`setup.sh`**, not in the pipeline — `run_pipeline.sh` only runs stages.
`bash setup.sh check --for train|eval|all` verifies an environment without installing anything (useful
on air-gapped machines); `bash setup.sh eval` / `bash setup.sh train` install the respective set, with
`--conda <name>` / `--venv <dir>` to build an isolated one. It refuses quietly-broken combinations by
warning when it sees the other side's marker package (`unsloth` vs `vllm`) already installed.

**All three pipeline wrappers default to `USE_CONDA=0` — they use whatever Python is already active and build
nothing.** That suits managed cloud environments (Lightning Studio and similar) that ship a complete env
and no `conda` on PATH. Pass `--use-conda` (or `USE_CONDA=1`, or `--env <name>`, which implies it) to get
the original behavior: `run_pipeline.sh` creating/activating `selective_sft`, `train.sh` and `eval.sh`
each building their own env and pip-installing. Under `--use-conda`, `run_pipeline.sh`'s `train` stage
lets `train.sh` manage `ssft_train` itself; without it, the stage passes `--skip-setup` so nothing is
rebuilt. When the two environments above are NOT separated, the torch/vLLM conflict is yours to avoid.

`requirements-sft-lock.txt` is the fully-pinned transitive lock for the train env. `torchao<0.18` is
load-bearing: 0.18 breaks `import unsloth`.

`train.sh` / `eval.sh` build their env on first run and drop a stamp file (`.setup_done_<env>_v<N>`) to
skip reinstalling. Bump the `_vN` suffix in the script when the requirements change, or the stamp will
mask the new deps. `--reinstall` forces, `--skip-setup` skips.

## Commands

```bash
# Environment (separate from the pipeline)
bash setup.sh check --for train             # verify without installing
bash setup.sh eval                          # requirements.txt + latex2sympy
bash setup.sh train --conda ssft_train      # requirements-sft.txt in its own env

# Data prep, once
python prepare_s1k.py                       # -> data/s1k/train.jsonl
bash run_pipeline.sh --stages prep,split    # -> data/s1k/solution_segments.jsonl

# Default pipeline: attribution + selective SFT (assumes solution_segments.jsonl exists)
bash run_pipeline.sh                        # = --stages ig,segments,train
bash run_pipeline.sh --stages ig --force     # ig stage appends; --force clears the old file first
bash run_pipeline.sh --offline               # air-gapped: sets HF_HUB_OFFLINE + HF_DATASETS_OFFLINE
bash run_pipeline.sh --segment-mode cue      # paper's backtracking-cue split instead of "\n\n"
DRY_RUN=1 bash run_pipeline.sh

# Training (creates env, trains, logs to logs/train_lora.log)
bash train.sh                       # selective SFT, LoRA — see config table below
bash train.sh --full-sft            # baseline: supervise the whole CoT
bash train.sh --full-finetune       # no LoRA (very heavy at seq 32768 on 7B)
bash train.sh --epochs 5 --lr 1e-5 --gpu 1
bash train.sh --lora-r 32 --lora-alpha 64 --target-modules "q_proj,v_proj"
bash train.sh --no-grad-checkpoint  # faster, more VRAM
bash train.sh --dry-run             # print commands only

# LoRA checkpoints are adapters — merge before eval (train env, CPU is fine)
cd SelectiveSFT && python merge_lora.py --adapter checkpoints/<run>/checkpoint-<step>

# Eval (separate env; resumable, prints an accuracy table + writes summary.json)
bash eval.sh --model /abs/path/checkpoint-<step>-merged --tag sel_ep3
bash eval.sh --base                 # un-finetuned base model
bash eval.sh --quick                # math500, 100 questions, 1 sample/question
bash eval.sh --tasks "aime24 math500" --n-sampling 1
bash eval.sh --gpu 0,1,2,3 --data-parallel   # one task per GPU (faster than TP)
bash eval.sh --overwrite            # re-score from scratch

# The only unit tests in the repo (vendored latex2sympy parser)
cd Eval/latex2sympy && pytest tests/            # single test: pytest tests/trig_test.py
```

All three wrappers take `--dry-run` and `-h`.

## Current training configuration

`train.sh` defaults, all overridable by flag or env var:

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Data | `data/s1k/solutions_selected.jsonl` (from `simplescaling/s1K-1.1`) |
| Tuning | LoRA r=16, alpha=16, dropout=0.05, bias=none |
| `target_modules` | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| LR / epochs | 5e-5 / 3 |
| Effective batch | 32 (`per_device 1` x `grad_accum 32` x 1 GPU) |
| Optimizer | `adamw_torch`, betas (0.9, 0.999), eps 1e-8, weight_decay 0.0 |
| Scheduler | cosine + warmup, `warmup_ratio=0.1` (HF builds this as a `LambdaLR`) |
| `max_seq_length` | 32768 (= Qwen2.5-7B's `max_position_embeddings`; no RoPE scaling) |
| Gradient checkpointing | on, unsloth's variant (required at 32k on 7B) |
| Segmentation | `paragraph` — split on every `\n\n` |

Effective batch is `per_device x grad_accum x WORLD_SIZE` — launching under DDP with more than one process multiplies it past 32.

## Pipeline data flow

```
simplescaling/s1K-1.1 (HuggingFace)
  │  prepare_s1k.py — question / deepseek_thinking_trajectory / last \boxed{}
  ▼
data/s1k/train.jsonl                 (question, solution, answer)
  │  Attribution/segment_split.py — split on "\n\n" (--segment_mode cue for the paper's split)
  ▼
data/s1k/solution_segments.jsonl     (+ segments[])
  │  Attribution/grad_analyze.py — Integrated Gradients from each segment's tokens
  │                                to the \boxed{answer} tokens (ig_steps=50)
  ▼
Attribution/processed_data/s1k/IG.jsonl
  │  Attribution/get_important_segments.py — per-segment score = sum|IG| / sqrt(len);
  │    keep top segments up to --cumulative_ratio (0.7) of the mass, drop those with
  │    |sum IG| / sum|IG| > --coherence_max (0.8)
  ▼
data/s1k/solutions_selected.jsonl    (+ selected_spans_ids[])
  │  SelectiveSFT/train_mask.py — labels = -100 except the selected segments
  ▼
SelectiveSFT/checkpoints/<model>_epoch<E>_lr<LR>_len<L>[_fullsft][_lora]/checkpoint-<step>
  │  SelectiveSFT/merge_lora.py (LoRA only) -> checkpoint-<step>-merged
  │  Eval/math_eval.py via eval.sh
  ▼
Eval/outputs_<tag>/<task>/<tag>/*_metrics.json  +  Eval/outputs_<tag>/summary.json
```

`segment_utils.py` (repo root) owns the splitting rules and is imported by both
`Attribution/segment_split.py` and `SelectiveSFT/train_mask.py`, which each insert the repo root on
`sys.path`. It guarantees `"".join(split_segments(t)) == t` and no empty segments — `grad_analyze.py`
derives token spans from cumulative segment lengths, so a splitter that dropped characters would
silently misalign every downstream span, and a zero-length segment divides by zero in
`get_important_segments.py`. Keep both invariants if you add a split mode.

## Sharp edges

- **Masking is anchored to character offsets, not cumulative token counts.** `train_mask.py` tokenizes
  with `return_offsets_mapping=True` and assigns each token to the segment holding its first character.
  The earlier approach — locate `response_template` in the full sequence, then add
  `len(tokenizer("".join(segments[:k])))` — mixes two different tokenizations (prompt+response vs.
  response alone) and drifts a token or two at the junction, leaking text from *unselected* segments
  into the supervised span. This requires a fast tokenizer; the script exits early if it doesn't get one.
- **A sample whose labels are all `-100` yields `nan` loss and poisons the run.** This happens when
  `max_seq_length` truncates away the response. `train_mask.py` drops such samples in `.map()` and
  reports the count; it does not crash.
- **`grad_analyze.py` opens `--output_data_file` in append mode.** Re-running without deleting it
  duplicates every record and breaks the length assert downstream. `run_pipeline.sh` refuses to proceed
  unless `--force`.
- **Checkpoint directory names come from bash, not Python.** `train.sh` passes `--output_dir` explicitly
  because `train_mask.py` used to build the name with an f-string over a float — `--lr 5e-5` became
  `_lr5e-05` and `1e-4` became `_lr0.0001`, neither matching the string `eval.sh` reconstructs. If you
  change the naming, change it in `train.sh` and `eval.sh` together.
- **LoRA checkpoints are adapters; vLLM can't load them.** Run `SelectiveSFT/merge_lora.py` in the
  **train** env (peft lives only there; CPU is fine), then eval the `-merged` directory. `eval.sh`
  detects a bare adapter dir, auto-uses a sibling `-merged` if present, and otherwise stops with the
  exact merge command.
- **`--think_prefix` exists because Qwen2.5 has no `<think>` token.** The original code added
  `<think>`/`</think>`/`<|reason_pad|>` and resized embeddings — but under LoRA those rows are frozen, so
  the new tokens would never train, and the eval-time prompt never emits `<think>` anyway. The default
  `none` drops the scaffolding so training and eval see the identical prefix. `special` (the old
  behavior) is rejected under LoRA unless `embed_tokens`/`lm_head` are in `target_modules`.
- **Eval scripts must run with cwd = `Eval/`**: `parser.py`/`grader.py` do
  `from latex2sympy.latex2sympy2 import ...` (resolved via the local package dir), and `--data_dir`
  defaults to `../data`. All wrappers `cd` there.
- **`math_eval.py` skips a task whose `*_metrics.json` already exists** — that is what makes `eval.sh`
  resumable after a crash. The output filename encodes
  `num_test_sample/seed/temperature/n_sampling/max_tokens`, so a `--quick` run and a full run coexist
  without clobbering. Use `--overwrite` to force re-generation.
- `--quick` deliberately does **not** shrink `--max-tokens`: truncating a reasoning model's generation
  drops the boxed answer and deflates accuracy.
- **`run_pipeline.sh`'s `setup` stage builds only the eval-side env.** Its `train` stage shells out to
  `train.sh` without `--skip-setup` so that script activates `ssft_train` itself — do not "optimize" that
  into a direct `train_mask.py` call under the pipeline's env.
- **`EarlyStopAtEpochCallback` used to hard-stop at `state.epoch >= 9`** regardless of `--epochs`. It is
  now driven by `--stop_at_epoch` and disabled by default.
- Every generated artifact is gitignored (`logs/`, `SelectiveSFT/checkpoints/`,
  `Attribution/processed_data/`, `Eval/outputs*`). `data/s1k/*` is produced by `prepare_s1k.py` and the
  attribution stages; the committed `data/limo/solutions_top70cohe80_lennorm_7B_J50.jsonl` is the
  paper's original LIMO training file.

## Defaults worth knowing

- Attribution model = training model = `Qwen/Qwen2.5-7B-Instruct` (the paper instead attributed with a
  7B model and trained a 1.5B one; `run_pipeline.sh --attr-model` still separates them).
- IG attribution is the expensive stage: `ig_steps=50` forward+backward passes over the full sequence
  per sample, on a 7B model. Paragraph splitting also produces far more segments per trace (~100-500)
  than the paper's cue splitting (~10-30), which changes how many segments clear the 70% mass threshold.
- Eval benchmarks and samples/question: `aime24:32 amc23:32 math500:6 minerva:6 gpqa:6 olympiad:6`,
  temperature 0.6, top_p 1, max 32768 tokens, prompt type `deepseek-longcot` (the prompt string matches
  what `train_mask.py` builds, so training and eval stay aligned).
- W&B is off everywhere (`REPORT_TO=none`, `WANDB_MODE=disabled`); set `REPORT_TO=wandb` +
  `WANDB_PROJECT` to re-enable.
