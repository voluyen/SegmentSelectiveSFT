import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
from unsloth import FastLanguageModel 
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import torch
import argparse
import sys
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from unsloth import is_bfloat16_supported
import gc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from segment_utils import DEFAULT_MODE, SEGMENT_PATTERNS, split_segments

# Tracking backend: dat REPORT_TO=wandb (kem WANDB_PROJECT) de bat lai.
REPORT_TO = os.environ.get("REPORT_TO", "none")
if REPORT_TO == "wandb":
    os.environ.setdefault("WANDB_PROJECT", "selective_sft")
else:
    # Khong dat WANDB_DISABLED: transformers da deprecate no va se in canh bao
    # lap lai moi lan khoi tao Trainer. report_to="none" ben duoi la du.
    os.environ["WANDB_MODE"] = "disabled"

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class EarlyStopAtEpochCallback(TrainerCallback):
    """Dung som tai --stop_at_epoch. Mac dinh tat (0 = chay du --epochs).

    Ban goc hard-code 'epoch >= 9' nen moi lan train qua 10 epoch deu bi cat
    am tham, khong lien quan gi den --epochs truyen vao.
    """
    def __init__(self, stop_at_epoch):
        self.stop_at_epoch = stop_at_epoch

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.stop_at_epoch and state.epoch >= self.stop_at_epoch:
            print(f"Epoch {state.epoch:.1f} reached. Stopping training early.")
            control.should_training_stop = True
        return control


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="GAIR/LIMO", type=str)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B-Instruct", type=str)
    parser.add_argument("--output_dir", default="", type=str,
                        help="De trong = tu dat ten theo model/epoch/lr/len (eval.sh dua vao quy uoc nay)")
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--max_seq_length", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--stop_at_epoch", type=float, default=0,
                        help="0 = chay du --epochs; >0 = dung khi state.epoch dat nguong nay")
    parser.add_argument("--seed", type=int, default=3407)

    # --- Optimizer / scheduler ---
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--optim", default="adamw_torch", type=str,
                        help="adamw_torch (mac dinh) hoac adamw_8bit de tiet kiem VRAM optimizer state")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--lr_scheduler_type", default="cosine", type=str)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # --- LoRA ---
    parser.add_argument("--full_finetune", action="store_true",
                        help="Full finetuning thay vi LoRA (mac dinh la LoRA)")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", default=",".join(DEFAULT_TARGET_MODULES), type=str)

    # Gom cac mau dai tuong duong vao cung batch -> gan nhu khong con padding.
    parser.add_argument("--group_by_length", action="store_true")
    # Tat gradient checkpointing: ton VRAM hon nhung khong phai tinh lai
    # activation trong backward -> nhanh hon dang ke.
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--deepseek", action="store_true",
                        help="Chat template DeepSeek R1 (template tu chen <think>). Bo di = template Qwen/ChatML.")
    parser.add_argument("--think_prefix", default="none", choices=["none", "plain", "special"],
                        help="Chi ap dung cho nhanh Qwen. none = khong chen <think> (khop voi "
                             "prompt luc eval); plain = chen <think> dang text thuong; "
                             "special = chen va them token dac biet + resize embedding (ban goc, can full finetune)")
    parser.add_argument("--segment_mode", default=DEFAULT_MODE, choices=sorted(SEGMENT_PATTERNS))
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--apply_all", action="store_true")

    args = parser.parse_args()
    args.use_lora = not args.full_finetune
    args.target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]

    # embed_tokens/lm_head bi dong bang khi dung LoRA, nen token moi them vao
    # se giu nguyen gia tri khoi tao ngau nhien suot qua trinh train.
    if args.use_lora and args.think_prefix == "special":
        embed_in_target = any(m in args.target_modules for m in ("embed_tokens", "lm_head"))
        if not embed_in_target:
            raise SystemExit(
                "--think_prefix special them token moi va resize embedding, nhung LoRA dong bang "
                "embed_tokens/lm_head nen token do khong bao gio duoc hoc.\n"
                "Chon --think_prefix none (khuyen nghi) hoac plain, hoac train bang --full_finetune."
            )
    return args

args = parse_args()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.model_name_or_path,
    max_seq_length = args.max_seq_length, # Choose any for long context!
    load_in_4bit = args.load_in_4bit,  # 4 bit quantization to reduce memory
    full_finetuning = not args.use_lora,
)

if not args.deepseek:
    # Chat template ChatML (Qwen). apply_chat_template(add_generation_prompt=True)
    # ket thuc bang "<|im_start|>assistant\n" - day cung la prefix ma
    # Eval/math_eval.py sinh ra luc eval, nen response_template phai khop.
    if args.think_prefix == "special":
        tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>", "<|reason_pad|>"]})
        model.resize_token_embeddings(len(tokenizer))
    think_str = "" if args.think_prefix == "none" else "<think>\n"
    instruction_template = "<|im_start|>user"
    response_template = "<|im_start|>assistant\n" + think_str
else:
    # Template cua DeepSeek R1 tu chen "<think>\n" sau <｜Assistant｜>.
    think_str = ""
    instruction_template = "<｜begin▁of▁sentence｜><｜User｜>"
    response_template = "<｜Assistant｜><think>\n"

# Mask bam theo offset ky tu nen khong con do chuoi token cua response_template
# nua; doi lai kiem tra mot lan o day rang chat template dung la ket thuc bang
# prefix ta mong doi. Sai --deepseek/--think_prefix se lo ra ngay tai buoc nay
# thay vi tao ra label sai am tham.
_probe = tokenizer.apply_chat_template(
    [{"role": "user", "content": "probe"}], tokenize=False, add_generation_prompt=True
)
if not (_probe + think_str).endswith(response_template):
    raise SystemExit(
        "Chat template cua %r ket thuc bang %r, khong khop response_template %r.\n"
        "Kiem tra co dat dung --deepseek (cho DeepSeek R1) hay --think_prefix (cho Qwen) khong."
        % (args.model_name_or_path, (_probe + think_str)[-40:], response_template)
    )
print("Chat template khop response_template: %r" % response_template)

if not tokenizer.is_fast:
    raise SystemExit(
        "Can fast tokenizer (tokenizer.json) de lay offset_mapping - do la cach duy nhat "
        "xac dinh dung bien segment trong chuoi token. Model %r chi co tokenizer cham."
        % args.model_name_or_path
    )

args.run_name = args.model_name_or_path.split("/")[-1] + "max_seq_" + str(args.max_seq_length) + "lr_" + str(args.learning_rate) + "epochs_" + str(args.epochs)
if not args.output_dir:
    # Quy uoc ten: eval.sh dung lai chuoi nay de tim checkpoint moi nhat.
    suffix = ""
    if not args.mask:
        # Khong --mask = long-CoT SFT thuong (supervise ca response). Doi ten
        # thu muc de checkpoint baseline khong de len ban selective.
        suffix += "_fullsft"
    if args.use_lora:
        suffix += "_lora"
    args.output_dir = f"./checkpoints/{args.model_name_or_path.split('/')[-1]}_epoch{args.epochs}_lr{args.learning_rate}_len{args.max_seq_length}{suffix}"
    args.run_name += suffix

USE_GC = not args.no_gradient_checkpointing
model.config.use_cache = False

if args.use_lora:
    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_r,
        target_modules = args.target_modules,
        lora_alpha = args.lora_alpha,
        lora_dropout = args.lora_dropout,
        bias = "none",
        # "unsloth" = ban gradient checkpointing tiet kiem VRAM cua unsloth,
        # can thiet o seq 32768. Trainer khong bat lai (xem SFTConfig ben duoi).
        use_gradient_checkpointing = "unsloth" if USE_GC else False,
        random_state = args.seed,
        use_rslora = False,
        loftq_config = None,
    )
    trainer_gc = False
else:
    if USE_GC:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()
    trainer_gc = USE_GC

# dataset.map(batched=True) duoc phep tra ve it dong hon dau vao, nen mau hong
# bi bo han thay vi lam ca batch chet.
SKIPPED = {"n": 0}


def segment_char_bounds(segments, offset):
    """Vi tri [start, end) theo KY TU cua tung segment trong full_text."""
    bounds, cursor = [], offset
    for seg in segments:
        bounds.append((cursor, cursor + len(seg)))
        cursor += len(seg)
    return bounds


def formatting_prompts_func(examples):
    questions = examples["question"]
    outputs = examples["solution"] 
    segments_ids = examples["selected_spans_ids"]

    input_ids_list = []
    labels_list = []
    for prompt, output, segment_id in zip(questions, outputs, segments_ids):
        messages = [
            {"role": "user", "content": prompt + "\nPlease reason step by step, and put your final answer within \\boxed{}."},
        ]
  
        input_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )   
        full_text = input_str + think_str + output

        full_tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=args.max_seq_length,
            return_offsets_mapping=True,
        )
        input_ids = full_tokens["input_ids"]
        offsets = full_tokens["offset_mapping"]
        labels = [-100] * len(input_ids)

        # Moc theo KY TU chu khong do do dai token cua tung doan roi cong don.
        # Tokenize rieng "".join(segments[:k]) cho ra so token khac voi chinh
        # doan do khi nam trong full_text (BPE merge khac nhau o bien), lam span
        # lech vai token va mask an sang segment KHONG duoc chon.
        response_char = len(input_str) + len(think_str)

        if not args.mask:
            supervise = lambda seg_idx: True
            bounds = None
        else:
            segment = split_segments(output, args.segment_mode)
            # Luon hoc segment dau, gan cuoi va cuoi du IG khong chon chung.
            add_ids = [0, len(segment) - 2, len(segment) - 1]
            keep = {i for i in add_ids + list(segment_id) if 0 <= i < len(segment)}
            bounds = segment_char_bounds(segment, response_char)
            supervise = lambda seg_idx: seg_idx in keep

        n_supervised = 0
        for i, (char_start, char_end) in enumerate(offsets):
            if char_end <= char_start:
                continue                      # token dac biet, khong anh xa ra ky tu nao
            if char_start < response_char:
                continue                      # phan prompt, luon giu -100
            if bounds is None:
                labels[i] = input_ids[i]
                n_supervised += 1
                continue
            # Token thuoc ve segment chua ky tu dau tien cua no.
            lo, hi = 0, len(bounds) - 1
            seg_idx = None
            while lo <= hi:
                mid = (lo + hi) // 2
                if char_start < bounds[mid][0]:
                    hi = mid - 1
                elif char_start >= bounds[mid][1]:
                    lo = mid + 1
                else:
                    seg_idx = mid
                    break
            if seg_idx is not None and supervise(seg_idx):
                labels[i] = input_ids[i]
                n_supervised += 1

        if n_supervised == 0:
            # Toan bo label = -100 thi cross-entropy tra ve nan va lam hong ca
            # run. Xay ra khi max_seq_length cat mat phan response. Bo mau nay.
            SKIPPED["n"] += 1
            continue

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list
    }
if "json" in args.data_names or args.data_names.endswith(".jsonl"):
    dataset = load_dataset("json", data_files=args.data_names)['train']
else:
    dataset = load_dataset(args.data_names, split = args.split)

dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=["question", "solution", "answer", "selected_spans_ids", "segments"],  
    load_from_cache_file=False,
)
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

world_size = int(os.environ.get("WORLD_SIZE", "1"))
effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
print("=" * 60)
print("  che do        : %s | %s" % ("LoRA" if args.use_lora else "full finetune",
                                     "selective (mask)" if args.mask else "full-CoT SFT"))
if args.use_lora:
    print("  lora          : r=%d alpha=%d dropout=%s" % (args.lora_r, args.lora_alpha, args.lora_dropout))
    print("  target_modules: %s" % ",".join(args.target_modules))
print("  segment_mode  : %s" % args.segment_mode)
print("  seq len       : %d" % args.max_seq_length)
print("  batch         : %d x %d accum x %d gpu = %d/step" % (
    args.per_device_train_batch_size, args.gradient_accumulation_steps, world_size, effective_batch))
print("  optim         : %s betas=(%s, %s) eps=%s wd=%s" % (
    args.optim, args.adam_beta1, args.adam_beta2, args.adam_epsilon, args.weight_decay))
print("  scheduler     : %s warmup_ratio=%s lr=%s epochs=%d" % (
    args.lr_scheduler_type, args.warmup_ratio, args.learning_rate, args.epochs))
print("  grad ckpt     : %s" % ("unsloth" if (args.use_lora and USE_GC) else ("on" if USE_GC else "off")))
print("  output_dir    : %s" % args.output_dir)
print("  so mau        : %d%s" % (
    len(dataset),
    "" if SKIPPED["n"] == 0 else " (da bo %d mau bi cat mat phan response o max_seq_length=%d)"
    % (SKIPPED["n"], args.max_seq_length)))
if len(dataset) == 0:
    raise SystemExit("Khong con mau nao sau khi loc - kiem tra --max_seq_length va du lieu dau vao.")
print("=" * 60)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    tokenizer = tokenizer,
    dataset_num_proc=2,
    packing = False,
    args = SFTConfig(
        max_seq_length = args.max_seq_length,
        remove_unused_columns = False,
        dataset_kwargs = {"skip_prepare_dataset": True},
        per_device_train_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        warmup_ratio = args.warmup_ratio,
        num_train_epochs = args.epochs, 
        learning_rate = args.learning_rate,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        lr_scheduler_type = args.lr_scheduler_type,
        output_dir = args.output_dir,
        optim = args.optim,
        weight_decay = args.weight_decay,
        adam_beta1 = args.adam_beta1,
        adam_beta2 = args.adam_beta2,
        adam_epsilon = args.adam_epsilon,
        seed = args.seed,
        report_to = REPORT_TO,
        run_name = args.run_name,
        save_strategy = "epoch",
        overwrite_output_dir=True,
        save_total_limit = 3,
        save_only_model=True,
        gradient_checkpointing=trainer_gc,
        group_by_length = args.group_by_length,
        max_grad_norm=args.max_grad_norm,
    ),
    callbacks=[EarlyStopAtEpochCallback(args.stop_at_epoch)],
)
trainer.train()
