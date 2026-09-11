import os
import gc
import json
import torch
from tqdm import tqdm
import argparse
import math
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# torch.OutOfMemoryError chi co tu torch 2.5 tro di.
OOM_ERROR = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError


class IntegratedGradientsAttribution:
    """
    Compute Integrated Gradients (IG) attributions from intermediate tokens
    to the model's final answer tokens.

    Notes:
      - We attribute the summed log-probability of answer tokens to input token embeddings.
      - The baseline is a sequence filled with `baseline_token_id` (often the pad token).
    """
    def __init__(self, model_name, gradient_checkpointing=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if not self.tokenizer.is_fast:
            raise SystemExit(
                "Can fast tokenizer (tokenizer.json) de lay offset_mapping - do la cach duy "
                "nhat xac dinh dung bien segment. Model %r chi co tokenizer cham." % model_name
            )
        print("pad_token_id", self.tokenizer.pad_token_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        self.model.eval()

        if gradient_checkpointing:
            # IG can backward qua toan bo chuoi ma khong dung optimizer, nen
            # activation cua ca 28 lop bi giu lai - day la phan ton VRAM nhat.
            # HF chi ap dung checkpointing khi module o training mode, nen phai
            # goi train(). Voi Qwen2 dropout = 0 nen ket qua khong doi; neu
            # model co dropout that thi bao loi thay vi lam sai am tham.
            drop = getattr(self.model.config, "attention_dropout", 0.0) or 0.0
            extra = [m.p for m in self.model.modules()
                     if isinstance(m, torch.nn.Dropout) and m.p > 0]
            if drop > 0 or extra:
                raise SystemExit(
                    "Model co dropout > 0 (attention_dropout=%s, Dropout layers=%s). "
                    "Bat gradient checkpointing phai chuyen sang train() nen dropout se "
                    "lam nhieu ket qua IG. Chay lai voi --no_gradient_checkpointing."
                    % (drop, extra)
                )
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.model.train()
            print("gradient checkpointing: BAT")
        else:
            print("gradient checkpointing: TAT")

    @torch.no_grad()
    def _embed(self, input_ids):
        return self.model.model.embed_tokens(input_ids)

    def compute_step_to_answer_attribution_integrated(self, input_ids, step_indices, answer_indices, baseline_token_id=0, steps=50):
        input_ids = torch.tensor(input_ids, dtype=torch.int32).unsqueeze(0).cuda()
        
        with torch.no_grad():
            input_embeddings = self._embed(input_ids)  # [1, L, D]
            baseline_embeddings = self._embed(torch.full_like(input_ids, baseline_token_id))      # [1, L, D]

        # Interpolation path: baseline -> input
        alphas = torch.linspace(0, 1, steps).view(-1, 1, 1, 1).to(torch.bfloat16).cuda()
        total_gradients = torch.zeros_like(input_embeddings) # [1, L, D]

        answer_token_ids = input_ids[0, answer_indices[0]:answer_indices[1]] # [A]
        
        for i in range(steps):
            interpolated_embedding = (baseline_embeddings + alphas[i] * (input_embeddings - baseline_embeddings)).detach()
            interpolated_embedding.requires_grad_(True)
        
            self.model.zero_grad()
            output = self.model(inputs_embeds=interpolated_embedding)
            logits = output.logits # [1, L, V]

            target_logits = logits[0, answer_indices[0]-1:answer_indices[1]-1] # [A, V]
            log_probs = torch.nn.functional.log_softmax(target_logits, dim=-1) # [A, V]
            target_log_probs = log_probs[range(len(answer_token_ids)), answer_token_ids]

            loss = target_log_probs.sum()
            loss.backward()

            total_gradients += interpolated_embedding.grad

        avg_gradients = total_gradients / steps
        input_diff = input_embeddings - baseline_embeddings  
        attributions = (input_diff * avg_gradients).sum(dim=-1).squeeze()
        attributions = attributions / attributions.norm()

        step_scores = []
        for step_idx in step_indices:
            step_score = attributions[step_idx[0]: step_idx[1]]
            step_scores.append(step_score.detach().cpu().float().tolist())

        del output, total_gradients, input_embeddings, baseline_embeddings
        torch.cuda.empty_cache()
        return step_scores
        

    def batch_compute_step_to_answer_attribution_integrated(
        self,
        input_ids,
        step_indices,
        answer_indices,
        baseline_token_id=0,
        steps=50,
        batch_size=1,   
    ):
        # [1, L]
        input_ids = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0).cuda()

        with torch.no_grad():
            input_embeddings = self._embed(input_ids)  # [1, L, D]
            baseline_embeddings = self._embed(torch.full_like(input_ids, baseline_token_id))      # [1, L, D]

        alphas = torch.linspace(0, 1, steps).view(steps, 1, 1).to(torch.bfloat16).cuda()
        total_gradients = torch.zeros_like(input_embeddings)  # [1, L, D]

        ans_start, ans_end = answer_indices
        answer_token_ids = input_ids[0, ans_start:ans_end]          # [A]

        step_pos = 0
        while step_pos < steps:
            chunk_end = min(step_pos + batch_size, steps)
            chunk_size = chunk_end - step_pos
            alphas_chunk = alphas[step_pos:chunk_end]  # [chunk_size, 1, 1]

            # Expand embeddings to [chunk_size, L, D]
            input_expand = input_embeddings.expand(chunk_size, -1, -1)
            baseline_expand = baseline_embeddings.expand(chunk_size, -1, -1)

            # Interpolation path: baseline -> input
            interpolated_embeddings = baseline_expand + alphas_chunk * (input_expand - baseline_expand).detach()
            interpolated_embeddings = interpolated_embeddings.to(dtype=torch.bfloat16)
            interpolated_embeddings.requires_grad_(True)

            # forward：batch = chunk_size
            self.model.zero_grad(set_to_none=True)
            # Goi thang base model roi tu ap lm_head len DUNG cac vi tri can.
            # Neu goi self.model(...) thi lm_head chay tren ca L vi tri, sinh
            # tensor [B, L, 152064] (~2 GB o L=6800, bf16) cong gradient cua no,
            # trong khi chi dung vai vi tri cua dap an. Ket qua khong doi.
            hidden = self.model.model(inputs_embeds=interpolated_embeddings).last_hidden_state
            target_hidden = hidden[:, ans_start-1:ans_end-1, :]      # [chunk_size, A, H]
            target_logits = self.model.lm_head(target_hidden)        # [chunk_size, A, V]
            log_probs = torch.nn.functional.log_softmax(target_logits, dim=-1)  # [chunk_size, A, V]

            # answer_token_ids: [A] -> [chunk_size, A]
            answer_ids_expand = answer_token_ids.unsqueeze(0).expand(chunk_size, -1)  # [chunk_size, A]

            target_log_probs = log_probs.gather(
                dim=-1,
                index=answer_ids_expand.unsqueeze(-1)
            ).squeeze(-1) # [chunk_size, A]

            loss = target_log_probs.sum()
            grads_chunk = torch.autograd.grad(
                loss,
                interpolated_embeddings,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]   # [chunk_size, L, D]

            # Sum gradients over the chunk's interpolation points
            total_gradients += grads_chunk.sum(dim=0, keepdim=True)  # -> [1, L, D]
            step_pos = chunk_end

        avg_gradients = total_gradients / steps  # [1, L, D]
        input_diff = input_embeddings - baseline_embeddings  # [1, L, D]
        attributions = (input_diff * avg_gradients).sum(dim=-1).squeeze(0)  # [L]
        attributions = attributions / attributions.norm()

        step_scores = []
        for step_idx in step_indices:
            step_attr = attributions[step_idx[0]: step_idx[1]]
            step_scores.append(step_attr.detach().cpu().float().numpy().tolist())

        del input_embeddings, baseline_embeddings, total_gradients
        torch.cuda.empty_cache()

        return step_scores
    

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True, help="HuggingFace model name or local path")
    p.add_argument("--input_data", type=str, required=True, help="Path to input jsonl")
    p.add_argument("--output_data_file", type=str, required=True, help="Path to output jsonl (appended)")
    p.add_argument("--output_ig_file", type=str, required=True, help="Path to output IG jsonl")
    p.add_argument("--ig_steps", type=int, default=20, help="Number of IG steps")
    p.add_argument("--no_gradient_checkpointing", action="store_true",
                   help="Tat gradient checkpointing: nhanh hon nhung ton VRAM hon nhieu")
    p.add_argument("--output_compact_file", type=str, default="",
                   help="File tong hop theo SEGMENT (3 so/segment) thay vi tung token. "
                        "De trong = tu dat ten <output_ig_file> doi duoi thanh _compact.jsonl. "
                        "Nho hon ~30 lan ma get_important_segments.py van dung duoc.")
    p.add_argument("--overwrite", action="store_true",
                   help="Tinh lai tu dau. Mac dinh: neu da co ket qua do dang thi chay tiep tu do.")
    p.add_argument("--max_input_tokens", type=int, default=0,
                   help="0 = khong gioi han. >0 = mau dai hon nguong nay se duoc gan diem 0 "
                        "thay vi tinh IG, de khong OOM giua chung")
    # p.add_argument("--baseline_token", type=str, default="pad", choices=["pad", "zero"], help="Baseline token choice")
    return p.parse_args()


if __name__ == "__main__":
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

    args = parse_args()
    attribution_calculator = IntegratedGradientsAttribution(
        args.model_name, gradient_checkpointing=not args.no_gradient_checkpointing
    )

    input_data = []
    with open(args.input_data, "r") as f:
        for line in f:
            json_obj = json.loads(line.strip())  
            input_data.append(json_obj)

    skipped_oom = 0
    skipped_long = 0
    os.makedirs(os.path.dirname(args.output_data_file), exist_ok=True)

    # --- Chay tiep tu ket qua do dang ---
    # Stage nay chay nhieu gio nen dut giua chung la chuyen binh thuong. Doc lai
    # phan da xong, doi chieu 'question' de chac chan khop dung mau, roi chay
    # tiep. Khong khop (doi dataset, doi cach chia segment) thi dung han thay vi
    # tron hai lan chay vao nhau.
    done = 0
    write_mode = 'w'
    if not args.overwrite and os.path.exists(args.output_data_file):
        existing = []
        with open(args.output_data_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing.append(json.loads(line))
                except json.JSONDecodeError:
                    # Dong cuoi co the bi cat giua chung neu bi kill dot ngot.
                    print("Bo dong cuoi bi hong trong %s" % args.output_data_file)
                    break
        for i, rec in enumerate(existing):
            if i >= len(input_data) or rec.get("question") != input_data[i].get("question"):
                raise SystemExit(
                    "Ket qua cu trong %s khong khop du lieu dau vao tai mau %d.\n"
                    "Co ve la cua lan chay voi dataset/cach chia segment khac. "
                    "Chay lai voi --overwrite de tinh lai tu dau." % (args.output_data_file, i)
                )
        done = len(existing)
        if done >= len(input_data):
            print("Da co du %d/%d mau, khong con gi de tinh." % (done, len(input_data)))
        elif done > 0:
            print("Chay tiep tu mau %d/%d (%d mau da xong)." % (done, len(input_data), done))
        write_mode = 'a'
        del existing

    with open(args.output_data_file, write_mode) as f:
        input_template = "{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        for n in tqdm(range(done, len(input_data)), initial=done, total=len(input_data)): 
            each_data = input_data[n]
            user_msg = input_template.format(input=each_data["question"])
            user_tokens = attribution_calculator.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=True,
                add_generation_prompt=True,
            )
            
            pred_thoughts = each_data["segments"]
            # Span token cua tung segment lay tu offset_mapping cua CA chuoi da
            # ghep, khong do do dai token cua tung prefix cong don: tokenize
            # rieng "".join(segments[:k]) cho so token khac voi chinh doan do khi
            # nam trong chuoi day du (BPE merge khac o bien) -> span lech vai
            # token va IG bi gan nham segment.
            joined = "".join(pred_thoughts)
            enc = attribution_calculator.tokenizer(
                joined, add_special_tokens=False, return_offsets_mapping=True
            )
            assistant_tokens = enc["input_ids"]
            token_offsets = enc["offset_mapping"]

            seg_char_start, cursor = [], 0
            for seg in pred_thoughts:
                seg_char_start.append(cursor)
                cursor += len(seg)

            # Token thuoc segment chua ky tu dau tien cua no.
            token_seg = []
            si = 0
            for (c0, c1) in token_offsets:
                if c1 <= c0:
                    token_seg.append(-1)
                    continue
                while si + 1 < len(seg_char_start) and c0 >= seg_char_start[si + 1]:
                    si += 1
                token_seg.append(si)

            # Mot luot duy nhat qua token thay vi quet lai ca mang cho tung
            # segment: chia "\n\n" cho 100-500 segment tren vai nghin token nen
            # vong lap long la O(segment x token), cham gap ~100 lan.
            first_tok, last_tok = {}, {}
            for i, sk in enumerate(token_seg):
                if sk < 0:
                    continue
                if sk not in first_tok:
                    first_tok[sk] = i
                last_tok[sk] = i

            assistant_token_spans = []
            for k in range(len(pred_thoughts)):
                if k in first_tok:
                    assistant_token_spans.append((first_tok[k], last_tok[k] + 1))
                else:
                    # Segment khong chiem token nao -> span rong, diem IG = 0.
                    assistant_token_spans.append((0, 0))

            # Shift segment spans by user prompt length
            offset = len(user_tokens)
            adjusted_spans = [(start + offset, end + offset) if end > start else (0, 0)
                              for (start, end) in assistant_token_spans]

            answer_string = "</think> So, the final answer is \\boxed{" + each_data['answer'] + "}"
            
            answer_tokens = attribution_calculator.tokenizer(answer_string, add_special_tokens=False)["input_ids"] 
            answer_tokens_split = attribution_calculator.tokenizer.convert_ids_to_tokens(answer_tokens)

            for token_index in range(len(answer_tokens_split)):
                if 'boxed' in answer_tokens_split[token_index]:
                    ans_start = token_index + 1
            assert '{' in answer_tokens_split[ans_start]
            ans_end = len(answer_tokens_split)
            is_end = False
            stack = 0
            for m in range(ans_start, len(answer_tokens_split)):
                if '{' in answer_tokens_split[m] or '}' in answer_tokens_split[m]:
                    for each_tok in answer_tokens_split[m]:
                        if each_tok == "{":
                            stack += 1
                        elif each_tok == "}":
                            stack -= 1
                            if stack == 0:
                                ans_end = m #+1
                                is_end = True
                                break
                    if is_end:
                        break
            
            full_tokens = user_tokens + assistant_tokens + answer_tokens
            answer_indices = (ans_start + 1 + len(user_tokens + assistant_tokens), ans_end + len(user_tokens + assistant_tokens))

            # Diem 0 cho moi segment, giu dung so segment de file IG van thang
            # hang voi file segment (get_important_segments.py assert dieu nay).
            zero_scores = [[0.0] * max(0, e - st) for (st, e) in adjusted_spans]

            if args.max_input_tokens and len(full_tokens) > args.max_input_tokens:
                print("  bo qua mau %d: %d token > --max_input_tokens %d"
                      % (n, len(full_tokens), args.max_input_tokens))
                importance_scores = zero_scores
                skipped_long += 1
            else:
                try:
                    importance_scores = attribution_calculator.batch_compute_step_to_answer_attribution_integrated(
                        full_tokens, adjusted_spans, answer_indices,
                        baseline_token_id=attribution_calculator.tokenizer.pad_token_id,
                        steps=args.ig_steps)
                except OOM_ERROR:
                    # Mot mau qua dai khong duoc lam chet ca job nhieu gio. Gan
                    # diem 0 -> train_mask.py roi ve 3 segment mac dinh cho mau nay.
                    print("  OOM o mau %d (%d token), gan diem 0 va chay tiep"
                          % (n, len(full_tokens)))
                    attribution_calculator.model.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    importance_scores = zero_scores
                    skipped_oom += 1

            input_data[n]["attribution"] = importance_scores

            f.write(json.dumps(input_data[n], ensure_ascii=False) + '\n') 

   
    # get_important_segments.py chi can 3 so moi segment chu khong can diem tung
    # token: Strength = sum|IG| / sqrt(N), Consistency = |sum IG| / sum|IG|.
    # Nen ngoai file IG day du, ghi them ban compact nho hon ~30 lan - du de
    # chay tiep pipeline va du nho de tai ve.
    compact_path = args.output_compact_file
    if not compact_path:
        base = args.output_ig_file
        compact_path = (base[:-6] if base.endswith(".jsonl") else base) + "_compact.jsonl"

    # Doc theo tung dong thay vi nap ca file vao bo nho: file nay co the vai
    # tram MB.
    n_rows = 0
    with open(args.output_data_file, 'r') as fin, \
         open(args.output_ig_file, "w") as f_full, \
         open(compact_path, "w") as f_small:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            attribution = json.loads(line)["attribution"]
            f_full.write(json.dumps(attribution, ensure_ascii=False) + '\n')

            compact = []
            for seg in attribution:
                n_tok = len(seg)
                sum_abs = float(np.sum(np.abs(seg))) if n_tok else 0.0
                sum_signed = float(np.sum(seg)) if n_tok else 0.0
                compact.append([n_tok, round(sum_abs, 8), round(sum_signed, 8)])
            f_small.write(json.dumps({"segments": compact}) + '\n')
            n_rows += 1

    print("Da ghi %d mau" % n_rows)
    print("  IG day du : %s (%.1f MB)"
          % (args.output_ig_file, os.path.getsize(args.output_ig_file) / 1e6))
    print("  IG compact: %s (%.1f MB)  <- dung file nay cho get_important_segments"
          % (compact_path, os.path.getsize(compact_path) / 1e6))

    if skipped_oom or skipped_long:
        print("CANH BAO: %d mau OOM, %d mau vuot --max_input_tokens -> deu duoc gan diem 0. "
              "Nhung mau nay se chi hoc 3 segment mac dinh (dau/gan cuoi/cuoi)."
              % (skipped_oom, skipped_long))
 
    