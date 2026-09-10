"""Tai simplescaling/s1K-1.1 va doi ve dung format ma pipeline nay can.

Pipeline doc jsonl voi 3 truong: question / solution / answer, trong do
`solution` la long-CoT trace con `answer` la dap an tran (grad_analyze.py ghep
no vao "\\boxed{...}" de lam target attribution).

Map tu s1K-1.1:
  question -> question
  solution <- deepseek_thinking_trajectory   (trace R1, giong dang cua LIMO:
                                              ket thuc bang **Final Answer** \\boxed{...})
  answer   <- \\boxed{...} cuoi cung trong trace, fallback ve truong `solution`
              cua s1K (voi AIME/TheoremQA day la dap an tran san)
"""
import argparse
import json
import os
import re


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="simplescaling/s1K-1.1", type=str)
    p.add_argument("--split", default="train", type=str)
    p.add_argument("--trace_field", default="deepseek_thinking_trajectory", type=str,
                   help="Truong chua long-CoT trace (doi sang gemini_thinking_trajectory neu muon)")
    p.add_argument("--output_data_file", default="data/s1k/train.jsonl", type=str)
    p.add_argument("--answer_source", default="trace", choices=["trace", "gt"],
                   help="trace = \\boxed{} cuoi trace (target attribution khop voi cai trace tu ket luan); "
                        "gt = truong `solution` cua s1K (dap an chuan, co the lech voi trace)")
    p.add_argument("--max_samples", default=0, type=int,
                   help="0 = lay het. >0 = chi lay ngan nay mau, de chay thu ca duong ong cho nhanh")
    p.add_argument("--max_answer_chars", default=200, type=int,
                   help="Bo mau co dap an dai hon nguong nay - target \\boxed{} qua dai thi IG vo nghia")
    return p.parse_args()


def last_boxed(text):
    """Lay noi dung \\boxed{...} cuoi cung, dem ngoac de khong cat nham."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def main():
    args = parse_args()
    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split)
    print("Tai xong %s: %d mau, cot = %s" % (args.dataset, len(ds), ds.column_names))

    rows, skipped, disagree = [], {"no_trace": 0, "no_answer": 0, "answer_too_long": 0}, 0
    for ex in ds:
        trace = (ex.get(args.trace_field) or "").strip()
        if not trace:
            skipped["no_trace"] += 1
            continue

        gt = (ex.get("solution") or "").strip()
        if args.answer_source == "gt":
            answer = gt or last_boxed(trace)
        else:
            answer = last_boxed(trace) or gt
        if not answer:
            skipped["no_answer"] += 1
            continue
        if len(answer) > args.max_answer_chars:
            skipped["answer_too_long"] += 1
            continue

        boxed = last_boxed(trace)
        if boxed is not None and gt and boxed != gt:
            disagree += 1

        rows.append({
            "question": ex["question"].strip(),
            "solution": trace,
            "answer": answer,
        })

    if args.max_samples > 0:
        rows = rows[: args.max_samples]
        print("Cat con %d mau dau (--max_samples)" % len(rows))

    out = args.output_data_file
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Giu %d mau, bo %d (%s)" % (len(rows), sum(skipped.values()), skipped))
    print("Trace ket luan khac dap an chuan: %d mau (dang dung --answer_source=%s)"
          % (disagree, args.answer_source))
    print("Da ghi: %s" % out)
    if rows:
        lens = [len(r["solution"]) for r in rows]
        print("Do dai trace (ky tu): min=%d trung binh=%d max=%d"
              % (min(lens), sum(lens) // len(lens), max(lens)))


if __name__ == "__main__":
    main()
