"""Chia solution thanh segment, ghi them truong `segments` vao jsonl.

Truoc day file nay hard-code tokenizer '../../models/DeepSeek-R1-Distill-Qwen-7B'
va duong dan LIMO, khong nhan tham so nao - run_pipeline.sh phai sed va mot ban
runtime tam. Gio nhan tham so truc tiep.
"""
import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from segment_utils import DEFAULT_MODE, SEGMENT_PATTERNS, split_segments  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_data_file", default="../data/s1k/train.jsonl", type=str)
    p.add_argument("--output_data_file", default="../data/s1k/solution_segments.jsonl", type=str)
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct", type=str,
                   help="Chi dung de bao cao do dai trace, khong anh huong cach chia. "
                        "Dat 'none' de bo qua thong ke va khong can transformers.")
    p.add_argument("--segment_mode", default=DEFAULT_MODE, choices=sorted(SEGMENT_PATTERNS),
                   help="paragraph = cat tai moi '\\n\\n'; cue = cat tai tu khoa backtracking (cach cua paper)")
    return p.parse_args()


def main():
    args = parse_args()

    # Viec chia segment thuan tuy la xu ly chuoi; tokenizer chi de in thong ke
    # do dai. Import muon de stage nay chay duoc ca o moi truong khong co torch.
    tokenizer = None
    if args.tokenizer.lower() not in ("none", ""):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    input_data = []
    with open(args.input_data_file, "r") as f:
        for line in f:
            input_data.append(json.loads(line.strip()))

    segment_num, all_len = [], []
    for i, each_data in tqdm(enumerate(input_data), total=len(input_data)):
        cur_response = each_data["solution"]
        if tokenizer is not None:
            all_len.append(len(tokenizer(cur_response, add_special_tokens=False)["input_ids"]))

        segments = split_segments(cur_response, args.segment_mode)
        segment_num.append(len(segments))
        input_data[i]["segments"] = segments

    print("che do chia: %s" % args.segment_mode)
    print("so mau: %d" % len(input_data))
    print("segment/mau: trung binh %.1f, min %d, max %d"
          % (float(np.mean(segment_num)), int(np.min(segment_num)), int(np.max(segment_num))))
    if all_len:
        print("token/trace: trung binh %d, max %d" % (int(np.mean(all_len)), int(np.max(all_len))))

    os.makedirs(os.path.dirname(os.path.abspath(args.output_data_file)), exist_ok=True)
    with open(args.output_data_file, "w") as f:
        for item in input_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Da ghi: %s" % args.output_data_file)


if __name__ == "__main__":
    main()
