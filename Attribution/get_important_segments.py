import json
import math
import numpy as np
import random
import argparse
import re
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_data_file", type=str, required=True, help="Path to input jsonl")
    p.add_argument("--IG_score_data_file", type=str, required=True, help="Path to input IG scores")
    p.add_argument("--output_data_file", type=str, required=True, help="Path to output jsonl (appended)")
    p.add_argument("--cumulative_ratio", type=float, default=0.7,
                   help="Giu segment manh nhat den khi dat ty le nay cua tong IG (top70 trong ten file goc)")
    p.add_argument("--coherence_max", type=float, default=0.8,
                   help="Bo segment co |sum IG| / sum|IG| lon hon nguong nay (cohe80 trong ten file goc)")
    return p.parse_args()

args = parse_args()

input_data = []
with open(args.input_data_file, "r") as f: 
    for line in f:
        json_obj = json.loads(line.strip())  
        input_data.append(json_obj)

all_IG_list = []
with open(args.IG_score_data_file, "r") as f: 
    for line in f:
        json_obj = json.loads(line.strip())  
        all_IG_list.append(json_obj)
print("sample number", len(input_data), len(all_IG_list))


ratio_list = []
for i in range(len(input_data)):
    assert len(all_IG_list[i]) == len(input_data[i]["segments"]), f"{i} {len(all_IG_list[i])}. {len(input_data[i]['segments'])}"
    cur_IGs = all_IG_list[i]
        
    all_segs_IG_stres = []
    for each_seg in cur_IGs:
        abs_each_seg = [abs(_) for _ in each_seg]
        # Chia theo "\n\n" tao ra nhieu segment rat ngan; segment 0 token se
        # lam phep chia cho len**0.5 ra nan va keo hong ca thu tu sap xep.
        if len(abs_each_seg) == 0:
            all_segs_IG_stres.append(0.0)
        else:
            all_segs_IG_stres.append(np.sum(abs_each_seg)/(len(abs_each_seg)**0.5))
    indexed_sorted = sorted(enumerate(all_segs_IG_stres), key=lambda x: -x[1])
    sorted_indices = [idx for idx, val in indexed_sorted]
    sorted_inst_IG_stre = np.array([val for idx, val in indexed_sorted])

    total_stre = sorted_inst_IG_stre.sum()
    if total_stre <= 0:
        # Khong co tin hieu attribution nao -> khong chon segment nao, de
        # train_mask.py roi ve 3 segment mac dinh (dau / gan cuoi / cuoi).
        input_data[i]['selected_spans_ids'] = []
        ratio_list.append(0.0)
        continue
    normed = sorted_inst_IG_stre / total_stre
    cumsum = np.cumsum(normed)
    for j in range(len(cumsum)):
        if cumsum[j] >= args.cumulative_ratio:
            break
    important_index = sorted(sorted_indices[:j+1])
        
    IG_dire_list = []
    for _ in range(len(input_data[i]["segments"])):
        denom = np.sum(np.abs(cur_IGs[_]))
        # denom == 0 khi segment rong hoac toan bo IG bang 0 -> coi nhu khong
        # coherent (1.0) de bo qua, thay vi 0/0 = nan (nan <= 0.8 la False,
        # dung ngau nhien nhung khong hien y do).
        IG_dire_list.append(abs(np.sum(cur_IGs[_])) / denom if denom > 0 else 1.0)

    select_span_ids = [_ for _ in important_index if IG_dire_list[_] <= args.coherence_max]
    assert select_span_ids == sorted(select_span_ids)
    
    input_data[i]['selected_spans_ids'] = select_span_ids
    ratio_list.append(len(select_span_ids)/len(cur_IGs))
        
with open(args.output_data_file, 'w') as f:
    for n in range(len(input_data)):
        f.write(json.dumps(input_data[n], ensure_ascii=False) + '\n')
print("Ty le segment duoc chon: trung binh %.3f" % float(np.mean(ratio_list)))
print("Da ghi: %s" % args.output_data_file)



