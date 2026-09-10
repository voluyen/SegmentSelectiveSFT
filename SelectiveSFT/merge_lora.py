"""Gop adapter LoRA vao weight goc -> mot thu muc model day du.

Can vi eval chay bang vLLM va math_eval.py nhan --model_name_or_path nhu mot
model binh thuong, khong biet gi ve adapter. Chay trong env TRAIN (ssft_train)
vi chi env do co peft; khong can GPU.

  python merge_lora.py --adapter checkpoints/<run>/checkpoint-75
  # -> checkpoints/<run>/checkpoint-75-merged
"""
import argparse
import json
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, type=str, help="Thu muc checkpoint chua adapter_config.json")
    p.add_argument("--output_dir", default="", type=str, help="De trong = <adapter>-merged")
    p.add_argument("--base_model", default="", type=str, help="De trong = doc tu adapter_config.json")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device_map", default="cpu", type=str, help="cpu (mac dinh, an toan) hoac auto")
    return p.parse_args()


def main():
    args = parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg_path = os.path.join(args.adapter, "adapter_config.json")
    if not os.path.exists(cfg_path):
        raise SystemExit("Khong thay %s - thu muc nay khong phai adapter LoRA." % cfg_path)
    with open(cfg_path) as f:
        adapter_cfg = json.load(f)

    base = args.base_model or adapter_cfg.get("base_model_name_or_path")
    if not base:
        raise SystemExit("Khong xac dinh duoc base model, truyen --base_model.")
    out = args.output_dir or (args.adapter.rstrip("/") + "-merged")

    print("base model : %s" % base)
    print("adapter    : %s" % args.adapter)
    print("output     : %s" % out)

    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=getattr(torch, args.dtype),
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    # Tokenizer lay tu adapter neu co (truong hop da them token), khong thi tu base.
    tok_src = args.adapter if os.path.exists(os.path.join(args.adapter, "tokenizer_config.json")) else base
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        print("Canh bao: embedding %d != vocab tokenizer %d"
              % (model.get_input_embeddings().weight.shape[0], len(tokenizer)))

    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    print("Xong. Eval bang: bash eval.sh --model %s" % os.path.abspath(out))


if __name__ == "__main__":
    main()
