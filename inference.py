# inference.py
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    apply_chat_template_safe, msgs_zero_shot, msgs_cot, msgs_few_shot, msgs_thinking,
    chatml_think_prefill_prompt, load_gsm8k, run_generation, save_jsonl, save_txt,
)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--mode", choices=["zero_shot", "cot", "few_shot", "thinking", "chatml_think_prefill"], default="zero_shot")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--few-shot-k", type=int, default=3)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--output", default="outputs/inference.jsonl")
    p.add_argument("--save-txt", action="store_true")
    args = p.parse_args()

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    print(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ds, test = load_gsm8k(limit=args.limit)
    questions = [ex["question"] for ex in test]
    refs = [ex["answer"] for ex in test]

    prompts = []
    if args.mode == "zero_shot":
        for q in questions:
            msgs = msgs_zero_shot(q)
            prompts.append(apply_chat_template_safe(tokenizer, msgs, add_generation_prompt=True))
    elif args.mode == "cot":
        for q in questions:
            msgs = msgs_cot(q)
            prompts.append(apply_chat_template_safe(tokenizer, msgs, add_generation_prompt=True))
    elif args.mode == "few_shot":
        train = ds["train"]
        examples = [(train[i]["question"], train[i]["answer"]) for i in range(min(args.few_shot_k, len(train)))]
        for q in questions:
            msgs = msgs_few_shot(q, examples)
            prompts.append(apply_chat_template_safe(tokenizer, msgs, add_generation_prompt=True))
    elif args.mode == "thinking":
        for q in questions:
            msgs = msgs_thinking(q)
            prompts.append(apply_chat_template_safe(tokenizer, msgs, add_generation_prompt=True, enable_thinking=True))
    elif args.mode == "chatml_think_prefill":
        for q in questions:
            prompts.append(chatml_think_prefill_prompt(q))
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print(f"Running inference on {len(prompts)} examples...")
    outs, t = run_generation(
        model, tokenizer, prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample or (args.mode == "thinking"),
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    qps = len(prompts) / max(t, 1e-9)
    print("========== REPORT ==========")
    print(f"Total: {len(prompts)} | Time: {t:.2f}s | Throughput: {qps:.2f} q/s")
    print("============================")

    save_jsonl(args.output, questions, outs, refs)
    print(f"Saved: {args.output}")
    if args.save_txt:
        txt_path = args.output.replace(".jsonl", ".txt")
        save_txt(txt_path, outs)
        print(f"Saved: {txt_path}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()