# common.py
import os
import re
import json
import time
from typing import List, Dict, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------- Evaluation helpers ----------
def extract_answer(text: str) -> Optional[str]:
    if text is None:
        return None
    text = text.split("####")[-1].strip()
    numbers = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*\.?\d*|\.\d+", text)
    if not numbers:
        return None
    return numbers[-1].replace(",", "")

def is_correct(model_output: str, ground_truth: str) -> bool:
    m = extract_answer(model_output)
    t = extract_answer(ground_truth)
    if m is None or t is None:
        return False
    try:
        return abs(float(m) - float(t)) < 1e-4
    except ValueError:
        return False

def compute_accuracy(outputs: List[str], refs: List[str]) -> float:
    correct = 0
    for o, r in zip(outputs, refs):
        if is_correct(o, r):
            correct += 1
    return correct / max(1, len(refs))


# ---------- Tokenizer/template helpers ----------
def apply_chat_template_safe(
    tokenizer,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True,
    enable_thinking: Optional[bool] = None,
) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    if enable_thinking is not None:
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)
    return tokenizer.apply_chat_template(messages, **kwargs)


# ---------- Prompt builders ----------
def msgs_zero_shot(q: str) -> List[Dict[str, str]]:
    system = ("You are a helpful assistant that solves math problems. Just provide the final numerical answer "
              "in the format '#### <answer>' without any units, symbols or explanations.")
    user = f"{q}\n\nProvide only the numerical answer after '####': #### "
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def msgs_cot(q: str) -> List[Dict[str, str]]:
    system = ("You are a detailed math assistant. Provide a step-by-step explanation, then on a new line the final "
              "numerical answer as '#### <answer>' with no symbols or units.")
    user = (f"{q}\n\nPlease think step-by-step and then write the final numerical answer, "
            "with no symbols or units, in the format '#### <answer>'.")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def msgs_few_shot(q: str, examples: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    system = ("You are a math assistant. Provide a step-by-step explanation, then on a new line the final "
              "numerical answer as '#### <answer>' with no symbols or units.")
    few = ["### Examples ###"]
    for exq, exa in examples:
        few.append(f"Question: {exq}\nAnswer: {exa}\n")
    sys_with_examples = system + "\n\n" + "\n".join(few)
    user = f"Question: {q}\nAnswer:"
    return [{"role": "system", "content": sys_with_examples}, {"role": "user", "content": user}]

def msgs_thinking(q: str) -> List[Dict[str, str]]:
    system = ("You are a math assistant. Provide a step-by-step explanation, then on a new line the final "
              "numerical answer as '#### <answer>' with no symbols or units.")
    user = f"Question: {q}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def chatml_think_prefill_prompt(q: str, system_prompt: Optional[str] = None) -> str:
    sys_msg = system_prompt or (
        "You are a detailed math assistant. Provide a step-by-step explanation, then on a new line the final "
        "numerical answer as '#### <answer>' with no symbols or units. The answer format is very important."
    )
    tpl = (
        "<|im_start|>system\n{system_message}<|im_end|>\n"
        "<|im_start|>user\n{user_message}<|im_end|>\n"
        "<|im_start|>assistant\n<think>"
    )
    return tpl.format(system_message=sys_msg, user_message=q)


# ---------- Data loading ----------
def load_gsm8k(limit: int = 0):
    ds = load_dataset("openai/gsm8k", "main")
    test = ds["test"]
    if limit and limit > 0:
        test = test.select(range(min(limit, len(test))))
    return ds, test


# ---------- Generation ----------
def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def run_generation(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 32,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
) -> Tuple[List[str], float]:
    model.eval()
    outputs = []
    total_t = 0.0

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(model.device)

        cuda_sync()
        t0 = time.time()
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                top_k=top_k if do_sample else None,
                pad_token_id=tokenizer.pad_token_id
            )
        cuda_sync()
        total_t += (time.time() - t0)

        in_len = inputs["input_ids"].shape[1]
        new_tokens = gen_ids[:, in_len:]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        outputs.extend([t.strip() for t in texts])

    return outputs, total_t


# ---------- I/O ----------
def save_jsonl(path: str, questions: List[str], outputs: List[str], refs: List[str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, (q, o, r) in enumerate(zip(questions, outputs, refs)):
            f.write(json.dumps({
                "id": i,
                "question": q,
                "model_output": o,
                "ground_truth_answer": r
            }) + "\n")

def save_txt(path: str, outputs: List[str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(o + "\n")