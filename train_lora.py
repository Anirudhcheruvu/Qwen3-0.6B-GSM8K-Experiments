# train_lora.py
from typing import Dict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback
from peft import LoraConfig, get_peft_model
import re

SYSTEM_PROMPT = (
    "You are a detailed math assistant. For each problem, first, provide a clear, step-by-step explanation of your reasoning. "
    "After your explanation, provide the final numerical answer on a new line in the format '#### <answer>', without any special symbols or units. "
    "The answer format is very important."
)
CHAT_TEMPLATE = (
    "<|im_start|>system\n{system_message}<|im_end|>\n"
    "<|im_start|>user\n{user_message}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n{assistant_thinking}\n</think>\n{assistant_answer}<|im_end|>\n"
)

def process_to_chatml(example: Dict) -> Dict:
    ans = example["answer"]
    idx = ans.find("####")
    assistant_thinking = ans[:idx].strip() if idx != -1 else ""
    assistant_answer = ans[idx:].strip() if idx != -1 else ans.strip()
    text = CHAT_TEMPLATE.format(
        system_message=SYSTEM_PROMPT,
        user_message=example["question"],
        assistant_thinking=assistant_thinking,
        assistant_answer=assistant_answer,
    )
    key = "<|im_start|>assistant\n<think>"
    klen = len(key)
    ka = text.find(key)
    prompt = text[:ka + klen].strip()
    completion = text[ka + klen :].strip()
    return {"prompt": prompt, "completion": completion}

def add_completion_mask(example, tokenizer):
    p_tok = tokenizer(example["prompt"])
    c_tok = tokenizer(example["completion"])
    example["input_ids"] = p_tok["input_ids"] + c_tok["input_ids"]
    example["attention_mask"] = p_tok["attention_mask"] + c_tok["attention_mask"]
    prompt_mask = [0] * len(p_tok["input_ids"])
    comp_mask = [1] * len(c_tok["input_ids"])
    example["completion_mask"] = prompt_mask + comp_mask
    return example

class GenerationEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, max_new_tokens=1024, num_samples=5, batch_size=8):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.max_new_tokens = max_new_tokens
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.scores = []

    def extract_answer(self, text: str):
        text = text.split("####")[-1].strip()
        nums = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*\.?\d*|\.\d+", text)
        if not nums:
            return None
        return nums[-1].replace(",", "")

    def is_correct(self, a: str, b: str):
        if a is None or b is None:
            return False
        try:
            return abs(float(a) - float(b)) < 1e-4
        except ValueError:
            return False

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()
        tok = self.tokenizer
        tok.padding_side = "left"
        self.scores = []

        prompts = self.eval_dataset["prompt"][: self.num_samples]
        gts = self.eval_dataset["completion"][: self.num_samples]

        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]
            mi = tok(batch, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    input_ids=mi["input_ids"],
                    attention_mask=mi["attention_mask"],
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            for j, ids in enumerate(out):
                text = tok.decode(ids, skip_special_tokens=False)
                st = text.find("<|im_start|>assistant")
                gen = text[st:].strip() if st != -1 else text.strip()
                pred = self.extract_answer(gen)
                true = self.extract_answer(gts[min(i + j, len(gts) - 1)])
                self.scores.append(1.0 if self.is_correct(pred, true) else 0.0)

        avg = sum(self.scores) / len(self.scores) if self.scores else 0.0
        kwargs.get("metrics", {})["eval_score"] = avg
        self.trainer.log({"eval_score": avg})
        if hasattr(state, "log_history") and state.log_history:
            state.log_history[-1]["eval_score"] = avg

def main():
    model_name = "Qwen/Qwen3-0.6B"
    output_dir = "./qwen3-gsm8k-lora-r4"
    logging_dir = "./logs"

    base = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("openai/gsm8k", "main")
    train = ds["train"].map(process_to_chatml)
    eval_ds = ds["test"].map(process_to_chatml)

    train_tok = train.map(lambda ex: add_completion_mask(ex, tokenizer),
                          remove_columns=["question", "answer", "prompt", "completion"])
    eval_tok = eval_ds.map(lambda ex: add_completion_mask(ex, tokenizer),
                           remove_columns=["question", "answer", "prompt", "completion"])

    sft_cfg = SFTConfig(
        report_to="tensorboard",
        output_dir=output_dir,
        logging_dir=logging_dir,
        num_train_epochs=6,
        learning_rate=1e-6,
        lr_scheduler_type="cosine",
        dataset_kwargs={"skip_prepare_dataset": True},
        completion_only_loss=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        load_best_model_at_end=True,
        metric_for_best_model="eval_score",
        greater_is_better=True,
    )

    eval_cb = GenerationEvalCallback(
        tokenizer=AutoTokenizer.from_pretrained(model_name),
        eval_dataset=eval_ds.select(range(min(130, len(eval_ds)))),
        max_new_tokens=1024,
        num_samples=5,
        batch_size=16,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        args=sft_cfg,
        callbacks=[eval_cb, EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.0)],
    )

    eval_cb.trainer = trainer
    trainer.train()

if __name__ == "__main__":
    main()