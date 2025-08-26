# Qwen3-0.6B on GSM8K — Benchmarks

Fast, reproducible experiments on GSM8K with Qwen3-0.6B:
- Prompting modes: direct answer, chain-of-thought (CoT), few-shot CoT, and “thinking” with `<think>` tags
- Full SFT and LoRA fine-tuning (ChatML + `<think>` prefill)

## Results and insights

Prompting strategies (GSM8K test)

<img width="380" height="514" alt="image" src="https://github.com/user-attachments/assets/742ab578-a373-4c99-86a0-7af3e1ef6bd7" />

- Direct Answer: 8.18%
- Chain of Thought (CoT): 54.13%
- Few-shot with CoT: 43.06%

Takeaway: CoT boosts accuracy substantially over direct answers. Few-shot CoT underperformed plain CoT—likely because a 650M-parameter model is more easily distracted by extra examples than larger LLMs.

Effect of chat format and think tags

<img width="380" alt="image" src="https://github.com/user-attachments/assets/6e7ad4ea-60e5-4aa7-8d91-afee39b1ce89" />

- CoT without `<think>`: 54.1%
- CoT with `<think>`: 66.7%

Takeaway: After SFT, the model is aligned to produce reasoning inside `<think>...</think>`. Prompting with those tags at inference improves accuracy. Smaller model appears to be especially sensitive to the template details.

LoRA vs full fine-tuning on GSM8K

<img width="380" alt="image" src="https://github.com/user-attachments/assets/7984f8e6-d3f3-4d85-81f0-faaa9e9a1dbc" />

- Full fine-tuning: 48.1%
- LoRA rank 4: 44.65%
- LoRA rank 32: 43.21%

Takeaway:
- Full SFT outperformed LoRA variants in these runs; changing LoRA rank didn’t materially change accuracy.
- Counter-intuitively, full SFT sometimes underperformed the base model on direct-answer inference. A plausible reason: the base model’s pre-trained “thinking style” was stronger than GSM8K’s reasoning traces, so narrow SFT disrupted useful priors.

## Models on the Hugging Face Hub

Profile: https://huggingface.co/AnirudhCheruvu

- LoRA r4 (FP32): https://huggingface.co/AnirudhCheruvu/Qwen3-GSM8k-LoRA-r4-FP32-eval-score
- LoRA r32 (FP32): https://huggingface.co/AnirudhCheruvu/Qwen3-GSM8k-LoRA-r32-FP32-eval-score
- Full SFT (FP32, eval-score): https://huggingface.co/AnirudhCheruvu/Qwen3-GSM8k-SFT-FP32-eval-score
- Full SFT (BF16): https://huggingface.co/AnirudhCheruvu/Qwen3-GSM8k-SFT-BF16
- Full SFT (FP32): https://huggingface.co/AnirudhCheruvu/Qwen3-GSM8k-SFT-FP32

## Medium article

- Medium: https://medium.com/@anirudhcheruvu2014/experiments-on-qwen3-0-6b-f531d0291f8f
## Quickstart

Install (uses this repo’s `requirements.txt` as-is):
```bash
pip install -r requirements.txt
```

Run inference (writes JSONL to `outputs/`):
```bash
# Zero-shot (direct answer)
python inference.py --model-id Qwen/Qwen3-0.6B \
  --mode zero_shot --batch-size 32 \
  --output outputs/zero_shot.jsonl

# Chain-of-Thought (CoT)
python inference.py --model-id Qwen/Qwen3-0.6B \
  --mode cot --batch-size 16 \
  --output outputs/cot.jsonl

# Few-shot (+ CoT), k=3 examples from GSM8K train
python inference.py --model-id Qwen/Qwen3-0.6B \
  --mode few_shot --few-shot-k 3 \
  --output outputs/few_shot_cot.jsonl

# Thinking mode (enable_thinking + sampling recommended)
python inference.py --model-id Qwen/Qwen3-0.6B \
  --mode thinking --do-sample --temperature 0.6 --top-p 0.95 --top-k 20 \
  --output outputs/thinking.jsonl

# ChatML prefill up to <think> (mirrors training template)
python inference.py --model-id Qwen/Qwen3-0.6B \
  --mode chatml_think_prefill \
  --output outputs/chatml_prefill.jsonl
```

Benchmark accuracy (GSM8K “#### answer” parsing with numeric tolerance):
```bash
python benchmark.py outputs/zero_shot.jsonl outputs/cot.jsonl outputs/few_shot_cot.jsonl outputs/thinking.jsonl
```

Evaluate uploaded checkpoints (examples):
```bash
# SFT FP32 with CoT
python inference.py --model-id AnirudhCheruvu/Qwen3-GSM8k-SFT-FP32-eval-score \
  --mode cot --output outputs/sft_fp32_cot.jsonl

# LoRA r4 in thinking mode
python inference.py --model-id AnirudhCheruvu/Qwen3-GSM8k-LoRA-r4-FP32-eval-score \
  --mode thinking --do-sample --temperature 0.6 --top-p 0.95 --top-k 20 \
  --output outputs/lora_r4_thinking.jsonl

# LoRA r32 with few-shot CoT (k=3)
python inference.py --model-id AnirudhCheruvu/Qwen3-GSM8k-LoRA-r32-FP32-eval-score \
  --mode few_shot --few-shot-k 3 --output outputs/lora_r32_fewshot_cot.jsonl
```

## Repository structure

- `common.py` — shared helpers (prompt builders, ChatML/`<think>` prefill, batching, scoring, I/O)
- `inference.py` — batched generation for modes: `zero_shot`, `cot`, `few_shot`, `thinking`, `chatml_think_prefill`
- `benchmark.py` — accuracy scorer for JSONL outputs
- `train.py` — full SFT on GSM8K using ChatML + `<think>` prefill
- `train_lora.py` — LoRA SFT (configurable rank and target modules)
- `requirements.txt` — dependencies

## Results artifacts (JSONL)

Include the evaluation outputs below for severa experimental settigns.

- `results/jsonl/`
  - `zero_shot_model_outputs.jsonl`
  - `cot_model_outputs.jsonl`
  - `few_shot_model_outputs.jsonl`
  - `Updated_few_shot_model_outputs.jsonl`
  - `thinking_model_outputs_reduced_batch.jsonl`
  - `BF16-thinking_model_outputs_reduced_batch.jsonl`
  - `Qwen3-GSM8k-SFT-FP32-eval-score.jsonl`
  - `Qwen3-GSM8k-SFT-FP32-temp0.jsonl`
  - `Qwen3-GSM8k-SFT-FP32-temp06.jsonl`
  - `Qwen3-GSM8k-SFT-BF16-temp0.jsonl`
  - `Qwen3-GSM8k-LoRA-r4-FP32-eval-score`
  - `Qwen3-GSM8k-LoRA-r32-FP32-eval-score`

Schema used by the scripts:
- Each line is a JSON object with `{"id","question","model_output","ground_truth_answer"}`.
- Accuracy is computed by extracting the last number after `####` from both model output and ground truth and comparing as floats with tolerance 1e-4.

## Implementation notes

- Prompt formatting
  - CoT prompts ask for reasoning and then a final line `#### <answer>` (no units/symbols).
  - The “thinking” path enables `<think>` reasoning during generation and mirrors the ChatML template used in training.
- Efficiency
  - Left padding for batched decoding; throughput stats printed per run.

## Requirements

`requirements.txt`:
- datasets
- trl
- transformers
- tensorboard
- peft
- tqdm
- accelerate


## Contributing

Issues and PRs are welcome. 

## Acknowledgements

Thanks to the Qwen team and the open-source ecosystem for tools like `transformers`, `trl`, `peft`, and `datasets`.
