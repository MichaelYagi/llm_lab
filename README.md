# Local LLM Fine‑Tuning (TinyLlama + LoRA)

This project fine‑tunes a small LLM (TinyLlama‑1.1B) locally using LoRA on a single NVIDIA GPU inside WSL2. It includes training, inference, GPU profiling, LoRA merging, and GGUF export for llama.cpp / ollama.

## Requirements

Hardware:
- NVIDIA GPU (RTX 3060 12GB recommended)
- 16–32GB RAM
- 10–20GB free disk space

Software:
- Windows 10/11 with WSL2
- Debian or Ubuntu inside WSL2
- Latest NVIDIA GPU driver on Windows
- Python 3.10+

Verify GPU passthrough:
nvidia-smi

## Setup

Create and activate a virtual environment:
python3 -m venv .venv
source .venv/bin/activate

Install PyTorch with CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cu121

Install project dependencies:
pip install -r requirements.txt

## Project Structure

llm-lab/
  data/
    alpaca_tiny.json
  models/
    tinyllama-lora/
    tinyllama-merged/
  train.py
  infer.py
  merge_lora.py
  export_gguf.py
  Makefile
  requirements.txt
  README.md

## Train the Model

make train

## Run Inference

make infer

Run inference with sampling parameters:
make infer-sample temp=0.7 top_p=0.9 max=200

## GPU Memory Profiling

make profile

## Merge LoRA Into a Standalone Model

make merge

## Export to GGUF (llama.cpp / ollama)

make gguf

## Troubleshooting

Check CUDA:
python3 - << 'EOF'
import torch
print(torch.cuda.is_available())
EOF

Reduce memory usage by lowering:
- per_device_train_batch_size
- max_length
- max_steps

## Next Steps

Once TinyLlama works, scale to:
- Phi‑2 (2.7B)
- Qwen‑1.5B
- Mistral‑7B (QLoRA)
