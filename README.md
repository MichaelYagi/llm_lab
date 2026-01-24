# LLM Lab — TinyLlama LoRA Fine‑Tuning Pipeline

A clean, local, and reproducible pipeline for fine‑tuning **TinyLlama** using **LoRA**, testing the adapted model, merging weights, and exporting to **GGUF** for use with llama.cpp and other local runtimes.

This repository is designed to be small, understandable, and easy to extend.

---

## 🚀 Features

- Fine‑tune **TinyLlama/TinyLlama-1.1B-Chat-v1.0** with **LoRA**
- Uses Hugging Face `transformers`, `datasets`, `peft`, and `accelerate`
- Simple Alpaca‑style JSON dataset included for quick experiments
- Inference script to test the fine‑tuned model
- Merge LoRA adapters into a standalone model
- Export merged model to **GGUF** for llama.cpp and similar tools
- Makefile‑driven workflow for consistent, one‑command operations

---

## 📦 Requirements

- Python 3.10+ (3.12 tested)
- A CUDA‑capable GPU (e.g., RTX 3060 or better)
- `python3-venv`
- CMake 3.14+ (for llama.cpp)

Dependencies are listed in `requirements.txt`.

---

## 🛠️ Installation

### Option 1: Using the Makefile
```bash
make install
```

This will:

-   Create `.venv/`
-   Upgrade `pip`
-   Install PyTorch (CUDA build)
-   Install all dependencies

Activate manually if needed:
```bash
source .venv/bin/activate
```

### Option 2: Manual installation
```bash
python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python3 -m pip install -r requirements.txt
```

### Installing llama.cpp (for GGUF conversion)
```bash
# Clone llama.cpp (once, outside this repo)
cd /mnt/c/Users/Michael/PycharmProjects
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build with CMake
cmake -B build
cmake --build build --config Release -j$(nproc)

# Install Python dependencies in your llm_lab venv
cd /mnt/c/Users/Michael/PycharmProjects/llm_lab
source .venv/bin/activate
pip install transformers sentencepiece protobuf gguf
```

---

## 📁 Project Structure
```
llm_lab/
│
├── train.py              # Fine‑tune TinyLlama with LoRA
├── infer.py              # Run inference with the fine‑tuned model
├── merge_lora.py         # Merge LoRA adapter into the base model
├── export_gguf.py        # Convert merged model to GGUF
│
├── data/
│   └── alpaca_tiny.json  # Small Alpaca‑style dataset for testing
│
├── models/               # Output directory (created automatically)
│   ├── tinyllama-lora/   # LoRA adapter weights
│   ├── tinyllama-merged/ # Merged full model
│   └── *.gguf            # Exported GGUF models
│
├── Makefile              # Workflow automation
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 📄 File‑by‑File Explanation

### `train.py`

-   Loads TinyLlama base model
-   Loads tokenizer and dataset
-   Builds causal training sequences
-   Applies LoRA adapters
-   Runs Hugging Face `Trainer`
-   Saves LoRA weights to `models/tinyllama-lora/`

### `infer.py`

-   Loads base model + LoRA adapter
-   Builds a prompt using the training format:
```
    User: <instruction>
    Assistant:
```

-   Generates a single clean answer
-   Prints the output

### `merge_lora.py`

-   Loads base model + LoRA adapter
-   Merges LoRA weights into a standalone model
-   Saves to `models/tinyllama-merged/`

### `export_gguf.py`

-   Uses llama.cpp's `convert_hf_to_gguf.py` script
-   Converts merged model to GGUF format
-   Saves `.gguf` files into `models/`

### `data/alpaca_tiny.json`

-   Small Alpaca‑style dataset
-   Contains `instruction` and `output` fields
-   Used for quick fine‑tuning tests

### `Makefile`

Provides shortcuts:

-   `make install` --- install dependencies
-   `make train` --- run LoRA training
-   `make infer` --- run inference
-   `make merge` --- merge LoRA weights
-   `make gguf` --- export GGUF
-   `make clean` --- remove generated model files

---

## ▶️ Usage

### Train the model
```bash
make train
```

### Run inference
```bash
make infer
```

### Merge LoRA into the base model
```bash
make merge
```

### Export GGUF
```bash
make gguf
```

---

## ⚙️ Configuration Notes

You can modify:

-   Model name
-   Dataset path
-   LoRA hyperparameters
-   TrainingArguments

Inside `train.py`.

---

## 🧩 Troubleshooting

### Model generates multi‑turn conversations

Use the training prompt format:
```
User: <instruction>
Assistant:
```

Strip extra turns:
```python
clean = decoded.split("User:")[0]
```

### Makefile errors

If you see:
```
missing separator
```

Your Makefile uses **spaces instead of tabs**.

### CUDA / VRAM issues

Lower:

-   Batch size
-   Sequence length
-   LoRA rank