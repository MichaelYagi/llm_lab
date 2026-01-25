# ------------------------------------------------------------
# Makefile for TinyLlama LoRA fine‑tuning workflow
# ------------------------------------------------------------
# This Makefile automates:
# - Virtual environment setup
# - Dependency installation
# - Training (LoRA)
# - Inference
# - Profiling
# - Merging LoRA weights
# - Exporting GGUF
# - Cleaning model artifacts
# ------------------------------------------------------------

# Path to the virtual environment directory
VENV = .venv

# Python interpreter inside the virtual environment
PYTHON = $(VENV)/bin/python3


# ------------------------------------------------------------
# Create the virtual environment
# ------------------------------------------------------------
# This target runs only if .venv/bin/activate does not exist.
# It ensures the venv is created before any other Python tasks.
$(VENV)/bin/activate:
	python3 -m venv $(VENV)


# ------------------------------------------------------------
# Install dependencies into the virtual environment
# ------------------------------------------------------------
# - Upgrades pip
# - Installs CUDA-enabled PyTorch
# - Installs all project dependencies from requirements.txt
install: $(VENV)/bin/activate
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install torch --index-url https://download.pytorch.org/whl/cu121
	$(PYTHON) -m pip install -r requirements.txt


# ------------------------------------------------------------
# Run LoRA training
# ------------------------------------------------------------
# Executes train.py using the venv Python interpreter with data from alpaca_tiny.json.
# Produces LoRA adapter weights in models/tinyllama-lora/
train:
	$(PYTHON) train.py


# ------------------------------------------------------------
# Run inference
# ------------------------------------------------------------
# Executes infer.py to test the fine‑tuned model.
infer:
	$(PYTHON) infer.py


# ------------------------------------------------------------
# Profile GPU usage during training
# ------------------------------------------------------------
# Runs training with a profiling flag for performance diagnostics.
profile:
	$(PYTHON) train.py --profile-gpu


# ------------------------------------------------------------
# Merge LoRA adapter into the base model
# ------------------------------------------------------------
# Produces a merged full model in models/tinyllama-merged/
merge:
	$(PYTHON) merge_lora.py


# ------------------------------------------------------------
# Export merged model to GGUF format
# ------------------------------------------------------------
# Runs export_gguf.py, which calls convert-hf-to-gguf.
gguf:
	$(PYTHON) export_gguf.py


# ------------------------------------------------------------
# Run inference with sampling parameters
# ------------------------------------------------------------
# Usage example:
#   make infer-sample temp=0.8 top_p=0.9 max=200
infer-sample:
	$(PYTHON) infer.py \
        --temperature $(temp) \
        --top_p $(top_p) \
        --max_new_tokens $(max)


# ------------------------------------------------------------
# Clean generated model artifacts
# ------------------------------------------------------------
# Removes LoRA adapters, merged models, and GGUF exports.
clean:
	rm -rf models/tinyllama-lora
	rm -rf models/tinyllama-merged