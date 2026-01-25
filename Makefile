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
# Default target - shows help
# ------------------------------------------------------------
.DEFAULT_GOAL := help

# ------------------------------------------------------------
# Help target - lists all available commands
# ------------------------------------------------------------
.PHONY: help
help:
	@echo "TinyLlama Fine-Tuning Makefile"
	@echo "=============================="
	@echo ""
	@echo "Setup:"
	@echo "  make install          - Create venv and install dependencies"
	@echo ""
	@echo "Training Pipeline:"
	@echo "  make all              - Run full pipeline (train + merge + gguf)"
	@echo "  make pipeline         - Same as 'all'"
	@echo "  make train            - Train LoRA adapter only"
	@echo "  make merge            - Merge LoRA weights into base model"
	@echo "  make gguf             - Export merged model to GGUF format"
	@echo ""
	@echo "Testing:"
	@echo "  make infer            - Run inference with merged model"
	@echo "  make profile          - Profile GPU usage during training"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean            - Remove generated model artifacts"
	@echo "  make clean-all        - Remove venv + model artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make all              - Complete end-to-end pipeline"
	@echo "  make train infer      - Train then test immediately"
	@echo ""


# ------------------------------------------------------------
# Run complete pipeline: install → train → merge → gguf → infer-sample → clean
# ------------------------------------------------------------
.PHONY: all pipeline
all: install train infer profile merge gguf infer-sample clean
	@echo ""
	@echo "✅ Complete pipeline finished!"
	@echo "   - LoRA adapter:    models/tinyllama-lora/"
	@echo "   - Merged model:    models/tinyllama-merged/"
	@echo "   - GGUF export:     models/tinyllama-merged.gguf"
	@echo ""

pipeline: all


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
.PHONY: install
install: $(VENV)/bin/activate
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install torch --index-url https://download.pytorch.org/whl/cu121
	$(PYTHON) -m pip install -r requirements.txt
	@echo ""
	@echo "✅ Installation complete!"
	@echo "   Activate with: source $(VENV)/bin/activate"
	@echo ""


# ------------------------------------------------------------
# Run LoRA training
# ------------------------------------------------------------
# Executes train.py using the venv Python interpreter with data from alpaca_tiny.json.
# Produces LoRA adapter weights in models/tinyllama-lora/
.PHONY: train
train:
	@echo "🚀 Starting LoRA training..."
	$(PYTHON) train.py
	@echo "✅ Training complete! LoRA adapter saved to models/tinyllama-lora/"


# ------------------------------------------------------------
# Run inference
# ------------------------------------------------------------
# Executes infer.py to test the fine‑tuned model.
.PHONY: infer
infer:
	@echo "🧪 Running inference..."
	$(PYTHON) infer.py


# ------------------------------------------------------------
# Profile GPU usage during training
# ------------------------------------------------------------
# Runs training with a profiling flag for performance diagnostics.
.PHONY: profile
profile:
	@echo "📊 Profiling GPU usage during training..."
	$(PYTHON) train.py --profile-gpu


# ------------------------------------------------------------
# Merge LoRA adapter into the base model
# ------------------------------------------------------------
# Produces a merged full model in models/tinyllama-merged/
.PHONY: merge
merge:
	@echo "🔗 Merging LoRA adapter into base model..."
	$(PYTHON) merge_lora.py
	@echo "✅ Merge complete! Full model saved to models/tinyllama-merged/"


# ------------------------------------------------------------
# Export merged model to GGUF format
# ------------------------------------------------------------
# Runs export_gguf.py, which calls convert-hf-to-gguf.
.PHONY: gguf
gguf:
	@echo "📦 Exporting to GGUF format..."
	$(PYTHON) export_gguf.py
	@echo "✅ GGUF export complete! File: models/tinyllama-merged.gguf"


# ------------------------------------------------------------
# Run inference with sampling parameters
# ------------------------------------------------------------
# Usage example:
#   make infer-sample temp=0.8 top_p=0.9 max=200
.PHONY: infer-sample
infer-sample:
	$(PYTHON) infer.py \
		--temperature $(temp) \
		--top_p $(top_p) \
		--max_new_tokens $(max)


# ------------------------------------------------------------
# Clean generated model artifacts
# ------------------------------------------------------------
# Removes LoRA adapters, merged models, and GGUF exports.
.PHONY: clean
clean:
	@echo "🧹 Cleaning model artifacts..."
	rm -rf models/tinyllama-lora
	rm -rf models/tinyllama-merged
	@echo "✅ Clean complete!"


# ------------------------------------------------------------
# Clean everything including virtual environment
# ------------------------------------------------------------
.PHONY: clean-all
clean-all: clean
	@echo "🧹 Removing virtual environment..."
	rm -rf $(VENV)
	@echo "✅ Full clean complete!"