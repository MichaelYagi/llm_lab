# ------------------------------------------------------------
# infer.py — Run inference using your LoRA‑fine‑tuned TinyLlama
# ------------------------------------------------------------

# AutoTokenizer loads the tokenizer used by the base model.
# AutoModelForCausalLM loads the base language model.
from transformers import AutoTokenizer, AutoModelForCausalLM

# PeftModel loads your LoRA adapter and merges it with the base model.
from peft import PeftModel


# ------------------------------------------------------------
# 1. Define model paths
# ------------------------------------------------------------
# Base model: the original pretrained TinyLlama.
base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# LoRA adapter: the fine‑tuned weights you produced in train.py.
adapter_path = "models/tinyllama-lora"


# ------------------------------------------------------------
# 2. Load tokenizer
# ------------------------------------------------------------
# The tokenizer must match the base model exactly.
tokenizer = AutoTokenizer.from_pretrained(base_model)


# ------------------------------------------------------------
# 3. Load the base model onto your GPU
# ------------------------------------------------------------
# device_map="auto" automatically places the model on your RTX 3060.
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto"
)


# ------------------------------------------------------------
# 4. Load your LoRA adapter
# ------------------------------------------------------------
# This merges your fine‑tuned LoRA weights into the base model.
model = PeftModel.from_pretrained(model, adapter_path)


# ------------------------------------------------------------
# 5. Build a test prompt
# ------------------------------------------------------------
# You can change this to anything you want to test.
prompt = "Explain what an MCP agent is in simple terms."


# ------------------------------------------------------------
# 6. Tokenize the prompt and move it to the GPU
# ------------------------------------------------------------
inputs = tokenizer(
    prompt,
    return_tensors="pt"   # Return PyTorch tensors
).to(model.device)        # Move tensors to the same device as the model


# ------------------------------------------------------------
# 7. Generate a response
# ------------------------------------------------------------
# max_new_tokens controls how long the answer can be.
output = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,        # Enable sampling for more natural responses
    temperature=0.7         # Controls creativity
)


# ------------------------------------------------------------
# 8. Decode and print the model's output
# ------------------------------------------------------------
print("\n=== Model Response ===\n")
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("\n=======================\n")
