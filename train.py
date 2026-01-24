# ------------------------------------------------------------
# train.py — Fine‑tune TinyLlama using LoRA on your RTX 3060
# ------------------------------------------------------------

# Hugging Face Transformers provides model architectures,
# tokenizers, and training utilities.
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

# Hugging Face Datasets loads and processes your dataset.
from datasets import load_dataset

# PEFT (Parameter‑Efficient Fine‑Tuning) enables LoRA/QLoRA.
from peft import LoraConfig, get_peft_model


# ------------------------------------------------------------
# 1. Choose a base model
# ------------------------------------------------------------
# TinyLlama is small enough to train quickly but still powerful.
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# ------------------------------------------------------------
# 2. Load the tokenizer
# ------------------------------------------------------------
# The tokenizer converts text → token IDs.
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Some models don't have a pad token; we set it to EOS for safety.
tokenizer.pad_token = tokenizer.eos_token


# ------------------------------------------------------------
# 3. Load your dataset
# ------------------------------------------------------------
# This loads the JSON file you created in Step 4.
dataset = load_dataset("json", data_files="data/alpaca_tiny.json")


# ------------------------------------------------------------
# 4. Preprocess dataset: convert text → token IDs
# ------------------------------------------------------------
def format(example):
    # Build a simple instruction prompt format.
    prompt = f"User: {example['instruction']}\nAssistant:"

    # Tokenize the prompt (input to the model).
    example["input_ids"] = tokenizer(
        prompt,
        truncation=True,
        max_length=512
    ).input_ids

    # Tokenize the expected output (labels for training).
    example["labels"] = tokenizer(
        example["output"],
        truncation=True,
        max_length=512
    ).input_ids

    return example

# Apply the formatting function to every example.
dataset = dataset.map(format)


# ------------------------------------------------------------
# 5. Load the base model
# ------------------------------------------------------------
# device_map="auto" automatically places the model on your GPU.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)


# ------------------------------------------------------------
# 6. Configure LoRA
# ------------------------------------------------------------
# LoRA injects small trainable matrices into the model so you
# only train ~1% of the parameters — perfect for your 12GB GPU.
lora = LoraConfig(
    r=16,                 # Rank of LoRA matrices (capacity)
    lora_alpha=32,        # Scaling factor
    lora_dropout=0.05,    # Regularization
    target_modules=["q_proj", "v_proj"]  # Apply LoRA to attention layers
)

# Wrap the base model with LoRA adapters.
model = get_peft_model(model, lora)


# ------------------------------------------------------------
# 7. Training configuration
# ------------------------------------------------------------
# These settings control how training behaves.
args = TrainingArguments(
    output_dir="models/tinyllama-lora",  # Where checkpoints go
    per_device_train_batch_size=2,       # Fits in 12GB VRAM
    gradient_accumulation_steps=4,       # Effective batch size = 2*4 = 8
    warmup_steps=50,                     # Stabilize early training
    max_steps=500,                       # Small run for testing
    learning_rate=2e-4,                  # Good LoRA LR
    fp16=True,                           # Use half precision for speed
    logging_steps=10,                    # Print progress every 10 steps
    save_steps=200,                      # Save checkpoints periodically
)


# ------------------------------------------------------------
# 8. Create the Trainer
# ------------------------------------------------------------
# Trainer handles batching, optimization, logging, etc.
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
)


# ------------------------------------------------------------
# 9. Start training
# ------------------------------------------------------------
trainer.train()


# ------------------------------------------------------------
# 10. Save the final LoRA adapter
# ------------------------------------------------------------
model.save_pretrained("models/tinyllama-lora")
