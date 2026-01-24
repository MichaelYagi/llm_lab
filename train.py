# ------------------------------------------------------------
# train.py — Fine‑tune TinyLlama using LoRA on your RTX 3060
# ------------------------------------------------------------

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model


# ------------------------------------------------------------
# 1. Choose a base model
# ------------------------------------------------------------
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# ------------------------------------------------------------
# 2. Load the tokenizer
# ------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# NEW: padding collator to fix batch shape mismatch
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True
)


# ------------------------------------------------------------
# 3. Load your dataset
# ------------------------------------------------------------
dataset = load_dataset("json", data_files="data/alpaca_tiny.json")


# ------------------------------------------------------------
# 4. Preprocess dataset: convert text → token IDs
# ------------------------------------------------------------
def format(example):
    prompt = f"User: {example['instruction']}\nAssistant:"

    example["input_ids"] = tokenizer(
        prompt,
        truncation=True,
        max_length=512
    ).input_ids

    example["labels"] = tokenizer(
        example["output"],
        truncation=True,
        max_length=512
    ).input_ids

    return example


dataset = dataset.map(format)


# ------------------------------------------------------------
# 5. Load the base model
# ------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)


# ------------------------------------------------------------
# 6. Configure LoRA
# ------------------------------------------------------------
lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora)


# ------------------------------------------------------------
# 7. Training configuration
# ------------------------------------------------------------
args = TrainingArguments(
    output_dir="models/tinyllama-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    max_steps=500,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=200,
)


# ------------------------------------------------------------
# 8. Create the Trainer
# ------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    data_collator=data_collator,   # <-- FIXED
)


# ------------------------------------------------------------
# 9. Start training
# ------------------------------------------------------------
trainer.train()


# ------------------------------------------------------------
# 10. Save the final LoRA adapter
# ------------------------------------------------------------
model.save_pretrained("models/tinyllama-lora")
