# ------------------------------------------------------------
# train.py — Fine‑tune TinyLlama using LoRA on your RTX 3060
# ------------------------------------------------------------

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
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


# ------------------------------------------------------------
# 3. Load your dataset
# ------------------------------------------------------------
dataset = load_dataset("json", data_files="data/alpaca_tiny.json")


# ------------------------------------------------------------
# 4. Preprocess dataset: build a single causal sequence
# ------------------------------------------------------------
def format(example):
    # Single sequence: prompt + answer
    text = f"User: {example['instruction']}\nAssistant: {example['output']}"

    enc = tokenizer(
        text,
        truncation=True,
        max_length=512,
    )

    example["input_ids"] = enc["input_ids"]
    example["attention_mask"] = enc["attention_mask"]
    # For causal LM, labels are usually the same as input_ids
    example["labels"] = enc["input_ids"].copy()

    return example


dataset = dataset.map(format)

# Keep only model-relevant columns
dataset = dataset.remove_columns(
    [col for col in dataset["train"].column_names
     if col not in ["input_ids", "attention_mask", "labels"]]
)


# ------------------------------------------------------------
# 5. Load the base model
# ------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
)


# ------------------------------------------------------------
# 6. Configure LoRA
# ------------------------------------------------------------
lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora)


# ------------------------------------------------------------
# 7. Data collator (handles padding for inputs + labels)
# ------------------------------------------------------------
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100,
    return_tensors="pt",
)


# ------------------------------------------------------------
# 8. Training configuration
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
# 9. Create the Trainer
# ------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    data_collator=data_collator,
)


# ------------------------------------------------------------
# 10. Start training
# ------------------------------------------------------------
trainer.train()


# ------------------------------------------------------------
# 11. Save the final LoRA adapter
# ------------------------------------------------------------
model.save_pretrained("models/tinyllama-lora")
