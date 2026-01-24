from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter = "models/tinyllama-lora"
output = "models/tinyllama-merged"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(base, device_map="cpu")

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, adapter)

print("Merging weights...")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained(output)
tokenizer = AutoTokenizer.from_pretrained(base)
tokenizer.save_pretrained(output)

print("Done! Merged model saved to:", output)
