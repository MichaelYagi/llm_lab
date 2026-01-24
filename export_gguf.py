import os

model_dir = "models/tinyllama-merged"
output = "models/tinyllama-merged.gguf"

cmd = f"convert-hf-to-gguf {model_dir} --outfile {output}"
print("Running:", cmd)
os.system(cmd)
