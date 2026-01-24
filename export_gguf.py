import subprocess
import sys

model_dir = "models/tinyllama-merged"
output = "models/tinyllama-merged.gguf"

llama_cpp_path = "/mnt/c/Users/Michael/PycharmProjects/llama.cpp"
cmd = [
    sys.executable,  # Use current Python interpreter
    f"{llama_cpp_path}/convert_hf_to_gguf.py",
    model_dir,
    "--outfile", output
]

print("Running:", " ".join(cmd))
result = subprocess.run(cmd)
sys.exit(result.returncode)