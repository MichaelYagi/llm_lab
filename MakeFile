VENV=.venv
PYTHON=$(VENV)/bin/python3

$(VENV)/bin/activate:
    python3 -m venv $(VENV)

install: $(VENV)/bin/activate
    $(PYTHON) -m pip install --upgrade pip
    $(PYTHON) -m pip install torch --index-url https://download.pytorch.org/whl/cu121
    $(PYTHON) -m pip install -r requirements.txt

train:
    $(PYTHON) train.py

infer:
    $(PYTHON) infer.py

profile:
    $(PYTHON) train.py --profile-gpu

merge:
    $(PYTHON) merge_lora.py

gguf:
    $(PYTHON) export_gguf.py

infer-sample:
    $(PYTHON) infer.py \
        --temperature $(temp) \
        --top_p $(top_p) \
        --max_new_tokens $(max)

clean:
    rm -rf models/tinyllama-lora
    rm -rf models/tinyllama-merged
    rm -rf models/*.gguf
