## Forecast Explainer — Local LLM (Microsoft Phi-1.5 → GGUF → Quantized → llama.cpp + Python)

This repo runs a compact, CPU-friendly LLM locally to generate concise explanations for time-series forecasts (weekly/monthly/quarterly). We use Microsoft Phi-1.5 (~1.3B params) converted to GGUF and quantized with llama.cpp so it runs fast on laptops (including Apple M-series) with low RAM.

```
What this README covers (steps 1–4)

1.1 Create a Hugging Face account & token

1.2 Create & activate a Python venv (script)

2. Download Phi-1.5 (model + tokenizer)

3.1 Convert HF model → GGUF (llama.cpp)

3.2 Quantize the GGUF

4. Quick C++ smoke test (llama.cpp binary)

5. Python and Terminal usage with llama-cpp-python
```

### Quick start (first things to do)
#### 1.1 — Create a Hugging Face account & token

1. Sign up at https://huggingface.co
 (create an account).

2. Go to Settings → Access Tokens and create a token (type: read or write as needed).

3. Save the token somewhere safe. You will export it into your shell environment for the downloader.

Example (macOS/Linux):
```
export HUGGINGFACE_HUB_TOKEN="hf_XXXXXXXXXXXXXXXXXXXXXXXX"
```

Example (Windows PowerShell):
```
setx HUGGINGFACE_HUB_TOKEN "hf_XXXXXXXXXXXXXXXXXXXXXXXX"
```

Replace hf_... with your token. ***I have already set up the `python script` for you, with authentication logic. Link mentioned below***

### 1.2 — Create & activate a Python venv (script)

**Bash** (setup_venv.sh) — create, activate, upgrade pip, install requirements and PyTorch CPU wheel:

```

python -m venv venv

# On Linux/macOS:
source venv/bin/activate

#vOn Windows
## Command Prompt
venv\Scripts\activate

# PowerShell
venv\Scripts\Activate.ps1
```


Once venv is activated, upgrade pip & install deps:

```
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

```

### 2. Download Phi-1.5 (model + tokenizer)

Use the Hugging Face transformers client to download the model & tokenizer into local_models/.

Link to actual downloading script: [HF Script](HF_ModelDownloader.py)

Example script download_phi.py: 

```python
# download_phi.py
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", os.environ.get("HUGGINGFACE_HUB_TOKEN", ""))

model_name = "microsoft/phi-1_5"
local_dir = "./local_models"

os.makedirs(f"{local_dir}/models", exist_ok=True)
os.makedirs(f"{local_dir}/tokenizers", exist_ok=True)

# Download tokenizer
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tok.save_pretrained(f"{local_dir}/tokenizers/{model_name.replace('/', '_')}")

# Download model (FP16/FP32 as available)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",          # or torch.float32 / torch.float16 if desired
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
model.save_pretrained(f"{local_dir}/models/{model_name.replace('/', '_')}")
```

**Notes**

torch_dtype="auto" lets HF choose best dtype. FP16 is common for smaller GGUF conversion; FP32 is safer for CPU-only workflows.

After this you will have folders under local_models/models/... and local_models/tokenizers/....

### 3.1 Convert HF model → GGUF (llama.cpp)

You need the convert-hf-to-gguf.py script from llama.cpp:

**Get llama.cpp if you don't have it yet**
```
git clone  https://github.com/ggerganov/llama.cpp

cd llama.cpp

pip install llama-cpp-python

```

llama.cpp ships a converter script ``(names vary: convert-hf-to-gguf.py or convert_hf_to_gguf.py)``.


Sometimes Your current convert_hf_to_gguf.py doesn’t recognize --tokenizer (older versions don’t need it because they expect the tokenizer files inside the --model folder).

Copy the tokenizer files into the model folder so the converter sees everything in one place:
```
cp -r /Users/jubinmohanty/Desktop/rag_pipeline/local_models/tokenizers/microsoft_phi-1_5/* \
      /Users/jubinmohanty/Desktop/rag_pipeline/local_models/models/microsoft_phi-1_5/
```

### 3.2 Script to convert HF model → GGUF and quantise it(reduce RAM & speed up inference)

```
python3 convert_hf_to_gguf.py \
  --outfile /Users/jubinmohanty/Desktop/rag_pipeline/phi1_5.gguf \
  --outtype q8_0 \
  /Users/jubinmohanty/Desktop/rag_pipeline/local_models/models/microsoft_phi-1_5
```

Output: ``phi1_5.gguf``.

What will happen:
* It will read model + tokenizer (make sure you already copied the tokenizer files into the model dir as I showed earlier).
* Output a file: /Users/jubinmohanty/Desktop/rag_pipeline/phi1_5.gguf
* Quantized to q8_0 (8-bit).

### 4. Quick C++ smoke test (llama.cpp binary)

Replace <BIN> with what you found (prefer llama-cli if it exists):
```
<BIN> \
  -m /Users/jubinmohanty/Desktop/rag_pipeline/phi1_5.gguf \
  -t 1 \
  -c 512 \
  -n 80 \
  -p "Explain solar energy in one short sentence."
```

***Examples***:

./build/bin/llama-cli -m /Users/jubinmohanty/Desktop/rag_pipeline/phi1_5.gguf -t 1 -c 512 -n 80 -p "..."

**or**

./build/bin/main -m /Users/jubinmohanty/Desktop/rag_pipeline/phi1_5.gguf -t 1 -c 512 -n 80 -p "..."


**Flags**:
```
-m : model path (GGUF)

-t : threads

-c : context tokens (n_ctx)

-n : max tokens to generate

-p : prompt
```

### 5. Terminal and python usage.

**Running the C++ binary (what you did with llama-cli)**

* You cd into /Users/jubinmohanty/Desktop/llama.cpp and run:

* ``./build/bin/llama-cli -m /path/to/model.gguf -p "..."``
This doesn’t need Python at all. It’s a standalone executable.

**Running from Python (llama-cpp-python)**

* Here you don’t import anything from your llama.cpp source folder.
* Instead, you install a Python package that provides the wrapper.

***Install once***

1. pip install llama-cpp-python
2. ⚠️ Do this inside your active venv (cpp_quantised_llm). So, you don’t pollute system Python.

***already included in repo — linked here***:
[optimised_inference_example.py](optimised_inference_example.py)

***Important***


n_ctx must be ≥ len(prompt tokens) + max_tokens. Leave a safety buffer (e.g., RESERVE_TOKENS = 32).

Use a small warmup call after loading to avoid a slow first generation:

``_ = llm.create_completion(prompt="Hi", max_tokens=1)``

Full Forecast explanation Pipeline: [Forecast_explanation_pipeline](Forecast_explanation_pipeline.py) 


