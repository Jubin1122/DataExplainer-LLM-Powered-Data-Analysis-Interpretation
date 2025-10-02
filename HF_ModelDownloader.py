from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from huggingface_hub import login

new_token = "hf_"
os.environ['HUGGINGFACE_HUB_TOKEN'] = new_token

model_name = "microsoft/phi-1_5"
local_model_dir = "./local_models"

try:
    # Login
    login(token=new_token)
    print("✅ Logged in successfully")
    
    # Create directories
    os.makedirs(f"{local_model_dir}/tokenizers", exist_ok=True)
    os.makedirs(f"{local_model_dir}/models", exist_ok=True)
    
    # Download and save tokenizer
    print("📥 Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(f"{local_model_dir}/tokenizers/{model_name.replace('/', '_')}")
    
    # Download and save model with CPU-friendly settings
    print("📥 Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for better CPU compatibility
        trust_remote_code=True,
        low_cpu_mem_usage=True  # Optimize for CPU memory
    )
    
    model.save_pretrained(f"{local_model_dir}/models/{model_name.replace('/', '_')}")
    print(f"✅ Model saved to: {local_model_dir}/models/{model_name.replace('/', '_')}")
    
except Exception as e:
    print(f"Error: {e}")