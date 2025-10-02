from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

def ultra_fast_inference():
    """Minimal version for maximum speed"""
    model_name = "microsoft/phi-1_5"
    local_model_dir = "./local_models"
    local_model_path = f"{local_model_dir}/models/{model_name.replace('/', '_')}"
    local_tokenizer_path = f"{local_model_dir}/tokenizers/{model_name.replace('/', '_')}"
    
    # Quick load
    tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Force MPS for speed
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(device)
    else:
        device = torch.device("cpu")
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to(device)
    
    model.eval()
    
    # Fast generation function
    def quick_generate(prompt, max_tokens=50):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=0.2,  # Very low for speed
                do_sample=False,   # Greedy = fastest
                pad_token_id=tokenizer.eos_token_id
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
    
    # Test
    prompts = [
        "Python factorial function:",
        "Explain AI:",
        "Benefits of solar energy:"
    ]
    
    for prompt in prompts:
        start = time.time()
        response = quick_generate(prompt, 40)
        speed = time.time() - start
        print(f"⏱️ {speed:.2f}s | {prompt} → {response}")

# Run ultra-fast version
ultra_fast_inference()