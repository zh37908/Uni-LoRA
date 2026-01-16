import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Add local peft to path
sys.path.append("/home/hzhaobi/Uni-LoRA/instruction_tuning/peft/src")

from peft import PeftModel

def merge_lora():
    base_model_path = "meta-llama/Llama-2-7b-hf"
    adapter_path = "/home/hzhaobi/Uni-LoRA/instruction_tuning/output/llama2_7b_vb/checkpoint-3235/adapter_model"
    output_path = "/home/hzhaobi/Uni-LoRA/instruction_tuning/output/llama2_7b_merged"

    print(f"Loading base model: {base_model_path}")
    # Using float16 for merging
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="cpu", # Merge on CPU to save GPU memory if needed, or "auto"
        trust_remote_code=True
    )

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path)
    
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    print("Success! Merged model saved to " + output_path)

if __name__ == "__main__":
    merge_lora()
