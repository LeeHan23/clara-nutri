from transformers import AutoModel, AutoTokenizer
import torch

local_model_path= "./clara-weights/compression-16"

print("Loading CLaRa-7B-Instruct from local directory...")

# Added use_fast=False to force the use of sentencepiece
tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=False)

model = AutoModel.from_pretrained(
    local_model_path, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to("mps")

print("Success! CLaRa is loaded onto the Mac Studio GPU.")
