from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
import torch

# 1. Initialize the API
app = FastAPI(title="CLaRa Inference API", description="The Brain for Nutribot")

# 2. Define what the incoming data from the Ubuntu server should look like
class QueryRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.3 # Lower temperature = more clinical/factual, less creative

# 3. Load the model globally (This happens ONLY when you start the script)
local_model_path = "./clara-weights/compression-16"
print("Loading CLaRa into Mac Studio memory. This may take a moment...")

tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=False)
model = AutoModel.from_pretrained(
    local_model_path, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to("mps")

print("Brain is online and listening!")

# 4. Create the endpoint that the Ubuntu server will talk to
@app.post("/generate")
async def generate_text(request: QueryRequest):
    try:
        # Format the input for the GPU
        inputs = tokenizer(request.prompt, return_tensors="pt").to("mps")
        
        # Generate the answer
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode the output back into human text
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the output (remove the original prompt from the response)
        final_answer = response_text[len(request.prompt):].strip()
        
        return {"answer": final_answer}

    except Exception as e:
        print(f"Error during generation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
