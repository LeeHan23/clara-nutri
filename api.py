from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
from typing import List, Optional
import torch
import traceback

app = FastAPI(title="CLaRa Inference API", description="The Brain for Nutribot")

class QueryRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.3
    documents: Optional[List[str]] = None

local_model_path = "./clara-weights/compression-16"
print("Loading CLaRa into Mac Studio memory. This may take a moment...")

tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=False)

model = AutoModel.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model = model.to(torch.float16)
model = model.to("mps")
model.eval()

print("Brain is online and listening!")

@app.post("/generate")
async def generate_text(request: QueryRequest):
    try:
        documents = request.documents if request.documents else [""]
        with torch.no_grad():
            outputs = model.generate_from_text(
                questions=[request.prompt],
                documents=[documents],
                max_new_tokens=request.max_tokens
            )
        answer = outputs[0] if outputs else ""
        return {"answer": answer}
    except Exception as e:
        full_trace = traceback.format_exc()
        print("=" * 80)
        print("FULL TRACEBACK:")
        print(full_trace)
        print("=" * 80)
        raise HTTPException(status_code=500, detail=str(e))
