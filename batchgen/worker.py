# worker.py
import os, torch
from fastapi import FastAPI, Request
from pydantic import BaseModel

import sys
from pathlib import Path

# Add project root (parent of batchgen/) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))


from gradio_demo import generate_image  # your original demo code

app = FastAPI()

class GenRequest(BaseModel):
    prompt: str
    width: int = 512
    height: int = 512

@app.post("/generate")
async def generate(req: GenRequest):
    image = generate_image(req.prompt, req.width, req.height)  # your existing function
    metadata = {
        "model": "StableXL",
        "prompt": req.prompt,
        "width": req.width,
        "height": req.height,
        "device": torch.cuda.current_device(),
    }
    return {"image": image, "metadata": metadata}

if __name__ == "__main__":
    import uvicorn

    # Set the device *before* querying current_device
    torch.cuda.set_device(0)  # This is safe: CUDA_VISIBLE_DEVICES maps 0 to the actual GPU
    gpu_id = torch.cuda.current_device()

    print(f"Worker running on GPU {gpu_id} (visible as CUDA:{gpu_id})")
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ["PORT"]))

