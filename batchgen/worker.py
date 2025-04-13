# worker.py
import os, torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
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
    gpu_id = int(os.environ["GPU"])
    torch.cuda.set_device(gpu_id)
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ["PORT"]))
