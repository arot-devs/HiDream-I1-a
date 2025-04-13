# worker.py
import os, torch
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

from hi_diffusers import HiDreamImagePipeline, HiDreamImageTransformer2DModel
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

app = FastAPI()

# ==== Model Definitions ====

MODEL_PREFIX = "./"
LLAMA_MODEL_NAME = "Meta-Llama-3.1-8B-Instruct"

MODEL_CONFIGS = {
    "dev": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Dev",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    },
    "full": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Full",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler
    },
    "fast": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    }
}

# Global state
pipe = None
current_model = None

def load_models(model_type):
    config = MODEL_CONFIGS[model_type]
    scheduler = config["scheduler"](num_train_timesteps=1000, shift=config["shift"], use_dynamic_shifting=False)

    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(LLAMA_MODEL_NAME, use_fast=False)
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        config["path"],
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    pipeline = HiDreamImagePipeline.from_pretrained(
        config["path"],
        scheduler=scheduler,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=torch.bfloat16,
    ).to("cuda", torch.bfloat16)
    pipeline.transformer = transformer

    return pipeline, config



# ==== API Endpoint ====

class GenRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    seed: int = -1
    model: str = "full"  # optional override


import base64
from io import BytesIO

def generate_image(prompt, width, height, seed, model_type="full"):
    global pipe, current_model

    if pipe is None or model_type != current_model:
        if pipe is not None:
            del pipe
            torch.cuda.empty_cache()
        print(f"Loading model: {model_type}")
        pipe, _ = load_models(model_type)
        current_model = model_type
        print(f"{model_type} model loaded successfully")

    config = MODEL_CONFIGS[model_type]
    generator = torch.Generator("cuda").manual_seed(seed)

    images = pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=config["guidance_scale"],
        num_inference_steps=config["num_inference_steps"],
        num_images_per_prompt=1,
        generator=generator,
    ).images

    img = images[0]

    # Convert PIL image to base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return img_str

@app.post("/generate")
async def generate(req: GenRequest):
    actual_seed = req.seed if req.seed != -1 else int(torch.randint(0, 2**32 - 1, (1,)).item())
    img_str = generate_image(req.prompt, req.width, req.height, actual_seed, model_type=req.model)

    metadata = {
        "model": req.model,
        "prompt": req.prompt,
        "width": req.width,
        "height": req.height,
        "seed": actual_seed,
        "device": torch.cuda.current_device(),
        "date": datetime.now().isoformat(),
    }

    return {"image": img_str, "metadata": metadata}


# ==== Entrypoint ====
if __name__ == "__main__":
    import uvicorn

    torch.cuda.set_device(0)
    gpu_id = torch.cuda.current_device()
    print(f"Worker running on GPU {gpu_id} (visible as CUDA:{gpu_id})")

    print("Loading default model (full)...")
    pipe, _ = load_models("full")
    current_model = "full"
    print("Model loaded successfully!")

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ["PORT"]))
