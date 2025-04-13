import gradio as gr
import requests
import base64
import json
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

CONTROLLER = "http://127.0.0.1:9000/generate"
CLIENT_ID = "my-client-1"
NUM_WORKERS = 4


def decode_image(img_b64):
    image = Image.open(BytesIO(base64.b64decode(img_b64.split(',')[-1])))
    return image


def generate_batch(prompt_text, width, height, front, back):
    prompts_raw = [line.strip() for line in prompt_text.strip().splitlines() if line.strip()]
    prompts = [f"{front}, {p}, {back}".strip(", ") for p in prompts_raw]
    results = [None] * len(prompts)

    def send(index, prompt):
        try:
            resp = requests.post(
                CONTROLLER,
                json={"prompt": prompt, "width": width, "height": height},
                headers={"X-Client-ID": CLIENT_ID},
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            if "image" in data and "metadata" in data:
                image = decode_image(data["image"])
                json_str = json.dumps(data["metadata"], indent=2)
                results[index] = (image, json_str)
            else:
                results[index] = (None, f"Missing fields for prompt: {prompt}")
        except Exception as e:
            results[index] = (None, f"Error for prompt '{prompt}': {str(e)}")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(send, idx, p) for idx, p in enumerate(prompts)]
        [f.result() for f in futures]

    images = [r[0] for r in results if r[0] is not None]
    metadata = [r[1] for r in results if r[0] is not None]
    errors = [r[1] for r in results if r[0] is None]

    return images, metadata, "\n".join(errors)


with gr.Blocks() as demo:
    gr.Markdown("""# Batch Image Generator\nEnter one prompt per line.""")
    with gr.Row():
        prompt_input = gr.Textbox(lines=10, label="Prompts (one per line)")
        with gr.Column():
            text_front = gr.Textbox(label="Text Front")
            text_back = gr.Textbox(label="Text Back")
            width = gr.Number(value=1024, label="Width")
            height = gr.Number(value=1024, label="Height")
            run_button = gr.Button("Generate")

    gallery = gr.Gallery(label="Generated Images", columns=4)
    metadata_output = gr.JSON(label="Metadata")
    error_output = gr.Textbox(label="Errors", lines=4)

    run_button.click(
        fn=generate_batch,
        inputs=[prompt_input, width, height, text_front, text_back],
        outputs=[gallery, metadata_output, error_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
