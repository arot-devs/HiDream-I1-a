import os
import json
import time
import hashlib
import requests
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from io import BytesIO
import base64
import threading

# === Config ===
CONTROLLER = "http://127.0.0.1:9000/generate"
CLIENT_ID = "my-client-1"
PROMPT_FILE = "user_prompts.txt"  # optional: use instead of PROMPTS
SAVE_DIR = "outputs"
NUM_WORKERS = 4  # Number of parallel requests
MAX_IN_FLIGHT = NUM_WORKERS * 2  # Allow some buffer

# === Read prompts ===
if os.path.exists(PROMPT_FILE):
    SAVE_DIR = f"outputs_{os.path.splitext(os.path.basename(PROMPT_FILE))[0]}"
    print(f"Using prompts from {PROMPT_FILE} | Saving to {SAVE_DIR}")
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        PROMPTS = [line.strip() for line in f if line.strip()]
else:
    PROMPTS = ["a cat", "a dog", "a castle", "a dragon"] * 4

# === Ensure save dir ===
os.makedirs(SAVE_DIR, exist_ok=True)

# === Save function ===
def save_image_and_metadata(img_b64, metadata, index):
    prefix = f"{index:0{len(str(len(PROMPTS)))}}"
    
    # Compute hash to avoid naming collisions
    meta_str = json.dumps(metadata, sort_keys=True)
    digest = hashlib.sha1(meta_str.encode()).hexdigest()[:8]
    
    img_filename = f"{prefix}_{digest}.png"
    json_filename = f"{prefix}_{digest}.json"

    # Decode and save image
    image = Image.open(BytesIO(base64.b64decode(img_b64.split(',')[-1])))
    image.save(os.path.join(SAVE_DIR, img_filename))

    # Save metadata
    metadata["client_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(SAVE_DIR, json_filename), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved: {img_filename}, {json_filename}")

# === Request function with semaphore ===
semaphore = threading.Semaphore(MAX_IN_FLIGHT)

def send(index, prompt, width, height):
    with semaphore:
        try:
            resp = requests.post(
                CONTROLLER,
                json={"prompt": prompt, "width": width, "height": height},
                headers={"X-Client-ID": CLIENT_ID},
                timeout=300
            )
            resp.raise_for_status()
            data = resp.json()
            if "image" in data and "metadata" in data:
                save_image_and_metadata(data["image"], data["metadata"], index)
            else:
                print(f"Failed: Missing image or metadata for prompt: {prompt}")
        except Exception as e:
            print(f"Error at line {index+1} ('{prompt}'): {e}")

# === Main Execution ===
if __name__ == "__main__":
    start = time.time()
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(send, idx, p, 1024, 1024)
            for idx, p in enumerate(PROMPTS)
        ]
        [f.result() for f in futures]
    print("Done in", round(time.time() - start, 2), "seconds.")