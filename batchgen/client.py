# client.py
import requests
import threading

CONTROLLER = "http://127.0.0.1:9000/generate"
CLIENT_ID = "my-client-1"

def send(prompt, w, h):
    resp = requests.post(CONTROLLER, json={
        "prompt": prompt,
        "width": w,
        "height": h
    }, headers={"X-Client-ID": CLIENT_ID})
    data = resp.json()
    print("Got metadata:", data["metadata"])

# Example usage
prompts = ["a cat", "a dog", "a castle", "a dragon"]
threads = [threading.Thread(target=send, args=(p, 512, 512)) for p in prompts]
[t.start() for t in threads]
[t.join() for t in threads]
