import subprocess, requests, time, threading
from flask import Flask, request, jsonify
from queue import Queue
from uuid import uuid4
import os, pathlib

NUM_GPUS = 4
WORKERS = []
PORTS = [8001 + i for i in range(NUM_GPUS)]
worker_path = str(pathlib.Path(__file__).resolve().parent / "worker.py")

from concurrent.futures import ThreadPoolExecutor
import subprocess, os

def launch_worker(i, port):
    return {
        "gpu": i,
        "port": port,
        "url": f"http://127.0.0.1:{port}/generate",
        "proc": subprocess.Popen(
            ["python", worker_path],
            env={
                **os.environ,
                "PORT": str(port),
                "CUDA_VISIBLE_DEVICES": str(i)
            }
        )
    }

# Concurrently launch workers
with ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(launch_worker, i, port)
        for i, port in enumerate(PORTS)
    ]
    WORKERS = [f.result() for f in futures]


# Round-robin load balancer
next_worker = 0
worker_lock = threading.Lock()

def get_next_worker():
    global next_worker
    with worker_lock:
        w = WORKERS[next_worker]
        next_worker = (next_worker + 1) % NUM_GPUS
    return w

# Client session tracking
CLIENTS = {}

def prune_clients():
    while True:
        now = time.time()
        for k in list(CLIENTS.keys()):
            if now - CLIENTS[k]["last_seen"] > 300:  # 5 min timeout
                del CLIENTS[k]
        time.sleep(60)

threading.Thread(target=prune_clients, daemon=True).start()

# Flask controller
app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    client_id = request.headers.get("X-Client-ID", str(uuid4()))
    CLIENTS[client_id] = {"last_seen": time.time()}

    req_json = request.get_json()

    for _ in range(2):  # try twice
        worker = get_next_worker()
        try:
            resp = requests.post(worker["url"], json=req_json, timeout=60)
            return jsonify(resp.json())
        except Exception as e:
            print(f"Worker {worker['gpu']} failed: {e}")
            continue
    return jsonify({"error": "All workers failed"}), 500

@app.route("/clients")
def list_clients():
    return jsonify({k: v["last_seen"] for k, v in CLIENTS.items()})

if __name__ == "__main__":
    app.run(port=9000)
