#!/bin/bash

# ==============

# Usage: 

# Launches N workers of HiDream-I1 to be used with controller.
# (usually, just run python batchgen/controller.py and processes will be managed by controller.)
# (this one is here just in case)

# Usage:                       ./batchgen/spawn_workers.sh
# To stop all workers, use:    pkill -f worker.py

# ==============

set -e

cd "/local/yada/apps/HiDream-I1-a"

SCRIPT_DIR="/local/yada/apps/HiDream-I1-a/batchgen"
WORKER_PATH="$SCRIPT_DIR/worker.py"

NUM_WORKERS=4    # <-- Number of workers to spawn
START_PORT=8001

for i in $(seq 0 $((NUM_WORKERS - 1)))
do
  PORT=$((START_PORT + i))
  echo "Starting worker on GPU $i at port $PORT..."
  CUDA_VISIBLE_DEVICES=$i PORT=$PORT python "$WORKER_PATH" > "$SCRIPT_DIR/worker_$i.log" 2>&1 &
done

echo "All workers started."
