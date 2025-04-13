# HiDream-l1-a


## Installation

1. git clone 

```bash
https://github.com/arot-devs/HiDream-I1-a
cd HiDream-I1-a
```

2. git clone main files : 

```bash
git clone https://huggingface.co/HiDream-ai/HiDream-I1-Full
```

3. git clone llama:

```bash
git clone https://huggingface.co/NousResearch/Meta-Llama-3.1-8B-Instruct
```

install reqs:

```bash
pip install -r requirements.txt
pip install sentencepiece
python gradio_demo.py
```


## Running Batch Gen

```bash
# controller will start 4 GPU workers, each taking a card (modify if needed)
python controller.py

# generator takes a txt, sends requests to controller and saves results to outputs/



