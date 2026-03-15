FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf python3.11 /usr/bin/python && ln -sf pip3 /usr/bin/pip

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/tmp/hf_cache
ENV HF_HOME=/tmp/hf_cache

CMD ["python", "scripts/run_lora.py", "--config", "configs/lora_config.yaml"]
