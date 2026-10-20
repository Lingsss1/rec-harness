FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/longseq-kuairand-bench
COPY requirements-linux.txt pyproject.toml LICENSE NOTICE ./
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121 \
    && python3 -m pip install -r requirements-linux.txt
COPY longseq_bench ./longseq_bench
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
RUN python3 -m pip install --no-deps -e .
ENTRYPOINT ["bash", "scripts/oneclick_a800.sh", "--skip-install"]
