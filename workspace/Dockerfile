FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /workspace

# =========================
# SYSTEM SETUP
# =========================
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git ffmpeg \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# =========================
# INSTALL PYTHON DEPENDENCIES
# =========================

# Copy requirements first (for caching)
COPY AnimeColor_Code/requirements.txt /workspace/requirements.txt

# Install everything in ONE layer (important)
RUN pip install --no-cache-dir \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r /workspace/requirements.txt && \
    pip install --no-cache-dir runpod

# =========================
# COPY PROJECT
# =========================
COPY . /workspace

# =========================
# FINAL SETUP
# =========================
RUN chmod +x start.sh

ENV PYTHONUNBUFFERED=1

CMD ["./start.sh"]