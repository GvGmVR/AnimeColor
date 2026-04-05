FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python setup
RUN pip3 install --upgrade pip

# Install torch WITH CUDA (IMPORTANT)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Copy project
COPY . /workspace

# Install remaining dependencies
RUN pip install -r requirements.txt

# RunPod SDK
RUN pip install runpod

# Make start script executable
RUN chmod +x start.sh

CMD ["./start.sh"]