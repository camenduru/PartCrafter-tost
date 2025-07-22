FROM ubuntu:22.04

WORKDIR /content

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV PYTHONDONTWRITEBYTECODE=True
ENV PATH="/home/camenduru/.local/bin:/usr/local/cuda/bin:${PATH}"

RUN apt update -y && apt install -y software-properties-common build-essential \
    libgl1 libglib2.0-0 zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev && \
    add-apt-repository -y ppa:git-core/ppa && apt update -y && \
    apt install -y python-is-python3 python3-pip sudo nano aria2 curl wget git git-lfs unzip unrar ffmpeg && \
    # aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run -d /content -o cuda_12.9.1_575.57.08_linux.run && sh cuda_12.9.1_575.57.08_linux.run --silent --toolkit && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run -d /content -o cuda_12.4.1_550.54.15_linux.run && sh cuda_12.4.1_550.54.15_linux.run --silent --toolkit && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf && ldconfig && \
    git clone https://github.com/aristocratos/btop /content/btop && cd /content/btop && make && make install && \
    adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home
    
USER camenduru

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128 && \
    pip install xformers --extra-index-url https://download.pytorch.org/whl/cu128 && \
    pip install trimesh huggingface-hub omegaconf accelerate pyrender diffusers transformers scikit-image einops opencv-python safetensors timm kornia runpod && \
    pip install https://github.com/camenduru/wheels/releases/download/3090/torch_cluster-1.6.3-cp310-cp310-linux_x86_64.whl && \
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch main https://github.com/wgsxm/PartCrafter /content/PartCrafter && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/wgsxm/PartCrafter/raw/main/model_index.json -d /content/PartCrafter/pretrained_weights/PartCrafter -o model_index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/wgsxm/PartCrafter/raw/main/vae/config.json -d /content/PartCrafter/pretrained_weights/PartCrafter/vae -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/wgsxm/PartCrafter/resolve/main/vae/diffusion_pytorch_model.safetensors -d /content/PartCrafter/pretrained_weights/PartCrafter/vae -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/wgsxm/PartCrafter/raw/main/transformer/config.json -d /content/PartCrafter/pretrained_weights/PartCrafter/transformer -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/wgsxm/PartCrafter/resolve/main/transformer/diffusion_pytorch_model.safetensors -d /content/PartCrafter/pretrained_weights/PartCrafter/transformer -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/wgsxm/PartCrafter/raw/main/scheduler/scheduler_config.json -d /content/PartCrafter/pretrained_weights/PartCrafter/scheduler -o scheduler_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/wgsxm/PartCrafter/raw/main/image_encoder_dinov2/config.json -d /content/PartCrafter/pretrained_weights/PartCrafter/image_encoder_dinov2 -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/wgsxm/PartCrafter/resolve/main/image_encoder_dinov2/model.safetensors -d /content/PartCrafter/pretrained_weights/PartCrafter/image_encoder_dinov2 -o model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/wgsxm/PartCrafter/raw/main/feature_extractor_dinov2/preprocessor_config.json -d /content/PartCrafter/pretrained_weights/PartCrafter/feature_extractor_dinov2 -o preprocessor_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/1038lab/RMBG-2.0/raw/main/BiRefNet_config.py -d /content/PartCrafter/pretrained_weights/RMBG-2.0 -o BiRefNet_config.py && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/1038lab/RMBG-2.0/raw/main/birefnet.py -d /content/PartCrafter/pretrained_weights/RMBG-2.0 -o birefnet.py && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/1038lab/RMBG-2.0/raw/main/config.json -d /content/PartCrafter/pretrained_weights/RMBG-2.0 -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/1038lab/RMBG-2.0/resolve/main/model.safetensors -d /content/PartCrafter/pretrained_weights/RMBG-2.0 -o model.safetensors

COPY ./worker_runpod.py /content/PartCrafter/worker_runpod.py
WORKDIR /content/PartCrafter
CMD python worker_runpod.py