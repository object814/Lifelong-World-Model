# For more information, please refer to https://aka.ms/vscode-docker-python
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Args for user creation - will be overridden by values from .env
ARG USER_NAME=developer
ARG USER_UID=1000
ARG WANDB_API_KEY=""

RUN apt-get update --allow-releaseinfo-change && \
    apt-get install -y gnupg && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys $(apt-key list | grep expired | awk '{print $2}') || true && \
    apt-get update && \
    apt-get install -y \
        build-essential \
        libssl-dev \
        libffi-dev \
        libgl1 \
        imagemagick \
        python3-dev \
        wget \
        curl \
        ca-certificates \
        python3-pip \
        sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install EGL / GL / OSMesa dependencies
RUN apt-get update && \
    apt-get install -y \
        libegl1 \
        libegl1-mesa \
        libegl-mesa0 \
        libgl1-mesa-glx \
        libgl1-mesa-dri \
        libgles2 \
        libosmesa6 \
        libx11-6 \
        libxext6 \
        libxrender1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install git
RUN apt-get update && \
    apt-get install -y git
# CHANGE to your desired workspace name
RUN git config --global --add safe.directory /Lifelong-World-Model

# Make 'python' point to python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Upgrade pip, setuptools, wheel
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Attempt to install CUDA-enabled PyTorch wheels for cu121
RUN python -m pip install --no-cache-dir --upgrade \
    "torch" --index-url https://download.pytorch.org/whl/cu121 \
    torchvision --index-url https://download.pytorch.org/whl/cu121 \
    torchaudio --index-url https://download.pytorch.org/whl/cu121 || \
    (echo "WARNING: CUDA-enabled torch install may have failed or fallen back to CPU-only." && exit 0)

# Creates a non-root user with an explicit UID and adds permission to access the folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN mkdir -p /Lifelong-World-Model
WORKDIR /Lifelong-World-Model

# Setup user
RUN addgroup --gid ${USER_UID} ${USER_NAME} && \
    adduser --disabled-password --gecos "" --uid ${USER_UID} --gid ${USER_UID} ${USER_NAME} && \
    chown -R ${USER_UID}:${USER_UID} /home/${USER_NAME} && \
    chown -R ${USER_NAME}:${USER_NAME} /Lifelong-World-Model && \
    chmod -R 777 /Lifelong-World-Model

# Change the root access so we always have access as needed.
RUN echo 'root:root' | chpasswd
USER ${USER_NAME}

#Finally to fix proper permissions and imports, we modify the python sys path.
ENV PYTHONPATH="${PYTHONPATH}:/Lifelong-World-Model"
ENV HYDRA_FULL_ERROR=1
ENV TORCH_LOGS="-all"
ENV TORCHDYNAMO_VERBOSE=1
ENV WANDB_API_KEY=${WANDB_API_KEY}

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python3"]