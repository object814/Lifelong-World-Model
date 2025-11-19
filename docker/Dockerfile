FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Build args (can be passed from docker-compose / .env)
ARG WORKSPACE_DIR="/workspace"
ARG USERNAME=developer
ARG UID=1000
ARG GID=1000

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV WORKSPACE_DIR=${WORKSPACE_DIR}
ENV HOME=/home/${USERNAME}
ENV JUPYTER_TOKEN=""

# Install system packages and sudo
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget curl ca-certificates \
    python3 python3-dev python3-pip python3-venv pkg-config \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    swig ffmpeg libavcodec-dev libavformat-dev libswscale-dev \
    libjpeg-dev libpng-dev libsdl2-2.0-0 libsdl2-dev \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Make 'python' point to python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Upgrade pip, setuptools, wheel
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Create group and user with provided UID/GID (aligns file ownership with host) and enable passwordless sudo
RUN set -eux; \
    # create group if it doesn't exist, then user
    if ! getent group ${GID} >/dev/null 2>&1; then addgroup --gid ${GID} ${USERNAME}; fi; \
    # adduser: disabled password, no interactive prompts; ensure uid/gid match build args
    adduser --disabled-password --gecos '' --uid ${UID} --gid ${GID} ${USERNAME} || true; \
    # create sudoers.d entry for passwordless sudo (safer than echoing into /etc/sudoers directly)
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME}; \
    chmod 0440 /etc/sudoers.d/${USERNAME}; \
    # ensure home exists and correct ownership
    mkdir -p /home/${USERNAME}; \
    chown -R ${UID}:${GID} /home/${USERNAME}

# Create and set permissions for workspace
RUN mkdir -p ${WORKSPACE_DIR} && chown -R ${UID}:${GID} ${WORKSPACE_DIR}

WORKDIR ${WORKSPACE_DIR}

# Install Python packages (system-level pip install)
RUN python -m pip install --no-cache-dir \
    jupyterlab notebook matplotlib pandas tqdm cloudpickle tensorboard \
    imageio-ffmpeg

# Gymnasium extras, Atari, Box2D
RUN python -m pip install --no-cache-dir \
    "gymnasium[accept-rom-license]" \
    "gymnasium[box2d]" \
    "gymnasium[atari]" \
    ale-py==0.8.1

# Stable-Baselines3 and HuggingFace SB3 helper
RUN python -m pip install --no-cache-dir stable-baselines3==2.0.0a5 huggingface-sb3

# Attempt to install CUDA-enabled PyTorch wheels for cu121
RUN python -m pip install --no-cache-dir --upgrade \
    "torch" --index-url https://download.pytorch.org/whl/cu121 \
    torchvision --index-url https://download.pytorch.org/whl/cu121 \
    torchaudio --index-url https://download.pytorch.org/whl/cu121 || \
    (echo "WARNING: CUDA-enabled torch install may have failed or fallen back to CPU-only." && exit 0)

# Extra RL goodies
RUN python -m pip install --no-cache-dir stable-baselines3[extra] box2d-py

# Switch to the runtime (non-root) user
USER ${USERNAME}
ENV PATH=${HOME}/.local/bin:${PATH}

# Expose Jupyter port
EXPOSE 8888

# Default command: start Jupyter Lab in the workspace
CMD bash -lc "jupyter lab --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token='${JUPYTER_TOKEN}' --NotebookApp.password='' --notebook-dir=${WORKSPACE_DIR}"
