FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    s3fs \
    awscli \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libglew-dev \
    libegl1-mesa-dev \
    xvfb \
    mesa-utils \
    wget \
    make \
    git \
    libglib2.0-0 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    libfontconfig1 \
    libdbus-1-3 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxrender1 \
    libfontconfig1 \
    libxi6 \
    libxkbcommon-x11-0 \
    libsm6 \
    libice6 \
    libfreetype6 \
    libx11-xcb1

# Set up miniconda
ENV CONDA_DIR=/opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p ${CONDA_DIR} && \
    rm ~/miniconda.sh

# Add conda to path
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Some fucking bullshit
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main https://repo.anaconda.com/pkgs/r

# Create conda environment
RUN conda create -y -n ksim python=3.11 && \
    conda install -y -c conda-forge libstdcxx-ng -n ksim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Initialize conda environment and install dependencies
SHELL ["conda", "run", "-n", "ksim", "/bin/bash", "-c"]
RUN pip install --upgrade --upgrade-strategy eager -r requirements.txt
RUN pip install ruff mypy

# Set up display environment variables
ENV DISPLAY=:101.0
ENV MUJOCO_GL=egl

# Create entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"] 