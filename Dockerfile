# Start from the CUDA base image
FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install essentials
RUN apt update -y && apt upgrade -y && \
    apt-get --fix-missing install -y wget build-essential checkinstall  libreadline-gplv2-dev  libncursesw5-dev  libssl-dev  libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev libgtk2.0-dev pkg-config

# Update package lists and install Python 3.8
RUN apt-get update \
    && apt-get install -y python3.8 python3.8-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# install libraries for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 --index-url https://download.pytorch.org/whl/cu111

# Continue with your Dockerfile commands
# For example:
COPY requirements.txt /app/

RUN pip install -r requirements.txt --no-cache-dir


# Copy your application code into the container
COPY yolox /app/yolox

COPY yolox.egg-info /app/yolox.egg-info
COPY __init__.py /app/__init__.py
COPY README.md /app/README.md
COPY exps /app/exps
COPY hubconf.py  /app/hubconf.py
COPY tools /app/tools
COPY setup.py /app/setup.py
#yolox - editable mode allows changes to the source code to be immediately available without reinstallation.
RUN pip install -e .


