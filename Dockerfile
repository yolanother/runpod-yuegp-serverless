# Base image -> https://github.com/runpod/containers/blob/main/official-templates/base/Dockerfile
# DockerHub -> https://hub.docker.com/r/runpod/base/tags
FROM runpod/base:0.4.0-cuda11.8.0

# The base image comes with many system dependencies pre-installed to help you get started quickly.
# Please refer to the base image's Dockerfile for more information before adding additional dependencies.
# IMPORTANT: The base image overrides the default huggingface cache location.
RUN apt update && apt install -y --no-install-recommends \
    ffmpeg git git-lfs

# --- Optional: System dependencies ---
# COPY builder/setup.sh /setup.sh
# RUN /bin/bash /setup.sh && \
#     rm /setup.sh


# Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt
RUN python3.11 -m pip install --upgrade --no-cache-dir torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu124

# NOTE: The base image comes with multiple Python versions pre-installed.
#       It is reccommended to specify the version of Python when running your code.

RUN git lfs install
RUN git clone https://github.com/deepbeepmeep/YuEGP/ /YuE

# switch to YuEGP directory
WORKDIR /YuE
RUN git clone https://huggingface.co/m-a-p/xcodec_mini_infer /YuEGP/inference/xcodec_mini_infer
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Add src files (Worker Template)
ADD src .

CMD python3.11 -u /handler.py
