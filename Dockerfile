FROM public.ecr.aws/z9b3d4t5/jobs-container-keras-export-base:dce56bdc4398de7347fef07f7b5586ac8748061b

# ARG ARCH=
# ARG CUDA=11.2
# FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}.2-base-ubuntu${UBUNTU_VERSION} as base
# ARG CUDA
# ARG CUDNN=8.1.0.77-1
# ARG CUDNN_MAJOR_VERSION=8
# ARG LIB_DIR_PREFIX=x86_64
# ARG LIBNVINFER=8.0.0-1
# ARG LIBNVINFER_MAJOR_VERSION=8
# # Let us install tzdata painlessly
# ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Avoid confirmation dialogs
ENV DEBIAN_FRONTEND=noninteractive
# Makes Poetry behave more like npm, with deps installed inside a .venv folder
# See https://python-poetry.org/docs/configuration/#virtualenvsin-project
ENV POETRY_VIRTUALENVS_IN_PROJECT=true


# System dependencies
# libs required for opencv
# https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt update && apt install -y wget git python3 python3-pip zip libgl1 libgl1-mesa-glx libglib2.0-0

# Latest setuptools
RUN python3 -m pip install --upgrade setuptools

# RUN git clone https://github.com/experiencor/keras-yolo2 && \
#     cd keras-yolo2 && \
#     git checkout master

# Install extra dependencies here
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /scripts

# Copy the normal files (e.g. run.sh and the extract_dataset scripts in)
COPY . ./

# The train command (we run this from the keras venv, which has all dependencies)
ENTRYPOINT [ "./run-python-with-venv.sh", "keras", "train.py" ]
