ARG CUDA_VERSION=11.6.2
ARG CUDNN_VERSION=8
ARG UBUNTU_VERSION=18.04  # Python2 required
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION} as base

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# Install basic packages
RUN apt-get update && apt-get install -y \
    wget \
    vim \
    build-essential \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-distutils \
    python3-setuptools \
    python-pip \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libeigen3-dev \
    libyaml-cpp-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libdc1394-22-dev \
    libsndfile1 libsndfile1-dev  # For sndfile

# Create a user to map from docker host
ARG USER_ID
ARG GROUP_ID
ARG USER_NAME
RUN addgroup --gid $GROUP_ID $USER_NAME --force-badname
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER_NAME --force-badname

# Python environment and arguments
ENV LANG C.UTF-8
ARG PYTHON=python3.7
ARG PIP=pip3
ARG PYTHON2=python
ARG PIP2=pip2

# Install required Python packages
# RUN apt-get update && apt-get install -y ${PYTHON}-pip
# RUN apt-get update && apt-get install -y ${PYTHON2}-pip
RUN ${PIP} --no-cache-dir install --upgrade pip

# Install required pip packages
COPY ./requirements.txt ./
RUN ${PIP} install --upgrade pip \
   && ${PIP} install --no-cache-dir -r requirements.txt \
   && rm requirements.txt

# COPY ./requirements_python2.txt ./
# RUN ${PIP2} install --upgrade pip \
#    && ${PIP2} install --no-cache-dir -r requirements_python2.txt \
#   && rm requirements_python2.txt

# Specify working dir for a project
ARG WORKING_DIR=/usr/src/app
WORKDIR ${WORKING_DIR}

# Customize command prompt
RUN echo 'PS1="nes-gan>:\w\$ "' >> /etc/bash.bashrc

# Setting up user environment
USER ${USER_NAME}
ENV LANG C.UTF-8
