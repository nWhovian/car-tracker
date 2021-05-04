FROM ubuntu:16.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    cpio \
    curl \
    sudo \
    lsb-release \
    software-properties-common \
    libgl1-mesa-glx \
    && \
    rm -rf /var/lib/apt/lists/*

RUN sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
    python3.6 \
    python3-setuptools \
    python3-pip \
    libpython3.6 \
    && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.6 /usr/bin/python3

RUN python3.6 -m pip install -U pip && \
    python3.6 -m pip install -U futures && \
    python3.6 -m pip install -U opencv-python==4.1.2.30 && \
    python3.6 -m pip install -U opencv-contrib-python && \
    python3.6 -m pip install -U numpy && \
    python3.6 -m pip install -U openvino && \
    python3.6 -m pip install -U scipy


COPY ./files/l_openvino_toolkit*.tgz /root/l_openvino_toolkit*.tgz


RUN cd /root/ && \
    tar -zxvf l_openvino_toolkit*.tgz && \
    rm l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit*/ && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg



RUN echo "source /opt/intel/openvino/bin/setupvars.sh" > /root/.bashrc



ADD main.py /

WORKDIR /root
