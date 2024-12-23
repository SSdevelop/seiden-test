FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# ARG MODEL="yolo_world_l_dual_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
# ARG WEIGHT="yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-0e566235.pth"

ENV FORCE_CUDA="1"
ENV MMCV_WITH_OPS=1

RUN apt-get update && apt-get install -y --no-install-recommends
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.10 \
    python3-pip     \
    python3-apt     \
    libgl1-mesa-glx \
    libsm6          \
    libxext6        \
    libxrender-dev  \
    libglib2.0-0    \
    git             \
    curl            \
    python3.10-dev  \
    python3-wheel
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN alias python3=python3.10
RUN pip3 install --upgrade pip
RUN pip3 install setuptools
RUN pip3 install opencv-python \
        supervision   \
        openmim       \
        wheel         \
        jupyterlab    \
    && pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
        --extra-index-url https://download.pytorch.org/whl/cu113
    # && pip3 install mmcv==2.0.0 mmdet==3.0.0 \
    # && mim install mmyolo==0.6.0

COPY . /yolo
WORKDIR /yolo

RUN pip3 install -e .
RUN mim install mmyolo==0.6.0

# RUN curl -o weights/$WEIGHT -L https://huggingface.co/wondervictor/YOLO-World/resolve/main/$WEIGHT

# ENTRYPOINT [ "python3", "demo.py" ]
# CMD ["configs/pretrain/$MODEL", "weights/$WEIGHT"]
