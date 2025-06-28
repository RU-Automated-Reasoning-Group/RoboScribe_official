FROM ubuntu:22.04

RUN apt update
RUN apt -y install software-properties-common
RUN apt -y install vim 
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt -y install ffmpeg libsm6 libxext6
RUN apt -y install git unzip
RUN apt -y install libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev patchelf curl

RUN mkdir -p /root/.mujoco && \
    curl -L -o /tmp/mujoco210-linux-x86_64.tar.gz https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && \
    tar -xzf /tmp/mujoco210-linux-x86_64.tar.gz -C /root/.mujoco && \
    rm /tmp/mujoco210-linux-x86_64.tar.gz
# ENV MUJOCO_GL=osmesa
# ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt -y install python3.9-dev 
RUN apt -y install python3-pip 
RUN apt -y install python3.9-distutils
RUN python3.9 -m pip install pip==22.0.3
RUN python3.9 -m pip install numpy==1.23.0
RUN python3.9 -m pip install torch==1.11.0
RUN python3.9 -m pip install tqdm==4.62.3


RUN python3.9 -m pip install scikit-learn==0.24.2
RUN python3.9 -m pip install matplotlib==3.9.4
RUN python3.9 -m pip install opencv-python==4.10.0.84
RUN python3.9 -m pip install commentjson==0.9.0
RUN python3.9 -m pip install gymnasium-robotics==1.2.4
RUN python3.9 -m pip install gymnasium==0.28.1
RUN python3.9 -m pip install cython==0.29.24
RUN python3.9 -m pip install gym==0.21.0
RUN python3.9 -m pip install mujoco-py==2.1.2.14


RUN python3.9 -m pip install stable-baselines3==2.0.0a2
RUN python3.9 -m pip install wandb==0.16.3
RUN python3.9 -m pip install d4rl==1.1
RUN python3.9 -m pip install mujoco==3.2.2
RUN python3.9 -m pip install perlin-noise==1.12 
RUN python3.9 -m pip install pyrallis==0.3.1
RUN python3.9 -m pip install tensorboard==2.10.1

RUN echo "alias python='python3.9'" >> ~/.bashrc
RUN echo "alias python3='python3.9'" >> ~/.bashrc

COPY RoboScribe /RoboScribe

RUN apt -y install wget
RUN python3.9 -m pip install gdown

RUN gdown --fuzzy https://drive.google.com/file/d/1WeHusQ0cSnFGgBfJzOtz1sVnVPt7h-a_/view?usp=sharing -O /tmp/demo.zip

# Create the target directory (in case it doesn't exist)
RUN mkdir -p /RoboScribe/store/demo_store

# Unzip and copy contents to the destination
RUN unzip /tmp/demo.zip -d /tmp/demo_content \
 && cp -r /tmp/demo_content/* /RoboScribe/store/ \
 && rm -rf /tmp/demo.zip /tmp/demo_content

RUN gdown --fuzzy https://drive.google.com/file/d/1okHY_ZIRTudRCgY_4FXRF6namYtt95JE/view?usp=sharing -O /tmp/maniskill_data.zip
RUN unzip /tmp/maniskill_data.zip -d /tmp/maniskill_data_content \
 && cp -r /tmp/maniskill_data_content/* /RoboScribe/environment/ManiSkill2-0.4.2

RUN python3.9 -m pip install /RoboScribe/environment/ManiSkill2-0.4.2/

RUN apt -y install libvulkan1 mesa-vulkan-drivers

WORKDIR /RoboScribe
