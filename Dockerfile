# Use a base image with CUDA 12.3 and Python 3.9
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies including Python 3.9
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.9 \
    python3.9-venv \
    python3.9-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

COPY requirements.txt /app/requirements.txt

# Install Python dependencies
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN pip3 install torch torchvision torchaudio
RUN pip install "monai[einops]"
RUN pip install "gudhi"
RUN pip install "torch-geometric"

# Install PyTorch, TensorBoard, and other dependencies
RUN pip install torch torchvision torchaudio tensorboard
RUN pip install wandb
## Now copy the rest of your project code for prod # comment when dev
#COPY . /app
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6
RUN pip install nibabel

# Set the default command to run the main script
CMD ["python3", "main.py"]
