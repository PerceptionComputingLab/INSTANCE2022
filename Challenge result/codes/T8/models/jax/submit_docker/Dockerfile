FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64 LIBRARY_PATH=/usr/local/cuda-11.6/lib64

RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN pip3 install --upgrade nibabel wandb optax dm-haiku ml_collections absl-py

RUN pip3 install git+https://github.com/e3nn/e3nn-jax.git


COPY *.pkl /home/
COPY functions.py /home/functions.py
COPY model.py /home/model.py
COPY config.py /home/config.py
COPY evaluate.py /home/evaluate.py
COPY diffeomorphism.py /home/diffeomorphism.py

CMD ["python3", "/home/evaluate.py"]

# To Build:
#   $ docker build -t scan .

# To Run:
#   $ docker run --gpus "device=0" -v PATH:/input -v PATH:/predict scan:latest