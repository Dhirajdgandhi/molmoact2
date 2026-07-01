# GPU image for offline MolmoAct2 evaluation on ephemeral SageMaker / EC2 hosts.
# Build:  docker build -t molmoact2-eval .
# Run:    docker run --gpus all \
#           -e HF_TOKEN -e GIT_USERNAME -e GIT_TOKEN \
#           -e GIT_USER_NAME -e GIT_USER_EMAIL \
#           -v molmoact2-hf-cache:/tmp/huggingface \
#           molmoact2-eval --checkpoint dhirajdg/molmoact2-record-test-step3000-eval03562-20260625

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_XET=1 \
    HF_HOME=/tmp/huggingface \
    HUGGINGFACE_HUB_CACHE=/tmp/huggingface/hub \
    TRANSFORMERS_CACHE=/tmp/huggingface/hub \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    MOLMOACT2_OUTPUT_DIR=/tmp/molmoact2-record-test/outputs \
    GIT_USER_NAME="Dhiraj Gandhi" \
    GIT_USER_EMAIL=ddgandhi.96@gmail.com

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY scripts/requirements.txt scripts/requirements.txt
RUN pip install --no-cache-dir -r scripts/requirements.txt

COPY scripts/ scripts/

WORKDIR /app/scripts

RUN chmod +x docker_entrypoint.sh sync_secrets.sh

ENTRYPOINT ["./docker_entrypoint.sh"]
CMD ["--help"]
