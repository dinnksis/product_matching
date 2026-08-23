FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir \
    transformers==4.57.6 \
    safetensors==0.6.2 \
    pandas==2.2.3 \
    pyarrow==21.0.0

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=true

WORKDIR /solution
ENTRYPOINT []
