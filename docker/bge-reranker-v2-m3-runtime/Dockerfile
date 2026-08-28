FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir \
    transformers==4.57.3 \
    safetensors==0.6.2 \
    sentencepiece==0.2.1 \
    pandas==2.2.3 \
    pyarrow==21.0.0

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=true \
    PM_MAX_LENGTH=384 \
    PM_PAIR_BATCH_SIZE=256

WORKDIR /solution
ENTRYPOINT []
