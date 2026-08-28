FROM dinakepech/ecup26-bge-reranker-v2-m3:1.0

RUN pip install --no-cache-dir \
        sentence-transformers==5.1.2 && \
    python -c "import sentence_transformers, torch; print(torch.__version__, sentence_transformers.__version__)"

ENV TOKENIZERS_PARALLELISM=true \
    RAYON_NUM_THREADS=20 \
    OMP_NUM_THREADS=20 \
    MKL_NUM_THREADS=20 \
    PM_MAX_LENGTH=192 \
    PM_BATCH_SIZE=1024 \
    PM_ATTENTION_IMPLEMENTATION=sdpa

WORKDIR /solution
