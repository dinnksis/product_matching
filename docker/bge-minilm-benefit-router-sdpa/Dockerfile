FROM dinakepech/ecup26-bge-minilm-sentence-transformers-sdpa:1.0

RUN pip install --no-cache-dir --no-deps --timeout 1800 --retries 20 \
        catboost==1.2.8 \
        rapidfuzz==3.14.5 && \
    python -c "from catboost import CatBoostClassifier; from rapidfuzz import fuzz; from sentence_transformers import CrossEncoder; print('router runtime imports: OK')"

ENV PM_MAX_LENGTH=384 \
    PM_BATCH_SIZE=1024 \
    PM_ROUTE_COVERAGE=0.40 \
    PM_ATTENTION_IMPLEMENTATION=sdpa

WORKDIR /solution
