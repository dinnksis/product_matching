FROM dinakepech/ecup26-embedding-catboost:1.0

# These Linux/Python 3.11 wheels are downloaded once outside Docker. The base
# image already contains torch, transformers, scipy, CatBoost and the Qwen
# weights, so the corrected runtime can be built without PyPI access.
COPY docker/embedding-catboost-runtime/st-wheels-linux/ /st-wheels/
RUN pip install --no-cache-dir --no-index --no-deps /st-wheels/*.whl && \
    rm -rf /st-wheels

# Replace the base-model-only metadata from image 1.0 with the exact
# SentenceTransformer structure used during training: Transformer + last-token
# pooling. The 1.19 GB weights and tokenizer remain in the cached base layer.
COPY docker/embedding-catboost-runtime/models/qwen3-embedding-0.6b/config.json \
    /opt/models/qwen3-embedding-0.6b/config.json
COPY docker/embedding-catboost-runtime/models/qwen3-embedding-0.6b/config_sentence_transformers.json \
    /opt/models/qwen3-embedding-0.6b/config_sentence_transformers.json
COPY docker/embedding-catboost-runtime/models/qwen3-embedding-0.6b/modules.json \
    /opt/models/qwen3-embedding-0.6b/modules.json
COPY docker/embedding-catboost-runtime/models/qwen3-embedding-0.6b/1_Pooling/config.json \
    /opt/models/qwen3-embedding-0.6b/1_Pooling/config.json

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PM_MODEL_DIR=/opt/models/qwen3-embedding-0.6b

WORKDIR /solution
ENTRYPOINT []
