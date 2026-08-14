FROM dinakepech/ecup26-minilm-s2:1.0

RUN pip install --no-cache-dir \
    scikit-learn==1.9.0 \
    scipy==1.17.1 \
    rapidfuzz==3.14.5 \
    joblib==1.5.3 \
    catboost==1.2.8

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=true

WORKDIR /solution
ENTRYPOINT []
