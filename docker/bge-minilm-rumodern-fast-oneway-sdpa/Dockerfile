FROM dinakepech/ecup26-bge-minilm-benefit-router-sdpa:1.0

RUN python -c "from transformers import ModernBertForSequenceClassification; from sentence_transformers import CrossEncoder; print('ModernBERT + CrossEncoder runtime: OK')"

ENV PM_MAX_LENGTH=384 \
    PM_BATCH_SIZE=1024 \
    PM_MINILM_COVERAGE=0.40 \
    PM_RUMODERN_COVERAGE=0.05 \
    PM_ATTENTION_IMPLEMENTATION=sdpa

WORKDIR /solution
