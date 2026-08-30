# Final-checkpoint full one-way BGE + MiniLM + RuModernBERT

This is a separate submission from `bge-minilm-rumodern-full-oneway-st-v1`.
It uses the final checkpoints from `configs/*_final` and does not overwrite the
old archive.

- all three models score 100% of pairs;
- one `A -> B` pass, no symmetric pass;
- SentenceTransformers CrossEncoder, FP16, SDPA;
- batch size 1024 and combined pair `max_length=384`;
- equal mean of the three sigmoid probabilities;
- no CatBoost or routing.

Serialization is copied from `src/data_pipeline.py` and matches final training:

```text
Категория: <category>
Название: <name>
<attribute key>: <attribute value>
...
```

The builder accepts a single safetensors file, Hugging Face sharding, or Kaggle
transport chunks. BGE transport chunks are verified and joined only in staging;
files under `configs/` are not modified.

Runtime image:

```text
dinakepech/ecup26-bge-minilm-rumodern-router-sdpa:1.0
```

Despite the historical image name, this runner does not import or execute CatBoost.

Build:

```powershell
.\.venv\Scripts\python.exe scripts\build_bge_minilm_rumodern_full_oneway_submit.py
```

Output: `submits/bge-minilm-rumodern-full-oneway-final.zip`.
