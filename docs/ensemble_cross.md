Задача: binary product matching, основная метрика — macro AP по категориям.

1. Обученные модели

На Kaggle 2×T4 проверены 4 cross-encoder модели, по одной human-эпохе:

- GTE: Alibaba-NLP/gte-multilingual-reranker-base
- RuModernBERT: deepvk/RuModernBERT-base
- BGE: BAAI/bge-reranker-v2-m3
- MiniLM: checkpoint после 5 эпох synthetic pretraining + human FT

Общие условия:
- serialization: S2_VALUES_ONLY
- max_length=384
- symmetric inference
- splits не менялись: ordinary/IID, hard, OOD

Ноутбуки:
notebooks/architecture_baselines/

Predictions:
preds/preds_gte/
preds/preds_rumodernbert/
preds/preds_bge/
preds/preds_minilm/

В каждой папке лежат predictions для IID/hard/OOD и training report.

Одиночные результаты macro AP:

BGE:
- ordinary 0.782222
- hard 0.375975
- OOD 0.641270
- mean 0.599822

MiniLM:
- ordinary 0.782508
- hard 0.368075
- OOD 0.616523
- mean 0.589035

RuModernBERT:
- ordinary 0.773296
- hard 0.350399
- OOD 0.633022
- mean 0.585572

GTE:
- ordinary 0.761850
- hard 0.348445
- OOD 0.615033
- mean 0.575109

2. Ансамбли

Перебраны все одиночные модели, пары, тройки и all-4 с:
- mean probability
- mean normalized rank

Главные результаты:

Лучший overall:
RuModernBERT + BGE + MiniLM, mean probability
- ordinary 0.796635
- hard 0.366133
- OOD 0.648387
- mean 0.603718

Лучшая пара:
BGE + MiniLM, mean rank
- ordinary 0.792348
- hard 0.374699
- OOD 0.640976
- mean 0.602674

Лучший OOD:
RuModernBERT + BGE, mean rank
- OOD 0.649710

Лучший hard:
BGE отдельно
- hard 0.375975

All-4:
- mean AP 0.601127
- всего +0.001305 относительно BGE
- на −0.002591 хуже лучшей тройки

GTE — самая слабая одиночная модель, а добавление её в all-4 ухудшает результат относительно тройки. Поэтому runtime-работа дальше шла без GTE.

Результаты:
reports/architecture_ensemble_v1/ensemble_results.csv
reports/architecture_ensemble_v1/ensemble_summary.json

Diversity/error analysis:
reports/architecture_diversity_v1/
- pairwise_diversity.csv
- model_uniqueness.csv
- hard_vs_minilm.csv
- correlation matrices для каждого split
- diversity_summary.json

3. Ускорение inference

Проверены BGE, MiniLM и RuModernBERT на 5k парах, Tesla T4.

Самый быстрый из native-вариантов:
PyTorch/Transformers + FP16 + SDPA + pretokenization +
exact-length bucketing + dynamic padding.

T4 результаты:

BGE:
- batch 32
- 84.94 с / 5k
- 58.87 pairs/s
- baseline был 93.02 с

MiniLM:
- batch 192
- 14.46 с
- 345.71 pairs/s
- baseline 16.67 с

RuModernBERT:
- batch 192
- 47.29 с
- 105.74 pairs/s
- baseline 73.98 с

Отчёт:
reports/inference_backend_benchmark_v1/report.md
reports/inference_backend_benchmark_v1/winner_summary.csv

Дополнительно проверены SentenceTransformers и vLLM:

SentenceTransformers:
- predictions эквивалентны
- BGE 121.42 с против native 107.34 с end-to-end
- MiniLM 20.17 с против native 15.35 с
- медленнее native

vLLM 0.14:
- predictions эквивалентны
- BGE 244.49 с, 29.05 pairs/s
- MiniLM 50.42 с, 175.40 pairs/s
- значительно медленнее native

Итоговый backend: native Transformers/PyTorch.
Не использовать SentenceTransformers или vLLM.

4. Competition submissions

Docker image:
dinakepech/ecup26-bge-reranker-v2-m3:1.0

BGE отдельно:
submits/bge-reranker-v2-m3-human-ft-v1.zip

Он прошёл старый 20-минутный запуск примерно за 14 минут и получил leaderboard
PR-AUC около 0.43. Это не доказывает прохождение текущего private-лимита 13 минут.

Полный оптимизированный BGE + MiniLM:
submits/bge-minilm-rank-ensemble-optimized-v2.zip

Он НЕ уложился по времени: несмотря на оптимизированный native inference,
BGE всё ещё обрабатывал 100% пар.

5. Быстрый каскад

Пытались использовать CatBoost C1A_category_no_rules как negative router.
Сам predict быстрый, но построение 118 признаков заняло 90 секунд на 59k пар,
то есть ориентировочно около 7 минут на private 275k. Поэтому C1A не подходит
как скоростной router.

Вместо него сделан MiniLM-gated BGE:

- MiniLM считается на всех парах
- BGE вызывается только при MiniLM probability > 0.0602078252
- порог зафиксирован по median ordinary validation
- примерно 50% ordinary пар идут в BGE
- rejected: MiniLM probability
- routed: mean(MiniLM probability, BGE probability)

Validation:
- ordinary 0.791841, delta −0.001017
- hard 0.373169, delta −0.000311
- OOD 0.639201, delta −0.002114

Архив:
submits/bge-minilm-minilm-gated-v1.zip

SHA-256:
395ae7ee3503cab65d58f8e3d0d7a957dde66bf693d4222fb3c84f1d0d30ae68

Этот gated archive собран, но его competition runtime ещё не подтверждён.

Cascade-анализ:
reports/c1a_bge_minilm_cascade_v1/
- mini_all_bge_routed_sweep.csv
- routing_sweep.csv
- summary.json

6. Google Sheets

Основные experiment/ensemble/diversity результаты записывались в:
- architecture_exps

Inference benchmarks:
- inference_benchmarks

Spreadsheet ID:
1CtqT52XOrFyHfFt6rCiOMlnq6snJMlsMOJ0ubH79ikA

Следующий логичный шаг:
засабмитить submits/bge-minilm-minilm-gated-v1.zip и проверить реальное
end-to-end время на H100. Если не проходит 13 минут — повысить фиксированный
MiniLM gate, уменьшая долю BGE, используя уже сохранённый routing sweep.