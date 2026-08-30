# Итоговые результаты и история экспериментов

Состояние на 30 августа 2026 года. Этот документ отделяет воспроизводимые
результаты на human validation от leaderboard и от экспериментов, которые были
только подготовлены или запущены в dry-run режиме.

Полный журнал запусков, включая неудачные и промежуточные варианты:
[Google Sheets — experiments](https://docs.google.com/spreadsheets/d/1CtqT52XOrFyHfFt6rCiOMlnq6snJMlsMOJ0ubH79ikA/edit?usp=sharing).
Тематические листы `experiments_v2`, `pretrain_exps`, `sft_exps` и `data_exps`
содержат runtime, macro AP, paired p-value, Holm correction и bootstrap CI там,
где эти величины были применимы.

## Протокол оценки

Исходная human-разметка содержит 365 654 пары и 711 304 карточки. Validation
формировалась component-disjoint: ни одна карточка не встречается одновременно
в train и validation. Основная метрика — средний category-wise
`average_precision_score`.

Для BGE SFT использовались:

- исходный human train: 306 669 пар;
- бывший OOD split: 41 171 пара;
- вместе: 347 840 пар, 89 291 positive, все 20 категорий;
- IID validation: 12 000 пар, 18 категорий;
- hard validation: 5 814 пар, 18 категорий.

Former OOD включён в train осознанно: категории из него уже присутствовали в
BGE pretraining. Поэтому OOD не является validation для BGE, predictions для
него не строятся, а в отчётах записывается sentinel `-1`. IID — единственный
primary split; hard используется как диагностика и не выбирает рецепт.

## Итоговая таблица

| семейство | рецепт | IID macro AP | hard macro AP | решение |
|---|---|---:|---:|---|
| MiniLM | 3 ep, LR `8e-5`, BCE | 0.808502 | 0.423286 | лучший MiniLM anchor |
| BGE | 1 ep, LR `1e-5`, BCE | 0.813638 | 0.411162 | отвергнут |
| BGE | 1 ep, LR `2e-5`, BCE | 0.818291 | 0.414717 | baseline |
| BGE | 1 ep, LR `4e-5`, BCE | 0.815148 | 0.409547 | отвергнут |
| BGE | 2 ep, LR `2e-5`, BCE | **0.823461** | **0.437775** | controlled winner |
| BGE | 2 ep, LR `2e-5`, sqrt category×class BCE | 0.822150 | 0.431759 | plain BCE сохранён |
| BGE H100 | 3 ep, LR `2e-5`, BCE, symmetric | **0.824975** | **0.461148** | финальный checkpoint |
| BGE H100 submission | single-pass longer-first | 0.823629 | 0.462631 | быстрый inference |

Двухэпоховый BGE дал относительно одноэпохового baseline `+0.005170` IID и
`+0.023057` hard macro AP. Для IID paired permutation p-value равен примерно
`0.064`, а component-bootstrap 95% CI — `[0.000004, 0.009875]`. В заранее
зафиксированном протоколе выбор делался по practical improvement `>0.002`,
поэтому был выбран e2, но результат не следует описывать как доказанный на
уровне `p<0.05`.

Трёхэпоховый H100 run стартовал не из тех же байтов checkpoint, что controlled
Kaggle-линия. Он подтверждает качество финального артефакта, но не является
чистой оценкой эффекта третьей эпохи. Его checkpoint SHA-256:
`d7e899ea3cd305db970aa6f3466eb71a138ad418c74b8b6ac730d1828c4a4ab8`.

## Как был выбран рецепт

### MiniLM

Для MiniLM проверялись learning rate, число эпох, effective batch, warmup,
weight decay, label smoothing, classifier dropout и несколько loss-функций.
Лучшим устойчивым anchor остался `minilm5_sft_e3_lr8e5_v1`: 3 эпохи,
LR `8e-5`, plain BCE, dropout `0.1`. Dropout `0` и `0.2`, а также дополнительные
regularization coordinates не дали практического IID-прироста.

Ранний 14-run sweep по `max_length ∈ {384, 512}`, LR
`{5e-6, 1e-5, 2e-5, 3e-5, 5e-5}` и горизонту 1–3 эпох дал лучшие отдельные
значения: IID `0.803345` (`512`, `2e-5`, 3 ep), hard `0.396926` (`384`,
`2e-5`, 3 ep) и OOD `0.653021` (`512`, `5e-5`, 1 ep). Эта линия была
предварительной; последующий locked campaign с другим выбранным pretrain
checkpoint поднял MiniLM anchor до IID `0.808502`.

### BGE

BGE checkpoint — `XLMRobertaForSequenceClassification`: 24 слоя, hidden size
1024, 16 attention heads, примерно 568M параметров. В controlled-линии сначала
сравнивались `1e-5`, `2e-5`, `4e-5` на одной эпохе. Оба соседа проиграли
`2e-5`, после чего горизонт был увеличен до двух эпох. E2 прошёл practical
IID-gate. Единственная перенесённая loss-абляция — sqrt category×class
reweighting — ухудшила IID на `0.001311` и hard на `0.006015`, поэтому финально
оставлен plain BCE.

На H100 был отдельно выполнен трёхэпоховый запуск с global effective batch 192,
BF16, SDPA, `max_length=384`, LR `2e-5`. Для submission symmetric A/B+B/A
заменён на детерминированный single-pass: первой ставится более длинная карточка.
Это почти сохраняет IID и укладывает inference в более реалистичный runtime.

## Ансамбль BGE + MiniLM

Фиксированный read-only probe проверил только два заранее заданных веса logit
blend. Вариант `0.7 × BGE e2 + 0.3 × MiniLM` дал IID macro AP `0.826023`, то
есть `+0.002563` к BGE e2, но hard macro AP снизился до `0.435617`
(`-0.002158`). Поэтому blend интересен при оптимизации именно текущего IID, но
консервативный single-model выбор — BGE. Это не доказательство прироста на
hidden test.

## Финальные ансамбли и routing

После одиночных моделей были подготовлены три production-oriented варианта:

- full BGE + MiniLM: обе модели считают 100% пар, probabilities смешиваются
  50/50;
- hierarchical 40/5: BGE считает все пары, compact CatBoost направляет 40% в
  MiniLM и 5% внутри этого subset в RuModernBERT;
- full triple: равное среднее probabilities BGE, MiniLM и RuModernBERT.

Подтверждённый runtime anchor — full BGE+MiniLM в one-way
SentenceTransformers/FP16/SDPA реализации. Для routed 40/5 и full triple
актуальный competition runtime не подтверждён: предыдущие H100-прогоны падали на
startup RuModernBERT до законченного forward. Их builders и bundles сохранены,
но README не заявляет, что они укладываются в лимит.

Routing models обучались component-disjoint на OOF предыдущих neural
checkpoints. Они полезны как эксперимент по compute allocation, но их AP нельзя
автоматически переносить на финальные all-human checkpoints без свежего OOF.
Подробный протокол и таблицы находятся в
[`final-ensemble-architecture.md`](final-ensemble-architecture.md).

MiniLM и RuModernBERT также имеют отдельный финальный all-human stage: после
выбора параметров на holdout каждый специалист дообучается на всех 365 654 human
парах, без validation и без дальнейшего выбора гиперпараметров. См.
[`final-human-finetuning.md`](final-human-finetuning.md).

## Генерация и расширение данных

В репозитории сохранён отдельный воспроизводимый `item_pipeline` для
standalone-генерации карточек и rule-first построения пар. Были реализованы:

- Qwen-based генерация карточек с resume и строгой схемой provenance;
- deterministic surface-positive augmentation;
- human-positive resampling и filtered-positive наборы;
- near-duplicate генерация;
- soft-positive tiers A/B;
- статистические и semantic atomic-rule каталоги;
- проверки совместимости двух правил, нормализация значений и audit receipts.

Эти направления дали инфраструктуру для контролируемых data-абляций, но ни одно
из них не заменило финальный human-only BGE рецепт. Точные строки запусков и
статусы находятся в Google Sheets; устройство pipeline описано в
[`item_pipeline/README.md`](../item_pipeline/README.md) и
[`surface-positive-augmentation.md`](surface-positive-augmentation.md).

11M probabilistic LLM labels исследовались отдельно. Они не смешаны с финальным
human SFT без отдельного решения: noise, confidence и leakage требуют иного
протокола. Документы и server trainer сохранены для воспроизводимости, а не как
заявление о финальном выигрыше.

## Неудачные, но полезные направления

- lexical/JSON HistGradientBoosting baseline: OOF macro AP около `0.535`;
- Qwen3-Reranker-0.6B zero-shot: macro AP около `0.428`, но слишком медленно;
- Jina reranker v2 zero-shot: прошёл runtime, leaderboard около `0.25`;
- BGE zero-shot: throughput хороший, но train macro AP только `0.4712`;
- дополнительный BGE loss reweighting не улучшил plain BCE;
- повышение/понижение LR относительно `2e-5` ухудшило IID;
- synthetic/data-generation ветки не дали достаточно надёжного прироста для
  включения в финальный рецепт.

Эти результаты важны: они сократили пространство поиска до plain supervised
BGE и показали, что более сложный loss или генерация сами по себе не гарантируют
улучшения ранжирования.

## Воспроизводимость и артефакты

Основные точки входа:

- controlled BGE campaign: `configs/bge_2ep_sft_hparam_search_v1.json` и
  `scripts/run_bge_2ep_sft_candidates.py`;
- BGE loss screen: `scripts/run_bge_2ep_sft_loss_confirmation.py`;
- H100 three-epoch workflow: `scripts/run_bge_3ep_h100.py`;
- submission builder: `scripts/build_bge_3ep_h100_submit.py`;
- submission source: `submits/bge-reranker-v2-m3-3ep-h100/`;
- exact comparison receipts: `reports/bge_2ep_sft_candidate_v1/` и
  `reports/bge_2ep_sft_loss_confirmation_v1/`;
- MiniLM reports: `reports/minilm_5ep_sft_hparam_search_v1/`.

Финальные веса намеренно не хранятся в Git. Приватный Kaggle Dataset:
[product-matching-bge-3ep-h100-oodtrain](https://www.kaggle.com/datasets/alexproger23/product-matching-bge-3ep-h100-oodtrain),
version 1. Dataset model-only; он не содержит локальных путей, prediction или
training data. Manifest SHA-256:
`7715f63243e5ba1fca7accc60060985f577423a1ac7d4a8fb26c343677ef9f35`.

Raw Parquet, checkpoints, private credentials и локальные `artifacts/`
игнорируются Git. Два воспроизводимых статистических файла превышают жёсткий
лимит GitHub 100 MiB; ещё один 84-МБ CSV является производной таблицей с
многострочными полями. Все три остаются локальными:

- `reports/atomic_rule_statistics_current/rule_statistics.csv`;
- `reports/atomic_rule_statistics_semantic_snapshot_20260826/rule_statistics.csv`;
- `reports/semantic_atomic_rule_statistics_all_pairs_20260827/prototype_embeddings.npy`.

Их генераторы, конфигурации, компактные summaries и остальные результаты
сохраняются в репозитории.
