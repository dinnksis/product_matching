# Эксперименты Qwen embeddings + CatBoost

## Общая схема

Эксперименты обучались только на 365 654 парах с ручной разметкой. Использовался
component-disjoint split: связанные компоненты графа товаров целиком попадали
либо в train, либо в validation. Получилось:

- train: 310 767 пар, positive rate `25.67%`;
- validation: 54 887 пар, positive rate `25.74%`;
- пересечение товарных ID между частями: 0;
- на категорию в validation приходится примерно 2.5–3.5 тысячи пар.

CatBoost обучался с весами, выравнивающими вклад 20 категорий, поскольку метрика
соревнования — среднее AP по категориям. Все приведённые ниже scores до отдельной
пометки являются локальным macro AP на этой validation, а не leaderboard score.

Общий код первых трёх экспериментов:

- реализация: `src/embedding_boosting.py`;
- конфигурация: `configs/embedding_boosting.json`;
- Kaggle launcher: `scripts/run_embedding_boosting_kaggle.py`;
- артефакты: `artifacts/kaggle/product-matching-qwen-embedding-boosting/embedding_boosting/`.

## 1. CatBoost по лексическим признакам названий

Эксперимент `01_names_lexical` не использует нейросетевые embeddings и
атрибуты. Для пары названий рассчитываются:

- обычное, token-set и token-sort сходство;
- полное совпадение;
- отношение и разница длин;
- Jaccard чисел из названий и наличие чисел с обеих сторон;
- one-hot категории.

Результат:

- macro AP: `0.501538`;
- overall AP: `0.595118`;
- 28 признаков;
- обучение CatBoost: `16.2 с` на GPU.

Артефакты:
`artifacts/kaggle/product-matching-qwen-embedding-boosting/embedding_boosting/01_names_lexical/`.

Это важный быстрый baseline: значительная часть качества достигается без
нейросети. Он также должен быть безопасным fallback, но текущий submission
использовал не эту обученную модель, а простую формулу из двух признаков.

## 2. Названия + Qwen3-Embedding-0.6B

Эксперимент `02_names_qwen_embedding` добавляет item embeddings названий из
`Qwen/Qwen3-Embedding-0.6B`:

- каждое уникальное название кодируется один раз;
- max length 96;
- используются первые 256 координат последнего токена;
- вектор L2-нормализуется;
- для пары строятся cosine, L1/L2/max distance, 256 абсолютных разностей и
  256 попарных произведений координат;
- лексические признаки эксперимента 1 сохраняются.

Результат:

- macro AP: `0.540683`;
- overall AP: `0.623562`;
- 544 признака;
- обучение CatBoost: `55.1 с` на GPU.

Артефакты:
`artifacts/kaggle/product-matching-qwen-embedding-boosting/embedding_boosting/02_names_qwen_embedding/`.

Прирост относительно эксперимента 1: `+0.03915` macro AP.

## 3. Названия + Qwen embeddings + структурированные атрибуты v1

Эксперимент `03_names_qwen_attributes` дополняет эксперимент 2 признаками из
JSON `attributes`. JSON не передаётся в модель одной сырой строкой. Он парсится,
ключи и значения нормализуются, после чего для пары считаются:

- число общих ключей и Jaccard наборов ключей;
- число и доля совпавших/конфликтующих значений;
- разница количества атрибутов и полное совпадение словарей;
- `both`, `match`, `conflict` для семейств brand, model, identifier, size,
  quantity, color, material, country и seller-related полей;
- до пяти наиболее информативных ключей отдельно для каждой категории.

Информативные ключи выбирались только на train по различию positive rate между
совпадением и конфликтом значения при достаточной поддержке. Это предотвращает
прямую утечку validation labels при выборе ключей.

Результат:

- macro AP: `0.574783`;
- overall AP: `0.646257`;
- 599 признаков;
- обучение CatBoost: `58.7 с` на GPU.

Артефакты:
`artifacts/kaggle/product-matching-qwen-embedding-boosting/embedding_boosting/03_names_qwen_attributes/`.

Прирост атрибутов относительно эксперимента 2: `+0.03410` macro AP.

### Конкурсный submission эксперимента 3

Для него были подготовлены:

- runner и модель: `submits/qwen-embedding-catboost/`;
- архив: `submits/qwen-embedding-catboost.zip`;
- Docker image: `dinakepech/ecup26-embedding-catboost:1.0`;
- Dockerfile: `docker/embedding-catboost-runtime/Dockerfile`.

На сайте соревнования submission получил `0.215914`, что резко расходится с
локальными `0.574783`. Split уже повторно проверен и выглядит корректно.
Основные оставшиеся гипотезы:

1. submission достиг soft deadline и вместо CatBoost использовал упрощённый
   lexical fallback;
2. embeddings из `SentenceTransformer` при обучении не совпали с ручным
   `AutoModel + last-token pooling` в контейнере;
3. в контейнер попала другая ревизия Qwen weights;
4. после исключения технических причин — сильный distribution shift скрытого
   теста.

После первого диагностического запуска обнаружена ошибка реализации submission:
обучение использовало `SentenceTransformer.encode` в `float16`, а контейнер —
ручной `AutoModel + last-token pooling` в `bfloat16`. Кроме того, контейнер мог
молча заменить CatBoost упрощённой формулой при достижении soft deadline. Эти
ветки признаны некорректными: из runner удалены ручной inference и fallback.
Исправленный runner должен использовать тот же SentenceTransformer snapshot и
тот же encode path, что и обучение. Повторный submission нельзя собирать до
проверки полного snapshot, воспроизведения validation и замера времени.

Fallback воспроизведён на локальной validation и даёт macro AP `0.397445`, то
есть один fallback не объясняет всё расхождение, но на другом распределении мог
просесть сильнее.

Для проверки запущен отдельный Kaggle kernel
`dinakepecheva/product-matching-submission-diagnostic`. Его локальный notebook:
`notebooks/qwen_embedding_submission_diagnostic.ipynb`. После завершения сюда
нужно добавить фактический score submission pipeline, корреляцию с исходными
validation predictions и строку лога о CatBoost/fallback.

## 4. Улучшенные структурированные атрибуты v2

Эксперимент `04_global_attributes_v2` оставляет старые names-only Qwen embeddings
из эксперимента 2, но расширяет обработку JSON-атрибутов:

- fuzzy similarity значений общих ключей: среднее и максимум;
- token Jaccard всех значений;
- нормализация измерений и единиц, признаки совпадения/конфликта измерений;
- fuzzy similarity отдельно внутри семейств brand/model/identifier/size и др.;
- до 12 информативных ключей на категорию вместо 5;
- минимальная поддержка ключа снижена с 200 до 100;
- CatBoost увеличен до 1600 деревьев, depth 9.

Важно: этот четвёртый результат ещё не использует rich-text embeddings имени и
атрибутов и не является экспериментом с 512 embedding dimensions. Это отдельные
последующие варианты `05` и `06`, отчёты которых нужно зафиксировать после
полного скачивания Kaggle output.

Результат эксперимента 4:

- macro AP: `0.596205`;
- overall AP: `0.660582`;
- 641 признак;
- обучение CatBoost: `118.8 с` на GPU.

Прирост относительно эксперимента 3: `+0.02142` macro AP; относительно
лексического baseline: `+0.09467`.

Где лежит:

- реализация: `src/attribute_boosting_v2.py`;
- конфигурация: `configs/attribute_boosting_v2.json`;
- launcher: `scripts/run_attribute_boosting_v2_kaggle.py`;
- скачанные артефакты:
  `artifacts/kaggle/product-matching-attribute-boosting-v2/attribute_boosting_v2/04_global_attributes_v2/`;
- удалённый завершённый kernel:
  `dinakepecheva/product-matching-attribute-boosting-v2`.

## Сводная таблица

| № | Эксперимент | Names lexical | Qwen name embeddings | Attributes | Local macro AP |
|---:|---|:---:|:---:|:---:|---:|
| 1 | `01_names_lexical` | да | нет | нет | `0.501538` |
| 2 | `02_names_qwen_embedding` | да | 256 dim | нет | `0.540683` |
| 3 | `03_names_qwen_attributes` | да | 256 dim | v1 | `0.574783` |
| 4 | `04_global_attributes_v2` | да | те же 256 dim | v2 | `0.596205` |

## Выводы и правила для следующих шагов

- Структурированные атрибуты дают устойчивый локальный прирост: сначала
  `+0.0341`, затем ещё `+0.0214` macro AP.
- Нельзя считать локальный результат пригодным для submission, пока точный
  submission runner не воспроизвёл его на той же validation.
- Координатные embedding-признаки требуют абсолютно идентичной модели, ревизии,
  tokenizer, pooling, truncation и dtype. Одного совпадения cosine недостаточно.
- Более надёжный вариант для переноса — сократить зависимость CatBoost от
  отдельных координат и проверить модель на инвариантных pair summaries:
  cosine/distances плюс lexical/attribute features.
- В основном submission не должно быть молчаливой замены алгоритма. Если позже
  потребуется fallback, это должен быть отдельно валидированный submission или
  явно проверенный ансамбль, а не ручная формула по таймеру.
- Перед следующим Docker push обязательны: exact-runner validation на Kaggle,
  сохранённый runtime log, замер числа required items и проверка ветки fallback.
- Следующее решение выбирается только после завершения текущей диагностики.
