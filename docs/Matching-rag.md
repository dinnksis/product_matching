# Matching Policy RAG

## 1. Цель

Нужно улучшить качество LLM-разметчика товарных пар, особенно на hard examples.

Проблема текущего подхода: даже большая Qwen с хорошим статическим промптом плохо знает **внутреннюю семантику human-разметки конкретного датасета**. Например, изменение некоторого атрибута в одной категории может означать другой товар, а в другой категории — допустимый вариант того же товара.

Поэтому вместо попытки вручную сформулировать все правила предлагается автоматически восстановить **matching policy** из human-labeled train.

Основная идея:

\[
\text{human pairs}
\rightarrow
\text{label-agnostic difference extraction}
\rightarrow
\text{matching policy induction}
\rightarrow
\text{RAG knowledge base}
\rightarrow
\text{stronger Qwen annotator}
\]

Критический принцип: **на этапе извлечения различий Qwen НЕ получает human label.**

Qwen должна хорошо решить более простую задачу:

> «Чем эти две карточки отличаются?»

А уже настоящие human labels используются статистически, чтобы определить:

> «Какие из этих различий human-разметка считает допустимыми, а какие меняют identity товара?»

---

# 2. Общая архитектура

Система состоит из пяти основных компонентов:

1. **Difference Extractor** — Qwen, которая сравнивает две карточки без label.
2. **Difference Taxonomy + Difference DB** — нормализованное хранилище найденных различий.
3. **Policy Induction** — восстановление правил matching policy по human labels.
4. **Matching Policy RAG** — индекс правил и подтверждающих human-примеров.
5. **RAG Annotator** — Qwen, которая получает новую пару + релевантные правила и принимает решение match/non-match.

Дополнительный шестой компонент:

6. **Policy-guided Data Generator** — использует найденные правила для генерации новых hard examples.

---

# 3. Правильный train/evaluation split

Сначала human dataset необходимо один раз разделить на:

- `policy_train`
- `validation`
- желательно отдельный `hard_validation`

Все правила, статистика и RAG строятся **только на policy_train**.

Validation labels никогда не должны участвовать:

- в построении правил;
- в выборе supporting examples;
- в RAG index.

Иначе получится leakage.

`hard_validation` нужен отдельно, потому что основной смысл системы — улучшить как раз те случаи, где обычный Qwen prompt сейчас работает плохо.

---

# 4. Этап 1. Difference Extractor

## Задача Qwen

На вход:

- category;
- product A;
- product B.

Human label отсутствует.

Qwen запрещено:

- решать, match это или нет;
- объяснять, почему товары одинаковые/разные;
- использовать собственное понимание matching policy.

Она должна только перечислить **наблюдаемые семантические различия**.

Например, концептуально результатом могут быть такие atomic differences:

- отличается объём памяти;
- отличается размер;
- отличается цвет;
- отличается model number;
- отличается количество единиц в комплекте;
- один объект — аксессуар, второй — основной продукт;
- значение есть только в одной карточке;
- различается поколение модели;
- одна карточка более специфична;
- единицы измерения различаются, но значения эквивалентны.

## Что сохранять для каждого различия

Минимальный набор полей:

### `attribute`

Что именно сравнивается:

- storage;
- size;
- color;
- model;
- model_generation;
- quantity;
- volume;
- compatibility;
- product_type;
- bundle_count;
- material;
- flavour;
- и т.д.

### `relation`

Как именно отличается:

- `different`
- `missing_on_left/right`
- `more_specific`
- `equivalent_after_normalization`
- `different_quantity`
- `different_variant`
- `different_generation`
- `incompatible`
- и т.д.

### `left_value / right_value`

Конкретные значения из карточек.

Они нужны для debugging, поиска human-примеров и дальнейшей генерации.

### `evidence`

Фрагменты исходных карточек, на которых основан вывод.

Это нужно для защиты от hallucination: каждое extracted difference должно быть grounded в исходном тексте.

### `confidence`

Насколько Qwen уверена, что различие действительно присутствует.

### `category`

Категория товара.

Она критична: правило почти всегда потенциально category-dependent.

---

# 5. Этап 2. Построение taxonomy различий

Нельзя сразу прогонять все ≈310k human pairs в полностью свободном формате.

Иначе Qwen создаст:

- `storage`
- `memory`
- `storage_capacity`
- `rom_size`
- `phone_memory`

как пять разных понятий.

Поэтому используется двухэтапная схема.

## Discovery phase

Взять примерно 10–30k разнообразных human pairs и позволить Qwen относительно свободно именовать differences.

После этого собрать все найденные типы и кластеризовать/суммаризировать их.

Получить контролируемую taxonomy, например:

- `MODEL`
- `MODEL_GENERATION`
- `STORAGE`
- `RAM`
- `SIZE`
- `COLOR`
- `QUANTITY`
- `VOLUME`
- `BUNDLE_COUNT`
- `COMPATIBILITY`
- `PRODUCT_TYPE`
- `ACCESSORY_VS_MAIN`
- `MATERIAL`
- ...

Отдельно создать небольшой фиксированный vocabulary для relation types.

## Production extraction

После стабилизации taxonomy снова запускаем Difference Extractor, но теперь просим выбирать canonical attribute/relation из известного словаря.

Разрешаем `OTHER/UNKNOWN`, чтобы taxonomy можно было расширять.

После этого прогоняем весь `policy_train`.

---

# 6. Difference DB

Основной промежуточный артефакт системы.

Концептуально:

```text
pair_id
category
attribute
relation
left_value
right_value
extractor_confidence
evidence
```

Human label хранится отдельно и присоединяется уже после extraction.

Таким образом база разделяет:

\[
\text{observed differences}
\]

и

\[
\text{human decision}.
\]

Это принципиально.

---

# 7. Этап 3. Восстановление Matching Policy

Теперь для каждого human pair имеется:

\[
D_i=\{d_1,d_2,\ldots,d_k\}
\]

и настоящий label:

\[
y_i\in\{0,1\}.
\]

Нужно выяснить, какие типы differences связаны с изменением identity.

## Первый уровень: простая статистика

Для каждого:

\[
category \times attribute \times relation
\]

считать:

- support;
- количество match;
- количество non-match;
- `P(match | difference)`;
- доверительный интервал.

Это уже даст множество сильных кандидатов на правила.

Но нельзя напрямую считать:

> difference встретился в negative → difference ломает match.

В одной паре может быть пять различных differences, а решающим является только одно.

Поэтому нужны ещё два уровня.

---

# 8. Minimal contrasts

Особенно ценны human pairs с минимальным количеством различий.

Например, если найдено много пар, где:

- совпадает почти всё;
- систематически отличается только `SIZE`;
- human label = MATCH;

то это очень сильное свидетельство:

\[
SIZE\_DIFFERENCE
\rightarrow
tolerated
\]

в данной категории.

Аналогично для non-match.

Поэтому в Difference DB нужно специально искать:

- пары с одним difference;
- пары с двумя differences;
- группы максимально похожих difference signatures с разными labels.

Их следует считать наиболее сильным evidence при построении policy.

---

# 9. Интерпретируемая модель поверх differences

Следующий уровень — обучить небольшой вспомогательный classifier:

\[
category + extracted\ differences
\rightarrow human\ label.
\]

Это **не финальная matching model**.

Она нужна исключительно для анализа matching policy.

Подойдут:

- logistic regression с interaction features;
- shallow decision trees;
- Explainable Boosting Machine;
- CatBoost + SHAP/interactions.

Основные вопросы:

- какие differences сильнее всего двигают решение;
- какие эффекты зависят от category;
- какие combinations критичны;
- когда один difference становится важным только при наличии другого.

Например правило может оказаться не:

\[
COLOR\_DIFF \rightarrow NONMATCH
\]

а:

\[
MODEL\_SAME + COLOR\_DIFF \rightarrow MATCH.
\]

---

# 10. Формирование Matching Rules

После статистики и модели создаётся отдельная база **Policy Rules**.

Одно правило должно описывать:

- scope/category;
- difference или combination of differences;
- эффект;
- human support;
- confidence;
- supporting human examples;
- counterexamples.

Эффект лучше хранить не бинарно, а как:

- `BREAKS_MATCH`
- `TOLERATED_DIFFERENCE`
- `WEAK_EVIDENCE`
- `CONTEXT_DEPENDENT`
- `UNKNOWN`

Важно сохранять exceptions.

Цель системы — не получить красивый список абсолютных законов, а получить **эмпирическое описание human policy**.

Для первого MVP можно считать правило сильным, если:

- достаточно human support;
- большая часть evidence указывает в одну сторону;
- эффект сохраняется на minimal contrasts;
- интерпретируемая модель подтверждает тот же сигнал.

Точные thresholds потом подобрать на validation.

---

# 11. Matching Policy RAG

Теперь строится RAG не по миллионам свободных Qwen explanations, а по **human-grounded policy**.

Я бы сделал два индекса.

## Rule Index

Содержит агрегированные правила:

```text
category
difference
effect
confidence
support
exceptions
```

## Human Case Index

Содержит реальные human-labeled examples вместе с их extracted differences.

Это позволяет для новой пары получать одновременно:

- абстрактное правило;
- несколько реальных human demonstrations.

---

# 12. Retrieval для новой пары

Для новой пары сначала запускается тот же Difference Extractor.

Получаем:

\[
D_{query}.
\]

Retrieval query строится прежде всего из:

- category;
- canonical differences;
- relation types.

Далее ищутся:

1. наиболее специфичные policy rules;
2. supporting human examples;
3. при необходимости counterexamples.

Предпочтительный приоритет retrieval:

\[
exact\ structured\ match
>
category+attribute
>
semantic\ similarity.
\]

То есть если известно:

`category=shoes`, `attribute=size`, `relation=different`,

не нужно надеяться только на vector similarity — соответствующее правило можно достать напрямую structured lookup.

Dense retrieval нужен для более сложных или неизвестных случаев.

---

# 13. RAG Annotator

На вход Qwen получает:

1. исходную пару товаров;
2. основной matching prompt;
3. извлечённые differences;
4. несколько наиболее релевантных human-derived rules;
5. несколько supporting human examples;
6. при необходимости известные exceptions/counterexamples.

Теперь Qwen должна определить match/non-match.

Главное отличие от текущего подхода:

**модель больше не должна держать всю matching policy в весах или одном огромном prompt.**

Ей в каждый момент предоставляется маленький локальный фрагмент policy, релевантный текущей паре.

Также желательно заставлять annotator возвращать:

- prediction;
- confidence;
- IDs использованных rules.

Последнее сильно упростит error analysis.

---

# 14. Главный эксперимент

Нужно сравнить на одном frozen human hard-validation четыре варианта одной и той же Qwen:

### A. Baseline

Текущий статический prompt.

### B. Example RAG

Текущий prompt + похожие human pairs.

### C. Rule RAG

Текущий prompt + matching-policy rules.

### D. Hybrid RAG

Текущий prompt + rules + supporting human pairs.

Особенно важно смотреть не только overall metric, но:

- hard subset;
- per-category;
- accuracy по каждому difference type.

Это ответит на главный исследовательский вопрос:

> действительно ли восстановленная matching policy помогает LLM решать сложные пары?

---

# 15. Применение к 11M weak pairs

Если Hybrid RAG показывает заметный gain, его можно использовать как более сильного teacher.

Не обязательно сразу переразмечать все 11M.

Сначала выбрать, например:

- пары, где исходная weak label не уверена;
- disagreement между текущим student и weak label;
- hard pairs;
- редкие difference types;
- категории с плохим качеством.

Именно эти пары прогонять через RAG Annotator.

Таким образом computation тратится на наиболее информативные данные.

---

# 16. Генерация новых данных из Policy Rules

Та же база автоматически становится системой controlled generation.

Есть два основных типа.

## Hard negative generation

Берётся human-backed правило:

\[
difference\ X \rightarrow BREAKS\_MATCH
\]

и создаётся пара, где максимально мало отличий кроме `X`.

## Hard positive generation

Берётся:

\[
difference\ X \rightarrow TOLERATED
\]

и создаётся пара, которая сильно отличается по `X`, но должна оставаться match.

При этом preferable использовать:

- реальные товары;
- реальные значения атрибутов;
- реальные human examples как demonstrations.

LLM должна генерировать вокруг подтверждённого правила, а не самостоятельно решать, что считается match.

---

# 17. Coverage-driven generation

После обучения matcher его ошибки также размечаются Difference Extractor'ом.

Можно построить таблицу:

```text
difference type
human support
training support
validation count
model error rate
```

И выбирать для генерации те области, где одновременно:

- высокая ошибка matcher;
- мало train examples;
- правило достаточно хорошо известно.

Получается feedback loop:

\[
train
\rightarrow
error\ analysis
\rightarrow
difference\ gaps
\rightarrow
targeted\ generation
\rightarrow
retrain.
\]

Это гораздо предпочтительнее случайного добавления synthetic data.

---

# 18. Prompt для Difference Extractor

Основная инструкция должна по смыслу быть примерно такой:

> Сравни две карточки товара и перечисли все содержательные различия между ними.
>
> Не пытайся определить, являются ли товары одинаковыми.
> Не пытайся объяснить match/non-match.
> Human label тебе неизвестен и не должен быть выведен.
>
> Для каждого различия определи:
> - какой semantic attribute различается;
> - значения на левой и правой стороне;
> - тип отношения между значениями;
> - фрагменты входа, подтверждающие вывод;
> - уверенность.
>
> Разделяй независимые различия на отдельные atomic differences.
> Не добавляй различия, которые невозможно подтвердить исходными карточками.
> Если два значения различаются текстуально, но семантически эквивалентны после нормализации, укажи это отдельно.
> Используй заданную canonical taxonomy атрибутов и relations. Если подходящего класса нет, используй OTHER.

---

# 19. Prompt для RAG Annotator

По смыслу:

> Определи, являются ли две карточки одним товаром согласно matching policy данного датасета.
>
> Тебе предоставлены правила, полученные статистически из human-разметки, и реальные human examples.
>
> Рассматривай эти правила как evidence о политике датасета, а не как универсальные знания о товарах.
>
> Более специфичные и хорошо подтверждённые category-specific rules имеют приоритет перед общими.
>
> Если несколько правил конфликтуют, учитывай их human support, specificity и приведённые exceptions.
>
> Используй исходные карточки как основной объект решения, а retrieved knowledge — для восстановления того, как подобные различия трактуются human-разметкой.

---

# 20. Quality control Difference Extractor

Это критический компонент: если extractor плохой, вся система сверху будет плохой.

До полного запуска нужно проверить хотя бы несколько тысяч примеров вручную/полуручно.

Основные тесты:

### Grounding

Каждое различие должно подтверждаться исходными данными.

### Symmetry

Если поменять Product A и Product B местами, должен получаться тот же набор differences с переставленными значениями.

### Stability

Повторные запуски не должны радикально менять canonical difference type.

### Coverage

Extractor не должен систематически пропускать важные различия вроде model number / quantity / size.

### Taxonomy quality

Не должно существовать десятков разных canonical названий одного и того же concept.

---

# 21. MVP-порядок реализации

Не нужно сразу строить всю систему на 310k.

### MVP 1

10–20k human pairs → free difference extraction → построение taxonomy.

### MVP 2

Зафиксировать taxonomy → извлечь differences для 50k human pairs.

Проверить:

- качество extraction;
- statistics `difference ↔ label`;
- наличие очевидных и интерпретируемых rules.

Если здесь ничего разумного не возникает, дальше систему не масштабировать.

### MVP 3

На этих же данных построить первые Policy Rules + Rule RAG.

Сравнить:

`baseline Qwen` vs `Rule RAG`

на frozen hard-validation.

### MVP 4

Добавить retrieval похожих human examples:

`Rule RAG` vs `Rule + Example RAG`.

### MVP 5

Если есть gain — прогнать Difference Extractor по всему policy_train и построить полноценную Policy DB.

### MVP 6

Использовать RAG Annotator для targeted relabeling части 11M weak pairs.

### MVP 7

Policy-guided generation + стандартный эксперимент:

\[
310k
\quad vs \quad
310k + 10k\ generated.
\]

---

# 22. Что должно получиться в итоге

У системы должно быть четыре основных reusable artifact:

**1. Difference Taxonomy**
Единый язык описания различий между товарами.

**2. Difference DB**
Human pairs, разложенные на atomic differences.

**3. Matching Policy DB**
Эмпирические правила вида:

\[
category + difference
\rightarrow
effect\ on\ identity
\]

с human support, confidence и exceptions.

**4. Matching Policy RAG**
Интерфейс, который по новой паре возвращает наиболее релевантные правила и supporting human examples.

После этого одна и та же инфраструктура используется и для:

- повышения качества Qwen-разметчика;
- переразметки weak data;
- анализа ошибок matcher;
- поиска неизвестных областей matching policy;
- controlled hard-positive/hard-negative generation.

## Центральная гипотеза

Основная гипотеза проекта:

> Большая часть ошибок LLM на hard product matching возникает не из-за неспособности сравнить два текста, а из-за незнания специфической matching policy датасета.

Поэтому Qwen следует использовать там, где она сильна:

\[
\text{semantic difference extraction}
\]

а matching policy получать из того источника, которому мы действительно доверяем:

\[
\boxed{\text{human labels}}
\]

и затем возвращать эту policy модели через RAG в момент принятия решения.
