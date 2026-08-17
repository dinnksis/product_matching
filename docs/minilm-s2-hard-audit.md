# Аудит MiniLM S2 на hard validation

Дата: 15 августа 2026 года.

Воспроизводимый запуск:

```powershell
python scripts/analyze_s2_hard_validation.py
```

Артефакты находятся в `reports/minilm_s2_hard_audit/`.

## Главный вывод

Hard-набор действительно содержит предполагаемые hard negatives и hard
positives, но он не является обычной случайной validation. Он специально
собран по ошибкам и неожиданностям старых моделей:

- 2 600 `lexical_surprise` anchors;
- 2 080 уверенных ошибок MiniLM v1;
- 520 model-disagreement/order-gap anchors;
- 614 соседних пар из тех же компонент.

Поэтому hard metric измеряет устойчивость к известным слабостям предыдущего
пайплайна, а не ожидаемый leaderboard score. Настройка модели непосредственно
под этот набор создаст новый selection bias.

S2 остаётся основным вариантом. CatBoost статистически улучшает hard macro AP,
но ухудшает IID и не исправляет именно самые интересные numeric/code/SKU-срезы.
Глобально включать его по текущим результатам не следует.

## Итоговые метрики

| Модель | IID macro AP | Hard macro AP | OOD macro AP |
|---|---:|---:|---:|
| MiniLM S2 | 0.738074 | 0.307319 | 0.598185 |
| S2 + CatBoost | 0.736722 | 0.314517 | 0.600112 |
| Delta | -0.001352 | **+0.007198** | +0.001927 |

Для hard delta парный стратифицированный bootstrap даёт 95% CI
`[+0.000648; +0.012736]`; положительная delta получена в 98,8% ресэмплов.
Bootstrap по 18 категориям даёт `[+0.001260; +0.013661]` и 99,27% положительных
ресэмплов.

Эффект при этом концентрирован: 72,6% суммы покатегорийных улучшений дают
ювелирные изделия, детские товары и хобби; пять лучших категорий дают 96,5%.
CatBoost улучшает 12 категорий и ухудшает 6.

## Чем hard отличается от IID

| Срез | Hard: доля | IID: доля | Enrichment | Hard positive rate | IID positive rate |
|---|---:|---:|---:|---:|---:|
| Полностью одинаковый normalized title | 3.59% | 0.85% | **4.23×** | 2.39% | 37.25% |
| Title token Jaccard ≥ 0.6 | 52.92% | 26.33% | **2.01×** | 9.91% | 38.18% |
| Exact extracted code | 7.14% | 4.33% | 1.65× | 14.22% | 41.81% |
| Critical conflict | 34.45% | 35.97% | 0.96× | **21.17%** | 11.26% |
| Numeric-context conflict | 20.50% | 21.91% | 0.94× | **21.73%** | 9.85% |
| Code conflict | 9.24% | 10.22% | 0.90× | **20.30%** | 8.48% |
| SKU с одной стороны | 10.70% | 14.64% | 0.73× | **42.93%** | 21.34% |

Hard не просто содержит больше конфликтов. В нём радикально меняется связь
признака с target: высокая похожесть названий почти всегда ведёт к negative, а
numeric/code conflicts заметно чаще, чем в IID, встречаются среди positive.
Это прямое следствие отбора `lexical_surprise` и ошибок старой MiniLM.

## Что происходит на предполагаемых hard-срезах

| Срез | Пар | Positive rate | S2 AP | CatBoost AP | Delta |
|---|---:|---:|---:|---:|---:|
| Все hard | 5 814 | 25.47% | 0.307319 | 0.314517 | +0.007198 |
| Высокая похожесть title | 3 077 | 9.91% | 0.212129 | 0.208130 | **-0.003999** |
| Critical conflict | 2 003 | 21.17% | 0.391977 | 0.388981 | **-0.002996** |
| Numeric-context conflict | 1 192 | 21.73% | 0.423533 | 0.422679 | -0.000854 |
| Unit conflict | 586 | 30.72% | 0.446235 | 0.441957 | -0.004278 |
| Code conflict | 537 | 20.30% | 0.442266 | 0.405286 | **-0.036980** |
| Attribute model conflict | 479 | 19.83% | 0.346190 | 0.387208 | **+0.041018** |
| SKU/human asymmetry | 622 | 42.93% | 0.480227 | 0.466382 | **-0.013845** |

Текущий CatBoost не решает поставленные классы ошибок:

- numeric conflict практически не меняется;
- code conflict и SKU/human asymmetry становятся хуже;
- заметный локальный выигрыш есть только для model conflict, извлечённого из
  явно названных attribute keys.

Это важный аргумент в пользу key-aware представления. S2 видит значения, но не
знает, означает ли `16 gb` RAM, накопитель, размер упаковки или другую величину.
В текущем feature parser надёжный `memory_conflict` найден всего у 17 пар, так
что делать вывод о памяти как отдельном классе пока нельзя.

## Обрезание и порядок пары

- 1 088 hard-пар (18,71%) упираются в token budget 256;
- в IID доля почти идентична: 18,66%; enrichment только 1.003×;
- hard AP на budget-hit парах равен 0.329563, без budget hit — 0.311777;
- средний `|f(A,B)-f(B,A)|` на hard равен 0.0473 против 0.0386 на IID;
- верхние 10% order-unstable пар начинаются с gap 0.1318.

Обрезание может портить отдельные карточки, но не объясняет общий провал hard.
Симметричный AB/BA inference уже уменьшает влияние порядка; полностью проблема
не исчезает, однако сейчас это вторичный фактор.

## Кандидаты на повторную проверку human labels

В hard-наборе:

- 204 negative с полностью одинаковым normalized title;
- 424 positive с critical conflict;
- 259 positive с numeric-context conflict;
- 109 positive с конфликтующими product codes.

Это не автоматически ошибочные labels, но крайние примеры плохо согласуются со
строгим правилом «одинаковая SKU/модификация».

Примеры negative с практически идентичной карточкой:

- `датчик Quattro Freni QF02B00006` против того же title и партномера;
- `MZ-CHG-12-3SKULL2/B чехол для классической гитары` против того же кода;
- `Lacalut Aktiv ... 75 мл` против того же продукта и объёма.

Примеры positive с существенными различиями:

- линзы с axis `170` против axis `100`;
- удилище `240 cm, 15–40 g` против `300 cm, 25–70 g`;
- кроссовки Geox размера 32 против кед Geox размера 31 другого цвета и материала;
- Country Life B-complex 60 капсул против magnesium glycinate 90 таблеток.

Возможны три объяснения: annotation noise, family-level политика вместо строгого
variant matching или важный контекст, отсутствующий в видимых полях. До
уточнения политики нельзя просто повышать вес всех numeric conflicts: среди них
259 размеченных positives, и такое перевзвешивание систематически испортит их.

## Решение и следующий шаг

1. Оставить **MiniLM S2 без CatBoost** текущим default: CatBoost ухудшает IID и
   уже показал худший leaderboard score.
2. Вручную перепроверить около 200 пар:

   - 50 negative exact-title;
   - 50 positive numeric/code conflicts;
   - 50 SKU/human positives и negatives;
   - 50 крупнейших CatBoost improvements/regressions.

3. Зафиксировать явное правило: SKU-level match или product-family match.
4. Если labels подтверждаются, следующим serialization experiment делать не
   новый общий sweep, а один key-aware вариант: title, затем приоритетные
   `brand/model/article/memory/size/quantity/color` как обычный `key: value`,
   остальные атрибуты — values-only в оставшемся token budget.
5. Если значимая часть крайних labels ошибочна, не учить модель воспроизводить
   эти противоречия. Сначала создать adjudicated hard audit и использовать его
   только для оценки, а не для подбора порога на текущем hard-наборе.

## Артефакты

- `summary.json` — общие метрики и два bootstrap;
- `hard_vs_iid_slices.csv` — сравнение distribution hard и IID;
- `slice_metrics.csv` — AP/FPR/FNR по типам пар;
- `label_slice_scores.csv` — positive/negative score behavior;
- `category_metrics.csv` — покатегорийные результаты;
- `top_s2_false_positives.csv` и `top_s2_false_negatives.csv` — крайние ошибки;
- `catboost_rank_improvements.csv` и `catboost_rank_regressions.csv` — пары,
  которые meta-model сильнее всего переставил в ранжировании;
- `label_policy_contradiction_pairs.csv` — компактные пары аналогичных кейсов,
  где один human target равен 1, а другой 0;
- `label_policy_contradiction_pairs_full.csv` — те же 91 сопоставление с полными
  диагностическими полями и previews атрибутов;
- `hard_diagnostics.parquet` — все 5 814 пар с диагностическими признаками.
