# Аудит hard labels и clean-hard benchmark

Дата расчёта: 16 августа 2026 года.

Воспроизводимый запуск:

```powershell
python scripts/audit_hard_clean.py
```

Исполненный notebook: `notebooks/05_hard_clean_audit.ipynb`. Артефакты
находятся в `reports/minilm_s2_hard_clean_audit/`.

## Главное

Низкий результат на hard нельзя объяснить только ошибками разметки.

- Объективный contradictory duplicate найден только один: это одна пара полных
  нормализованных представлений, встречающаяся в трёх строках с targets `1, 0,
  0`. Повторных unordered ID-пар с противоположными target нет.
- Ещё 882 строки, или 15,17% hard, имеют сильные label-dependent признаки
  неоднозначности. Это не доказанные ошибки и их labels не исправлялись.
- После их отделения остаётся 4 929 clean-пар. Macro ROC-AUC S2 увеличивается с
  `0.5847` до `0.6391`, поэтому неоднозначные labels действительно вредят
  оценке ranking.
- Но `0.6391` всё ещё намного ниже IID `0.8816` и OOD `0.8318`. Значит, основная
  часть провала hard сохраняется и связана с реальной сложностью/selection
  shift, а не только с label noise.

## Строгое определение subsets

Товары нормализуются через существующий `normalize_text`; полное
представление включает category, title и отсортированные нормализованные
`key/value` attributes.

`hard_conflicting`:

- одна unordered ID-пара имеет оба target; или
- одна unordered пара полных нормализованных representations имеет оба target.

`hard_suspicious` не содержит definite conflict, но выполняется хотя бы одно:

- negative имеет одинаковый normalized title или полностью одинаковые
  representations товаров;
- positive имеет numeric, product-code, model-code или critical-attribute
  conflict.

`hard_clean` не содержит ни definite conflict, ни этих сильных label-dependent
флагов.

`sku_vs_human_title` сохранён как отдельный secondary flag и сам по себе не
исключает строку. Наличие извлечённого SKU только с одной стороны не доказывает
ошибку target. Predictions S0/S2/CatBoost в построении subsets не используются.

## Размеры

| Subset | Пар | Positive | Prevalence |
|:---|---:|---:|---:|
| hard_all | 5 814 | 1 481 | 25,47% |
| hard_clean | 4 929 | 802 | 16,27% |
| hard_suspicious | 882 | 678 | 76,87% |
| hard_conflicting | 3 | 1 | 33,33% |

Сильные suspicious-флаги асимметричны относительно label: отделено 204
negative exact-title и 678 positive conflict строк. Поэтому prevalence clean
резко падает. Сырые AP hard_all и hard_clean нельзя использовать как оценку
эффекта очистки.

Покрытие отдельных флагов до удаления пересечений:

| Флаг | Строк |
|:---|---:|
| Negative exact normalized title | 204 |
| Positive numeric conflict | 259 |
| Positive code conflict | 109 |
| Positive model-code conflict | 95 |
| Positive critical-attribute conflict | 403 |
| SKU ↔ human-title asymmetry | 622 |

## Frozen-model метрики

Competition macro AP:

| Dataset/subset | Prevalence | S0 title | S2 values | S2 + CatBoost |
|:---|---:|---:|---:|---:|
| IID | 25,98% | 0,683217 | **0,738074** | 0,736722 |
| OOD | 22,24% | 0,503071 | 0,598185 | **0,600112** |
| hard_all | 25,47% | 0,307657 | 0,307319 | **0,314517** |
| hard_clean | 16,27% | 0,221657 | 0,236681 | **0,247444** |
| hard_suspicious | 76,87% | **0,886120** | 0,831469 | 0,855665 |

Сравнивать модели можно внутри строки. Сравнивать абсолютные AP разных строк
без поправки на prevalence нельзя. Особенно misleading высокий AP
hard_suspicious: positive prevalence там 76,87%.

Macro ROC-AUC как дополнительная prevalence-insensitive диагностика:

| Dataset/subset | S0 title | S2 values | S2 + CatBoost |
|:---|---:|---:|---:|
| IID | 0,8544 | **0,8816** | 0,8827 |
| OOD | 0,7714 | 0,8318 | **0,8356** |
| hard_all | **0,5947** | 0,5847 | 0,5845 |
| hard_clean | 0,6048 | 0,6391 | **0,6439** |
| hard_suspicious | 0,5948 | 0,5347 | **0,6325** |

На clean-hard S2 снова превосходит S0, то есть attribute values полезны после
отделения наиболее противоречащих label-policy случаев. CatBoost добавляет к S2
`+0.01076` macro AP на clean-hard, но одновременно даёт `-0.00135` на IID и
ранее не прошёл leaderboard gate. Этот результат не обосновывает глобальную
замену S2 на CatBoost ensemble.

## Что остаётся сложным в clean-hard

Одноклассовые slices не имеют определённого PR-AUC, поэтому для них сохранены
mean score и FPR/FNR при 0,5.

- 494 SKU ↔ human-title пары содержат оба класса. Macro ROC-AUC равен `0.4309`
  у S0, `0.4276` у S2 и `0.4178` у CatBoost. Политика соответствия SKU и
  human-readable title остаётся неразрешённой проблемой.
- На 380 clean negative с model conflict FPR равен 13,95% у S0, 21,05% у S2 и
  21,32% у CatBoost. Values-only не знает, что конкретное значение является
  моделью, а CatBoost это не исправляет.
- На 297 very-high-similarity negative S2 имеет FPR 17,17%; CatBoost — 18,52%.
- На 269 very-low-similarity positive S2 имеет FNR 52,79%; CatBoost — 51,30%.

Это подтверждает, что даже после консервативной очистки остаются настоящие hard
negatives и hard positives.

## Объективный конфликт

Три строки относятся к детским товарам. Два разных item ID имеют одно полное
нормализованное представление товара «мягкая плюшевая игрушка ... гусь обнимусь
130 см» и сравниваются с одним и тем же представлением второй карточки. Одна
строка размечена positive, две negative. Все три системы закономерно дают
одинаковые scores каждой строке, поэтому выполнить эти labels одновременно
невозможно.

Эти три строки исключаются из clean benchmark, но targets автоматически не
исправляются.

## Итоговое использование трёх tests

- IID — основной dev benchmark для выбора модели.
- OOD — проверка переноса на две полностью отложенные категории.
- hard_all — adversarial stress test, сохраняющий реальную смесь сложных и
  спорных случаев.
- hard_clean — более строгий robustness benchmark без доказанных конфликтов и
  наиболее сильных label-policy anomalies.
- hard_suspicious — очередь на ручную adjudication, а не train set.
- hard_conflicting — исключить из метрики до ручного решения.

Настраивать модель непосредственно по hard_clean или hard_suspicious не следует:
оба subsets являются специально отобранными и могут создать новый selection
bias.

## Артефакты

- `hard_clean.csv`, `hard_suspicious.csv`, `hard_conflicting.csv` — subsets с
  titles, attributes, flags и всеми frozen scores;
- `label_audit_summary.csv` — размеры и prevalence;
- `benchmark_metrics.csv` — S0/S2/CatBoost на IID, OOD и hard subsets;
- `hard_clean_slice_metrics.csv` — clean-hard slices;
- `label_contradictions.csv` — до 100 примеров каждого audit issue;
- `label_audit_flag_summary.csv` — покрытие флагов;
- `audit_report.json`, `report.md`, `COMPLETED` — машинная и краткая сводки.
