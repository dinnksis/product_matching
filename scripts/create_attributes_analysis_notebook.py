#!/usr/bin/env python3
"""Generate a reproducible notebook for category-aware JSON attribute analysis."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "02_attributes_analysis.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb["metadata"]["language_info"] = {"name": "python", "version": "3.11"}
    nb["cells"] = [
        markdown("""
# Анализ JSON-атрибутов для product matching

Цель — решить, какие сигналы из `attributes` стоит добавить к item embeddings и
бустингу. Мы не создаём глобальную таблицу из 34 тысяч ключей. Вместо этого
измеряем покрытие и сигнал ключей по категориям, семантические семейства полей и
pair-features: совпадения, конфликты, пропуски и числа.

По умолчанию pair-анализ использует стратифицированную выборку для скорости.
После проверки notebook установите `PAIR_SAMPLE_SIZE = None`, чтобы пересчитать
все human-пары.
"""),
        code("""
from pathlib import Path
from collections import Counter, defaultdict
import json, re, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score

ROOT = Path.cwd()
if not (ROOT / 'data').exists():
    ROOT = ROOT.parent
ITEMS_PATH = ROOT / 'data/items_human.parquet'
MATCHES_PATH = ROOT / 'data/matches.parquet'
REPORT_DIR = ROOT / 'reports/attributes_analysis'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PAIR_SAMPLE_SIZE = 50_000  # None = все пары
MIN_KEY_SUPPORT = 500
TOP_KEYS_PER_CATEGORY = 30
RANDOM_SEED = 42

items = pd.read_parquet(ITEMS_PATH, columns=['id', 'name', 'attributes', 'category'])
matches = pd.read_parquet(MATCHES_PATH, columns=['id1', 'id2', 'target'])
print(f'items={len(items):,}, pairs={len(matches):,}, categories={items.category.nunique()}')
"""),
        markdown("""
## 1. Безопасный разбор и базовая структура

Нормализация здесь минимальная: whitespace и регистр ключа. Значения не очищаем
агрессивно, чтобы не потерять дефисы, единицы, артикулы и модельные номера.
"""),
        code(r'''
SPACE_RE = re.compile(r'\s+')
NUMBER_RE = re.compile(r'\d+(?:[.,]\d+)?')

def clean(value):
    if value is None:
        return ''
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return SPACE_RE.sub(' ', str(value)).strip()

def normalize_key(key):
    return clean(key).casefold()

def normalize_value(value):
    return clean(value).casefold().replace('ё', 'е')

def parse_attributes(raw):
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        return {}
    return {normalize_key(k): normalize_value(v) for k, v in obj.items()
            if normalize_key(k) and normalize_value(v)}

started = time.perf_counter()
attribute_dicts = [parse_attributes(raw) for raw in items.attributes]
items_per_attribute = np.array([len(value) for value in attribute_dicts])
print(f'parsed in {time.perf_counter()-started:.1f}s')
pd.Series(items_per_attribute).describe(percentiles=[.5,.9,.95,.99])
'''),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.histplot(np.clip(items_per_attribute, 0, 50), bins=50, ax=axes[0])
axes[0].set(title='Количество непустых атрибутов на товар', xlabel='attributes count')
items.assign(attr_count=items_per_attribute).groupby('category').attr_count.median().sort_values().plot.bar(ax=axes[1])
axes[1].set(title='Медианное число атрибутов по категории', ylabel='median')
plt.tight_layout()
"""),
        markdown("""
## 2. Частота, покрытие и кардинальность ключей

Хороший структурный признак должен встречаться достаточно часто внутри категории.
Кардинальность помогает отличить словарные поля (`цвет`, `материал`) от почти
уникальных идентификаторов (`артикул`) и потенциального мусора.
"""),
        code("""
key_rows = []
for category, indices in items.groupby('category').groups.items():
    support = Counter()
    values = defaultdict(Counter)
    value_lengths = defaultdict(list)
    for index in indices:
        for key, value in attribute_dicts[index].items():
            support[key] += 1
            values[key][value] += 1
            value_lengths[key].append(len(value))
    category_size = len(indices)
    for key, count in support.items():
        key_rows.append({
            'category': category, 'key': key, 'support': count,
            'coverage': count/category_size, 'unique_values': len(values[key]),
            'unique_ratio': len(values[key])/count,
            'median_value_length': float(np.median(value_lengths[key])),
            'top_value_share': values[key].most_common(1)[0][1]/count,
        })
key_stats = pd.DataFrame(key_rows)
key_stats.to_csv(REPORT_DIR/'key_stats_by_category.csv', index=False)
display(key_stats.sort_values(['category','support'], ascending=[True,False]).groupby('category').head(10))
"""),
        code("""
global_keys = Counter(key for attrs in attribute_dicts for key in attrs)
global_key_stats = pd.DataFrame(global_keys.most_common(), columns=['key','support'])
global_key_stats['coverage'] = global_key_stats.support / len(items)
display(global_key_stats.head(40))
"""),
        markdown("""
## 3. Семантические семейства

Точные имена ключей сильно различаются между категориями. Семейства дают
небольшое стабильное пространство признаков, но ниже мы всё равно отдельно
оцениваем лучшие точные ключи каждой категории.
"""),
        code(r'''
FAMILY_PATTERNS = {
    'brand': r'бренд|brand|производител',
    'model': r'модель|model|серия|линейка',
    'identifier': r'артикул|партномер|part.?number|sku|mpn|oem|код товара',
    'size': r'размер|длина|ширина|высота|диаметр|толщина|габарит',
    'quantity': r'количеств|комплект|упаков|штук|шт\.?$|объем|объ.м|вес',
    'color': r'цвет|оттенок',
    'material': r'материал|состав|сырье|сырьё',
    'country': r'страна|производств',
    'seller_noise': r'продав|магазин|поставщик|валюта|цена|достав|гарант',
}
FAMILY_REGEX = {name: re.compile(pattern) for name, pattern in FAMILY_PATTERNS.items()}

def key_family(key):
    for family, pattern in FAMILY_REGEX.items():
        if pattern.search(key):
            return family
    return 'other'

key_stats['family'] = key_stats.key.map(key_family)
family_coverage = (key_stats.groupby(['category','family']).support.sum()
                   .reset_index().pivot(index='category', columns='family', values='support').fillna(0))
display(family_coverage)
'''),
        markdown("""
## 4. Стратифицированная выборка пар

Сэмплируем внутри комбинации `category × target`, чтобы редкие категории и
положительные примеры не исчезли. Для финального расчёта переключите на все пары.
"""),
        code("""
categories = items.set_index('id').category
pairs = matches.assign(category=matches.id1.map(categories))
if PAIR_SAMPLE_SIZE is not None and len(pairs) > PAIR_SAMPLE_SIZE:
    fraction = PAIR_SAMPLE_SIZE / len(pairs)
    pairs = (pairs.groupby(['category','target'], group_keys=False)
             .sample(frac=fraction, random_state=RANDOM_SEED)
             .reset_index(drop=True))
print(f'pair sample={len(pairs):,}')
display(pairs.groupby(['category','target']).size().unstack(fill_value=0))
"""),
        markdown("""
## 5. Pair-features из JSON

Строим компактные признаки без wide-table: общие ключи, точные совпадения,
конфликты, Jaccard ключей, совпадение/конфликт по семействам и числам.
"""),
        code(r'''
attrs_by_id = dict(zip(items.id, attribute_dicts))

def family_values(attrs):
    result = defaultdict(set)
    for key, value in attrs.items():
        result[key_family(key)].add(value)
    return result

def pair_features(row):
    left, right = attrs_by_id[row.id1], attrs_by_id[row.id2]
    left_keys, right_keys = set(left), set(right)
    common = left_keys & right_keys
    matches_count = sum(left[k] == right[k] for k in common)
    conflicts = len(common) - matches_count
    union = left_keys | right_keys
    result = {
        'shared_keys': len(common),
        'key_jaccard': len(common)/max(1,len(union)),
        'matching_values': matches_count,
        'conflicting_values': conflicts,
        'value_match_ratio': matches_count/max(1,len(common)),
        'value_conflict_ratio': conflicts/max(1,len(common)),
    }
    lf, rf = family_values(left), family_values(right)
    for family in FAMILY_PATTERNS:
        lv, rv = lf.get(family,set()), rf.get(family,set())
        result[f'{family}_both'] = int(bool(lv and rv))
        result[f'{family}_match'] = int(bool(lv & rv))
        result[f'{family}_conflict'] = int(bool(lv and rv and not (lv & rv)))
    left_numbers = set(NUMBER_RE.findall(' '.join(left.values())))
    right_numbers = set(NUMBER_RE.findall(' '.join(right.values())))
    result['numeric_jaccard'] = len(left_numbers & right_numbers)/max(1,len(left_numbers | right_numbers))
    return result

started = time.perf_counter()
feature_rows = [pair_features(row) for row in pairs.itertuples(index=False)]
pair_feature_df = pd.concat([pairs.reset_index(drop=True), pd.DataFrame(feature_rows)], axis=1)
print(f'pair features in {time.perf_counter()-started:.1f}s')
pair_feature_df.to_parquet(REPORT_DIR/'pair_features_sample.parquet', index=False)
pair_feature_df.head()
'''),
        markdown("""
## 6. Одномерный сигнал и условные вероятности

AP используется только как ориентир ранжирующего сигнала. Бинарные conflict-
признаки могут быть полезны бустингу даже при невысоком одиночном AP.
"""),
        code("""
feature_columns = [c for c in pair_feature_df.columns if c not in {'id1','id2','target','category'}]
rows = []
for feature in feature_columns:
    values = pair_feature_df[feature].fillna(0)
    rows.append({
        'feature': feature,
        'macro_ap': pair_feature_df.assign(_score=values).groupby('category').apply(
            lambda part: average_precision_score(part.target, part._score), include_groups=False).mean(),
        'positive_mean': values[pair_feature_df.target == 1].mean(),
        'negative_mean': values[pair_feature_df.target == 0].mean(),
    })
feature_signal = pd.DataFrame(rows).sort_values('macro_ap', ascending=False)
feature_signal.to_csv(REPORT_DIR/'pair_feature_signal.csv', index=False)
display(feature_signal)
"""),
        code("""
condition_rows = []
for family in FAMILY_PATTERNS:
    for state in ['match','conflict']:
        column = f'{family}_{state}'
        selected = pair_feature_df[column].eq(1)
        condition_rows.append({
            'family': family, 'state': state, 'support': int(selected.sum()),
            'positive_rate': pair_feature_df.loc[selected,'target'].mean() if selected.any() else np.nan,
        })
family_conditions = pd.DataFrame(condition_rows)
family_conditions.to_csv(REPORT_DIR/'family_conditions.csv', index=False)
display(family_conditions)
"""),
        markdown("""
## 7. Какие точные ключи действительно полезны

Для каждого достаточно частого ключа измеряем, насколько часто он присутствует
у обеих карточек и что означает точное совпадение либо конфликт. Рейтинг считается
отдельно внутри категории — ключ `модель` в электронике и `размер` в одежде не
обязаны иметь одинаковый смысл и покрытие.
"""),
        code("""
candidate_keys = (key_stats[key_stats.support >= MIN_KEY_SUPPORT]
                  .sort_values(['category','support'], ascending=[True,False])
                  .groupby('category').head(TOP_KEYS_PER_CATEGORY)
                  .groupby('category').key.apply(list).to_dict())
exact_rows = []
for category, part in pairs.groupby('category'):
    for key in candidate_keys.get(category, []):
        both = equal = conflict = positives_equal = positives_conflict = 0
        for row in part.itertuples(index=False):
            left, right = attrs_by_id[row.id1], attrs_by_id[row.id2]
            if key not in left or key not in right:
                continue
            both += 1
            if left[key] == right[key]:
                equal += 1; positives_equal += int(row.target)
            else:
                conflict += 1; positives_conflict += int(row.target)
        exact_rows.append({
            'category': category, 'key': key, 'both_support': both,
            'equal_support': equal, 'conflict_support': conflict,
            'positive_rate_equal': positives_equal/equal if equal else np.nan,
            'positive_rate_conflict': positives_conflict/conflict if conflict else np.nan,
        })
exact_key_signal = pd.DataFrame(exact_rows)
exact_key_signal['rate_gap'] = exact_key_signal.positive_rate_equal - exact_key_signal.positive_rate_conflict
exact_key_signal.to_csv(REPORT_DIR/'exact_key_pair_signal.csv', index=False)
display(exact_key_signal.sort_values(['both_support','rate_gap'], ascending=False).head(80))
"""),
        markdown("""
## 8. Потенциальный шум и итоговые кандидаты

`seller_noise` не удаляется автоматически: таблица ниже покажет его покрытие и
pair-сигнал. Удалять поле стоит только если оно высококардинально, нестабильно и
не даёт полезного conditional signal.
"""),
        code("""
noise = key_stats[key_stats.family.eq('seller_noise')].sort_values('support', ascending=False)
display(noise.head(60))

recommended = exact_key_signal[
    (exact_key_signal.both_support >= 200) &
    (exact_key_signal.rate_gap.abs() >= 0.10)
].sort_values(['category','rate_gap'], ascending=[True,False])
recommended.to_csv(REPORT_DIR/'recommended_exact_keys.csv', index=False)
display(recommended.groupby('category').head(15))
"""),
        markdown("""
## Как использовать результаты в следующем эксперименте

Предлагаемый первый boosting-набор:

1. cosine similarity Qwen3-Embedding для `name`;
2. `abs(e1-e2)` после PCA до 32–64 компонент либо только агрегаты;
3. name lexical features из существующего EDA;
4. компактные JSON pair-features из секции 5;
5. match/conflict для рекомендованных точных ключей своей категории;
6. категория как categorical feature в CatBoost.

Отдельные item embeddings вычисляются один раз на уникальный товар и затем
join-ятся к парам. Сначала сравниваем `name embeddings + lexical`, затем добавляем
структурные attributes. Полный сырой JSON в первый embedding-прогон не добавляем:
иначе невозможно понять источник прироста и растёт стоимость инференса.
"""),
        code("""
summary = {
    'items': len(items), 'pairs_analyzed': len(pairs),
    'unique_attribute_keys': len(global_keys),
    'median_attributes_per_item': float(np.median(items_per_attribute)),
    'recommended_exact_keys': len(recommended),
    'pair_sample_size_setting': PAIR_SAMPLE_SIZE,
}
(REPORT_DIR/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
summary
"""),
    ]
    nbf.write(nb, OUTPUT)
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    main()
