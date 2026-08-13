#!/usr/bin/env python3
"""Generate two separate notebooks for split and error-pattern audits."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
SPLIT_NOTEBOOK = ROOT / "notebooks/03_validation_split_audit.ipynb"
ERROR_NOTEBOOK = ROOT / "notebooks/04_error_pattern_analysis.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def dependency_cell():
    return code("""
import importlib.util
import subprocess
import sys

required_packages = {
    'rapidfuzz': 'rapidfuzz>=3.12,<4',
    'catboost': 'catboost==1.2.8',
    'sklearn': 'scikit-learn>=1.5,<2',
    'pyarrow': 'pyarrow>=17,<22',
    'matplotlib': 'matplotlib>=3.9,<4',
    'seaborn': 'seaborn>=0.13,<0.14',
    'tabulate': 'tabulate>=0.9,<1',
}
missing = [package for module, package in required_packages.items()
           if importlib.util.find_spec(module) is None]
if missing:
    print('Installing missing notebook dependencies:', missing)
    subprocess.run([sys.executable, '-m', 'pip', 'install', *missing], check=True)
    print('Dependencies installed. Restart the kernel once, then run all cells again.')
else:
    print('All notebook dependencies are available.')
""")


def split_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    nb.cells = [
        md("""
# 03. Аудит validation split

Цель — проверить не размер validation, а переносимость схемы разбиения. Одна и
та же names-only lexical CatBoost обучается на нескольких splits. Hard-negative
оценка намеренно вынесена в следующий notebook и не смешивается с основной AP.
"""),
        dependency_cell(),
        code("""
from pathlib import Path
import json, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

ROOT = Path.cwd()
if not (ROOT / 'src').is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.validation_audit import (
    build_split_scenarios, evaluate_split, lexical_pair_table, prepare_items,
    representation_overlap, reports_to_frame, save_json,
)

DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'reports' / 'validation_split_audit'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_FRACTION = 0.15
SEEDS = (13, 42, 77, 2026)
CATBOOST_ITERATIONS = 1200
CATBOOST_DEPTH = 8
"""),
        md("## 1. Подготовка параллельных представлений"),
        code("""
matches = pd.read_parquet(DATA_DIR/'matches.parquet', columns=['id1','id2','target'])
started = time.perf_counter()
prepared_items_path = OUTPUT_DIR/'prepared_items.parquet'
pair_features_path = OUTPUT_DIR/'pair_audit_features.parquet'
if prepared_items_path.exists() and pair_features_path.exists():
    items = pd.read_parquet(prepared_items_path)
    pairs = pd.read_parquet(pair_features_path)
    print('Reusing prepared audit tables from', OUTPUT_DIR)
else:
    items_raw = pd.read_parquet(DATA_DIR/'items_human.parquet', columns=['id','name','attributes','category'])
    items = prepare_items(items_raw)
    pairs = lexical_pair_table(items, matches)
    items.drop(columns=['attributes_parsed']).to_parquet(prepared_items_path, index=False)
    pairs.to_parquet(pair_features_path, index=False)
print(f'Prepared {len(items):,} items and {len(pairs):,} pairs in {(time.perf_counter()-started)/60:.1f} min')
display(pairs.groupby('category').target.agg(['size','mean']).rename(columns={'mean':'positive_rate'}))
"""),
        md("## 2. Несколько seed и альтернативные grouped splits"),
        code("""
scenarios = build_split_scenarios(items, matches, VALIDATION_FRACTION, SEEDS)
pd.DataFrame([{
    'split': split.name, 'train_pairs': int(split.train_mask.sum()),
    'validation_pairs': int(split.valid_mask.sum()), 'notes': split.notes,
} for split in scenarios])
"""),
        md("## 3. Одинаковая lexical CatBoost на каждом split"),
        code("""
reports = []
overlaps = []
cached_overlap_path = OUTPUT_DIR/'representation_overlap.csv'
cached_overlaps = pd.read_csv(cached_overlap_path) if cached_overlap_path.exists() else pd.DataFrame()
for split in scenarios:
    report_path = OUTPUT_DIR/f'{split.name}_report.json'
    predictions_path = OUTPUT_DIR/f'{split.name}_predictions.parquet'
    model_path = OUTPUT_DIR/f'{split.name}.cbm'
    if report_path.exists() and predictions_path.exists() and model_path.exists():
        report = json.loads(report_path.read_text(encoding='utf-8'))
        reports.append(report)
        if not cached_overlaps.empty:
            overlaps.append(cached_overlaps[cached_overlaps.split.eq(split.name)].copy())
        print(f'Reusing completed {split.name}:', report['macro_average_precision'])
        continue
    print(f'Running {split.name} ...', flush=True)
    report, predictions, model = evaluate_split(
        pairs, split, iterations=CATBOOST_ITERATIONS, depth=CATBOOST_DEPTH,
    )
    reports.append(report)
    predictions.to_parquet(predictions_path, index=False)
    model.save_model(model_path)
    overlap = representation_overlap(items, pairs, split)
    overlap.insert(0, 'split', split.name)
    overlaps.append(overlap)
    save_json(report_path, report)
    print(split.name, report['macro_average_precision'])

comparison = reports_to_frame(reports).sort_values('macro_average_precision', ascending=False)
overlap_table = pd.concat(overlaps, ignore_index=True)
comparison.to_csv(OUTPUT_DIR/'split_comparison.csv', index=False)
overlap_table.to_csv(OUTPUT_DIR/'representation_overlap.csv', index=False)
display(comparison)
figure, axis = plt.subplots(figsize=(10, 4))
sns.barplot(data=comparison, x='macro_average_precision', y='split', ax=axis, color='#4C78A8')
axis.set(title='Lexical CatBoost: macro AP по validation-сценариям', xlabel='macro AP', ylabel='')
figure.tight_layout(); figure.savefig(OUTPUT_DIR/'split_macro_ap.png', dpi=160, bbox_inches='tight')
plt.show()
"""),
        md("## 4. Насколько содержательные представления пересекаются между train и validation"),
        code("""
overlap_pivot = overlap_table.pivot(index='split', columns='representation', values='validation_seen_share')
display(overlap_pivot)
figure, axis = plt.subplots(figsize=(10, 5))
sns.heatmap(overlap_pivot, annot=True, fmt='.2f', vmin=0, vmax=1, cmap='YlOrRd', ax=axis)
axis.set(title='Доля validation-значений, уже встречавшихся в train', xlabel='', ylabel='')
figure.tight_layout(); figure.savefig(OUTPUT_DIR/'representation_overlap.png', dpi=160, bbox_inches='tight')
plt.show()
"""),
        md("## 5. Автоматический черновик отчёта"),
        code("""
best_stress = comparison.sort_values('macro_average_precision').iloc[0]
iid = comparison[comparison.split.eq('component_seed_42')].iloc[0]
lines = [
    '# Результаты аудита validation split', '',
    f'- Текущий component seed 42: macro AP `{iid.macro_average_precision:.6f}`.',
    f'- Самый сложный проверенный split: `{best_stress.split}`, macro AP `{best_stress.macro_average_precision:.6f}`.',
    '', '## Полная таблица', '', comparison.to_markdown(index=False), '',
    'Это автоматически созданный отчёт. Интерпретацию и финальную рекомендацию нужно перенести в `docs/validation-split-audit.md` после просмотра overlaps и ошибок.'
]
(OUTPUT_DIR/'report.md').write_text('\\n'.join(lines), encoding='utf-8')
save_json(OUTPUT_DIR/'manifest.json', {'status':'complete','seeds':SEEDS,'reports':reports})
(OUTPUT_DIR/'COMPLETED').write_text('ok\\n', encoding='utf-8')
print(OUTPUT_DIR/'report.md')
"""),
    ]
    return nb


def error_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    nb.cells = [
        md("""
# 04. Hard negatives, n-граммы и конкретные ошибки

Notebook читает predictions из `03_validation_split_audit.ipynb`; модели здесь
не обучаются. FPR/FNR — описательные показатели при фиксированном пороге 0.5.
Word n-граммы строятся для 1–3 слов, character n-граммы — для 3–5 символов.
Семантические комбинации анализируются отдельно.
"""),
        dependency_cell(),
        code("""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

ROOT = Path.cwd()
if not (ROOT / 'src').is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.validation_audit import (
    confusion_columns, hard_negative_table, pattern_error_rates,
    semantic_combination_rates, save_json,
)

INPUT_DIR = ROOT/'reports'/'validation_split_audit'
OUTPUT_DIR = ROOT/'reports'/'error_pattern_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_NAME = 'component_seed_42'  # после audit можно заменить на выбранный development split
THRESHOLD = 0.5
MIN_WORD_SUPPORT = 100
MIN_CHAR_SUPPORT = 200
"""),
        md("## 1. Загружаем честные validation predictions"),
        code("""
predictions = pd.read_parquet(INPUT_DIR/f'{SPLIT_NAME}_predictions.parquet')
predictions = confusion_columns(predictions, THRESHOLD)
print(predictions.error_type.value_counts())
category_error_rows = []
for category, part in predictions.groupby('category'):
    negative = part.target.eq(0); positive = part.target.eq(1)
    category_error_rows.append({
        'category': category, 'pairs': len(part), 'positive_rate': part.target.mean(),
        'fpr': part.loc[negative, 'is_fp'].mean() if negative.any() else np.nan,
        'fnr': part.loc[positive, 'is_fn'].mean() if positive.any() else np.nan,
    })
category_errors = pd.DataFrame(category_error_rows)
category_errors.to_csv(OUTPUT_DIR/'category_error_rates.csv', index=False)
display(category_errors)
"""),
        md("## 2. Hard negatives"),
        code("""
hard = hard_negative_table(predictions)
hard.to_parquet(OUTPUT_DIR/'hard_negatives.parquet', index=False)
display(hard[['id1','id2','category','name_1','name_2','predict','hard_score','critical_conflict']].head(100))
"""),
        md("## 3. Общие словесные и символьные n-граммы с частыми ошибками"),
        code("""
word_patterns = pattern_error_rates(predictions, kind='word', min_support=MIN_WORD_SUPPORT)
char_patterns = pattern_error_rates(predictions, kind='char', min_support=MIN_CHAR_SUPPORT)
word_patterns.to_csv(OUTPUT_DIR/'word_ngram_error_rates.csv', index=False)
char_patterns.to_csv(OUTPUT_DIR/'char_ngram_error_rates.csv', index=False)
display(word_patterns.sort_values(['fpr','negative_support'], ascending=False).head(80))
display(word_patterns.sort_values(['fnr','positive_support'], ascending=False).head(80))
"""),
        md("## 4. Семантические комбинации: brand + color, model + number conflict и другие"),
        code("""
combinations = semantic_combination_rates(predictions, min_support=50)
combinations.to_csv(OUTPUT_DIR/'semantic_combination_error_rates.csv', index=False)
display(combinations.head(100))
plot_data = combinations.dropna(subset=['fpr']).sort_values(['fpr','support'], ascending=False).head(25)
if len(plot_data):
    figure, axis = plt.subplots(figsize=(10, 7))
    sns.barplot(data=plot_data, x='fpr', y='combination', ax=axis, color='#E45756')
    axis.set(title='Семантические комбинации с высоким FPR', xlabel='FPR при threshold=0.5', ylabel='')
    figure.tight_layout(); figure.savefig(OUTPUT_DIR/'semantic_combination_fpr.png', dpi=160, bbox_inches='tight')
    plt.show()
"""),
        md("## 5. Конкретные FP/FN для ручной проверки"),
        code("""
false_positives = predictions[predictions.is_fp].sort_values('predict', ascending=False)
false_negatives = predictions[predictions.is_fn].sort_values('predict', ascending=True)
columns = [
    'id1','id2','category','name_1','name_2','target','predict',
    'name_token_set_ratio','name_numeric_jaccard','number_conflict',
    'measure_match','measure_conflict','brand_match','brand_conflict',
    'model_match','model_conflict','color_match','color_conflict',
]
false_positives[columns].head(1000).to_csv(OUTPUT_DIR/'top_false_positives.csv', index=False)
false_negatives[columns].head(1000).to_csv(OUTPUT_DIR/'top_false_negatives.csv', index=False)
display(false_positives[columns].head(50))
display(false_negatives[columns].head(50))
"""),
        md("## 6. Черновик отчёта"),
        code("""
top_combo = combinations.iloc[0] if len(combinations) else None
lines = ['# Результаты анализа ошибок', '', f'- Split: `{SPLIT_NAME}`.', f'- Порог FPR/FNR: `{THRESHOLD}`.',
         f'- FP: `{int(predictions.is_fp.sum())}`, FN: `{int(predictions.is_fn.sum())}`.',
         f'- Hard negatives сохранены: `{len(hard)}` строк.', '']
if top_combo is not None:
    lines.append(f"- Наибольший наблюдаемый FPR среди поддержанных комбинаций: `{top_combo.combination}` = `{top_combo.fpr:.3f}` при support `{int(top_combo.support)}`.")
lines += ['', 'Полные таблицы: `word_ngram_error_rates.csv`, `char_ngram_error_rates.csv`, `semantic_combination_error_rates.csv`, `top_false_positives.csv`, `top_false_negatives.csv`.']
(OUTPUT_DIR/'report.md').write_text('\\n'.join(lines), encoding='utf-8')
save_json(OUTPUT_DIR/'manifest.json', {'status':'complete','split':SPLIT_NAME,'threshold':THRESHOLD})
(OUTPUT_DIR/'COMPLETED').write_text('ok\\n', encoding='utf-8')
print(OUTPUT_DIR/'report.md')
"""),
    ]
    return nb


def main() -> None:
    SPLIT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(split_notebook(), SPLIT_NOTEBOOK)
    nbf.write(error_notebook(), ERROR_NOTEBOOK)
    print(f"Created {SPLIT_NOTEBOOK}")
    print(f"Created {ERROR_NOTEBOOK}")


if __name__ == "__main__":
    main()
