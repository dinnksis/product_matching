"""Create the reproducible EDA notebook committed with the project."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "01_human_data_eda.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


cells = [
    markdown(
        """
        # E-CUP 2026: базовый анализ ручной разметки товарных пар

        Цель отчёта — понять качество и структуру доступной ручной разметки,
        оценить силу дешёвых эвристик и сформировать план экспериментов для
        cross-encoder, лёгких моделей и ансамбля.

        Анализ использует только `data/items_human.parquet` и
        `data/matches.parquet`. Набор с 11 млн LLM-размеченных пар пока не
        скачан и в расчёты не входит. Все значения ниже получены из локальных
        parquet-файлов, а не из описания соревнования.
        """
    ),
    code(
        """
        from pathlib import Path
        import json
        import sys

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from IPython.display import Markdown, display
        from sklearn.metrics import average_precision_score

        ROOT = next(
            candidate
            for candidate in [Path.cwd(), *Path.cwd().parents]
            if (candidate / "pyproject.toml").exists()
        )
        sys.path.insert(0, str(ROOT / "src"))

        from product_matching.eda import (
            attribute_key_summary,
            build_pair_features,
            cross_validated_light_baseline,
            graph_summary,
            hard_examples,
            load_human_data,
            macro_average_precision,
            pair_category_summary,
            univariate_feature_scores,
            validate_human_data,
        )

        DATA_DIR = ROOT / "data"
        REPORT_DIR = ROOT / "reports"
        REPORT_DIR.mkdir(exist_ok=True)

        pd.set_option("display.max_columns", 30)
        pd.set_option("display.max_colwidth", 100)
        sns.set_theme(style="whitegrid", context="notebook")
        COLORS = {"negative": "#64748b", "positive": "#f97316", "model": "#0f766e"}
        """
    ),
    markdown("## 1. Загрузка и целостность данных"),
    code(
        """
        items, matches = load_human_data(DATA_DIR)
        integrity = validate_human_data(items, matches)
        display(integrity.to_frame())
        display(items.head(3))
        display(matches.head(3))

        display(Markdown(
            f'''
            **Фактический объём:** {len(items):,} карточек и {len(matches):,} пар.
            Все ID уникальны, ссылки из пар разрешаются, пропусков, self-pairs и
            повторяющихся неупорядоченных пар нет. Метка принимает значения
            {integrity['target_values']}.
            '''.replace(",", " ")
        ))
        """
    ),
    markdown("## 2. Категории и дисбаланс классов"),
    code(
        """
        category_summary, pair_categories, cross_category_pairs = pair_category_summary(items, matches)
        display(category_summary.style.format({"positive_rate": "{:.1%}"}))

        plot_data = category_summary.sort_values("positive_rate")
        fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
        axes[0].barh(plot_data.index, plot_data["pairs"], color="#2563eb")
        axes[0].set(title="Число размеченных пар", xlabel="Пар", ylabel="Категория")
        axes[0].tick_params(axis="y", labelsize=9)
        axes[1].barh(plot_data.index, plot_data["positive_rate"], color=COLORS["positive"])
        axes[1].axvline(plot_data["positive_rate"].mean(), color="#0f172a", ls="--", lw=1,
                        label="Среднее по категориям")
        axes[1].set(title="Доля дублей", xlabel="Положительная доля", ylabel="")
        axes[1].xaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
        axes[1].legend(loc="lower right")
        fig.tight_layout()
        plt.show()

        prevalence_macro = category_summary["positive_rate"].mean()
        display(Markdown(
            f'''
            Все пары находятся **внутри одной категории** (межкатегорийных пар:
            {cross_category_pairs}). При этом доля положительного класса меняется
            от **{category_summary['positive_rate'].min():.1%}** до
            **{category_summary['positive_rate'].max():.1%}**. Это делает обычный
            micro/overall AP недостаточным: отбор моделей, early stopping и подбор
            ансамбля нужно вести по macro AP из 20 category-wise AP. AP константного
            ранжирования здесь равен средней prevalence: **{prevalence_macro:.4f}**.
            '''
        ))
        """
    ),
    markdown("## 3. Поля карточек и JSON-атрибуты"),
    code(
        """
        length_summary = pd.DataFrame({
            "name_chars": items["name"].str.len().describe(percentiles=[.01, .1, .5, .9, .95, .99]),
            "attributes_chars": items["attributes"].str.len().describe(percentiles=[.01, .1, .5, .9, .95, .99]),
        })
        display(length_summary.round(1))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for axis, column, title, color in [
            (axes[0], "name", "Длина названия", "#2563eb"),
            (axes[1], "attributes", "Длина JSON-атрибутов", COLORS["positive"]),
        ]:
            values = items[column].str.len().clip(upper=items[column].str.len().quantile(.995))
            axis.hist(values, bins=50, color=color, alpha=.85)
            axis.set(title=f"{title} (до p99.5)", xlabel="Символов", ylabel="Карточек")
        fig.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        top_attribute_keys, attribute_coverage, attribute_quality = attribute_key_summary(items, top_n=20)
        display(attribute_quality.to_frame())

        fig, axes = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [1, 1.7]})
        top_plot = top_attribute_keys.sort_values("coverage")
        axes[0].barh(top_plot.index, top_plot["coverage"], color=COLORS["model"])
        axes[0].xaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
        axes[0].set(title="Покрытие частых ключей", xlabel="Доля карточек", ylabel="Ключ")

        sns.heatmap(
            attribute_coverage[top_attribute_keys.index[:15]],
            cmap="YlOrBr", vmin=0, vmax=1, ax=axes[1],
            cbar_kws={"label": "Доля карточек категории"},
        )
        axes[1].set(title="Покрытие ключей по категориям", xlabel="Ключ", ylabel="Категория")
        axes[1].tick_params(axis="x", rotation=55, labelsize=8)
        axes[1].tick_params(axis="y", labelsize=8)
        fig.tight_layout()
        plt.show()

        display(Markdown(
            f'''
            Все `attributes` валидны как JSON-объекты; пустых объектов —
            **{int(attribute_quality['empty_attribute_objects']):,}**. Медиана —
            **{attribute_quality['median_keys_per_item']:.0f}** ключей на карточку,
            однако словари заметно различаются между категориями. Поэтому полезнее
            сериализовать атрибуты как пары `ключ: значение` и учить модель понимать
            ключ, чем склеивать только значения. Для лёгкой ветки нужны
            category-specific признаки и нормализация синонимов ключей.
            '''.replace(",", " ")
        ))
        """
    ),
    markdown(
        """
        ## 4. Структура графа пар

        Ребро — размеченная пара, вершина — товар. Компоненты связности используются
        как группы в кросс-валидации: ни один товар и его соседние пары не переходят
        между train и validation.
        """
    ),
    code(
        """
        pair_data = build_pair_features(items, matches)
        graph_stats, component_groups = graph_summary(len(items), pair_data)
        display(graph_stats.to_frame())

        display(Markdown(
            f'''
            **{graph_stats['degree_one_share']:.1%}** товаров встречаются ровно в
            одной паре; максимальная степень — {int(graph_stats['maximum_item_degree'])}.
            Компоненты по всем рёбрам имеют медианный размер
            {graph_stats['median_all_edge_component_size']:.0f} и максимум
            {int(graph_stats['maximum_all_edge_component_size'])}. Внутри компонент,
            собранных только по положительным рёбрам, не найдено отрицательных рёбер
            ({int(graph_stats['negative_edges_inside_positive_components'])}). Это
            хороший sanity check транзитивности ручной разметки, хотя маленькие
            компоненты не позволяют считать его полной проверкой.
            '''
        ))
        """
    ),
    markdown("## 5. Насколько сильны дешёвые признаки"),
    code(
        """
        conditional_rows = []
        for column in ["name_exact", "identifier_overlap", "brand_equal", "brand_conflict"]:
            mask = pair_data.features[column].astype(bool).to_numpy()
            conditional_rows.append({
                "signal": column,
                "pairs_with_signal": int(mask.sum()),
                "positive_rate_with_signal": pair_data.targets[mask].mean() if mask.any() else np.nan,
                "positive_rate_without_signal": pair_data.targets[~mask].mean(),
            })
        conditional = pd.DataFrame(conditional_rows).set_index("signal")
        display(conditional.style.format({
            "positive_rate_with_signal": "{:.1%}",
            "positive_rate_without_signal": "{:.1%}",
        }))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for axis, column, title in [
            (axes[0], "name_token_set_ratio", "Token-set similarity названий"),
            (axes[1], "numeric_jaccard", "Jaccard числовых токенов"),
        ]:
            for target, label, color in [(0, "не дубль", COLORS["negative"]), (1, "дубль", COLORS["positive"])]:
                values = pair_data.features.loc[pair_data.targets == target, column]
                axis.hist(values, bins=30, density=True, histtype="step", lw=2, label=label, color=color)
            axis.set(title=title, xlabel="Similarity", ylabel="Плотность")
            axis.legend()
        fig.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        feature_scores = univariate_feature_scores(pair_data)
        display(feature_scores.style.format("{:.4f}"))

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_scores = feature_scores.sort_values("macro_ap")
        ax.barh(plot_scores.index, plot_scores["macro_ap"], color="#2563eb")
        ax.axvline(prevalence_macro, color=COLORS["positive"], ls="--", lw=2, label="prevalence baseline")
        ax.set(title="Macro AP отдельных быстрых признаков", xlabel="Macro average precision", ylabel="Признак")
        ax.legend()
        fig.tight_layout()
        plt.show()

        best_feature = feature_scores.index[0]
        display(Markdown(
            f'''
            Лучший одиночный сигнал — `{best_feature}` с macro AP
            **{feature_scores.iloc[0]['macro_ap']:.4f}**. Точное совпадение названий
            встречается в {int(conditional.loc['name_exact', 'pairs_with_signal']):,}
            парах, но лишь
            {conditional.loc['name_exact', 'positive_rate_with_signal']:.1%} из них
            положительны. Значит, правила вида «одинаковое имя ⇒ дубль» опасны:
            размеры, комплектации и иные варианты товара остаются критичными.
            '''.replace(",", " ")
        ))
        """
    ),
    markdown(
        """
        ## 6. Честный OOF-бейзлайн лёгкой модели

        `HistGradientBoostingClassifier` получает только дешёвые pair-features и
        код категории. Пять фолдов разделены по компонентам графа, то есть товары
        не пересекаются между train и validation. Это не претендент на финальное
        решение, а воспроизводимая нижняя граница и кандидат в ансамбль/gating.
        """
    ),
    code(
        """
        oof_scores, fold_scores = cross_validated_light_baseline(pair_data, component_groups)
        light_macro_ap, light_category_ap = macro_average_precision(
            pair_data.targets, oof_scores, pair_data.categories
        )
        overall_oof_ap = average_precision_score(pair_data.targets, oof_scores)
        display(fold_scores.style.format({"overall_ap": "{:.4f}"}))

        comparison = category_summary[["pairs", "positive_rate"]].join(
            light_category_ap.rename("light_oof_ap")
        ).sort_values("light_oof_ap")
        display(comparison.style.format({"positive_rate": "{:.4f}", "light_oof_ap": "{:.4f}"}))

        fig, ax = plt.subplots(figsize=(11, 8))
        positions = np.arange(len(comparison))
        ax.barh(positions - .18, comparison["positive_rate"], height=.36,
                color=COLORS["negative"], label="prevalence")
        ax.barh(positions + .18, comparison["light_oof_ap"], height=.36,
                color=COLORS["model"], label="лёгкая модель, OOF")
        ax.set_yticks(positions, comparison.index, fontsize=9)
        ax.set(title="Качество лёгкой модели по категориям", xlabel="Average precision", ylabel="Категория")
        ax.legend()
        fig.tight_layout()
        plt.show()

        display(Markdown(
            f'''
            Итог лёгкого OOF-бейзлайна: **macro AP = {light_macro_ap:.4f}**,
            overall AP = {overall_oof_ap:.4f}. Разрыв между ними подтверждает, что
            overall-метрика завышает впечатление от модели. Слабее всего работают
            категории **{comparison.index[0]}**, **{comparison.index[1]}** и
            **{comparison.index[2]}** — там особенно полезны cross-encoder,
            category-specific правила и работа с вариантами товара.
            '''
        ))
        """
    ),
    markdown("## 7. Уверенные ошибки лёгкой модели"),
    code(
        """
        hard_negatives, hard_positives = hard_examples(
            items, matches, pair_data, oof_scores, count=8
        )
        display(Markdown("### Отрицательные пары с высоким score"))
        display(hard_negatives)
        display(Markdown("### Положительные пары с низким score"))
        display(hard_positives)
        """
    ),
    markdown(
        """
        Уверенные ошибки показывают два режима. У отрицательных пар часто почти
        идентичны названия: различие может находиться в атрибутах/варианте либо быть
        спорным случаем разметки. У положительных пар встречаются заметно разные
        формулировки и, иногда, конфликтующие числа. Именно эти срезы стоит отдавать
        на ручной аудит и использовать для hard-negative mining. Нельзя автоматически
        считать их ошибками людей без просмотра полного контекста.
        """
    ),
    markdown(
        """
        ## 8. Предлагаемая архитектура решения

        ### Cross-encoder

        - Начать с предложенного `Qwen3-Reranker-0.6B` как основного нелинейного
          scorer. Формат пары: категория, название, затем стабильная сериализация
          `ключ: значение`; наиболее полезные идентификаторы и размеры ставить раньше.
        - Во время обучения случайно менять товары местами: score должен быть
          симметричен. Сэмплировать категории равномернее и выбирать checkpoint по
          macro AP, а не по loss/overall AP.
        - Сначала ограничить длину 256–384 токенами и замерить curve
          `качество — pairs/sec`; длинные хвосты attributes нельзя молча обрезать до
          того, как артикулы, бренд, модель, размер и комплектация вынесены в начало.
        - Делать hard-negative mining из уверенных ошибок лёгкой модели и текущего
          cross-encoder. Отдельно проверить пары с одинаковым названием и конфликтом
          числовых/variant-атрибутов.

        ### Лёгкая ветка

        - Расширить текущие признаки: word/character TF-IDF cosine, BM25-подобные
          overlaps, нормализованные единицы измерения, бренд, артикул/OEM/EAN,
          модель, цвет, размер и комплектация. Ключи атрибутов нормализовать через
          словарь синонимов.
        - Обучить CatBoost/LightGBM или компактную логистическую модель с
          category-specific взаимодействиями. Она полезна как быстрый baseline,
          второй независимый взгляд ансамбля и gate для дорогого scorer.
        - Осторожно использовать high-precision правила. Точное имя и даже бренд
          сами по себе недостаточны; правила должны учитывать конфликты варианта.

        ### Ансамбль и каскад

        - Собирать только OOF-предсказания базовых моделей и учить небольшой stacker
          на чистой ручной разметке. В качестве входов: cross-encoder logit, лёгкий
          score, identifier/variant flags и категория.
        - Сравнить один глобальный stacker с category-wise калибровкой. Оптимизировать
          macro AP; бинарный threshold для метрики не нужен — в submit следует писать
          непрерывный score.
        - Для скорости пропускать через cross-encoder только «серую зону» лёгкой
          модели. Крайним случаям назначать согласованные непрерывные scores, чтобы
          не разрушить ранжирование. Любой cascade валидировать одновременно по
          macro AP и wall-clock на объёмах Check/Public/Private.

        ### Когда появятся 11 млн LLM-размеченных пар

        - Проверить совпадения и конфликты с ручной выборкой до объединения.
        - Использовать LLM-разметку для pretraining/distillation с меньшим весом или
          confidence weights, затем fine-tune только на human labels.
        - Чистый component-disjoint human holdout не использовать ни для отбора
          псевдометок, ни для настройки правил. Сравнить обучение на human-only,
          LLM→human и совместную смесь.
        - Аудитировать disagreement: cross-encoder, лёгкая модель и LLM-метка. Это
          даст приоритетный список для повторной открытой LLM-разметки/ручной проверки.

        ### Скорость и воспроизводимость

        - Один раз нормализовать и токенизировать уникальные товары, затем собирать
          пары по ID. Батчи группировать по длине, использовать dynamic padding и
          BF16/квантизацию только после замера потери macro AP.
        - Хранить версии данных, seed, split-компоненты, конфиг сериализации и OOF
          predictions. Тренировочный и контейнерный inference-код должны использовать
          один и тот же preprocessor.
        - Перед отправкой запускать локальный контрактный тест: все входные пары
          сохранены ровно один раз, столбцы строго `id1,id2,predict`, score конечный,
          контейнер работает без сети.
        """
    ),
    markdown("## 9. Приоритет экспериментов"),
    code(
        """
        experiment_plan = pd.DataFrame([
            ["E0", "Текущие RapidFuzz/JSON признаки + HGB", "Воспроизводимая нижняя граница", "macro AP, CPU pairs/s"],
            ["E1", "Char+word TF-IDF + числовые/ID/variant признаки", "Сильный дешёвый scorer", "macro AP, latency, RAM"],
            ["E2", "Qwen3-Reranker-0.6B, human-only, 256 токенов", "Основной cross-encoder baseline", "macro AP, GPU pairs/s"],
            ["E3", "Сериализация/длина 128–512 + category-balanced sampler", "Найти качество/скорость", "macro AP × category, pairs/s"],
            ["E4", "LLM pretrain → human fine-tune", "Использовать 11 млн шумных меток", "Δ macro AP к E2"],
            ["E5", "OOF stacker: cross-encoder + TF-IDF + эвристики", "Стабильный ансамбль", "OOF macro AP"],
            ["E6", "Gate серой зоны + BF16/INT8", "Уложиться в 6/13 минут", "macro AP, wall-clock, archive size"],
        ], columns=["ID", "Эксперимент", "Цель", "Главные измерения"]).set_index("ID")
        display(experiment_plan)

        category_summary.to_csv(REPORT_DIR / "category_summary.csv")
        feature_scores.to_csv(REPORT_DIR / "univariate_feature_scores.csv")
        comparison.to_csv(REPORT_DIR / "light_baseline_by_category.csv")
        with (REPORT_DIR / "eda_summary.json").open("w", encoding="utf-8") as file:
            json.dump({
                "items": len(items),
                "pairs": len(matches),
                "positives": int(pair_data.targets.sum()),
                "positive_rate": float(pair_data.targets.mean()),
                "macro_prevalence": float(prevalence_macro),
                "light_oof_macro_ap": float(light_macro_ap),
                "light_oof_overall_ap": float(overall_oof_ap),
            }, file, ensure_ascii=False, indent=2)
        """
    ),
    markdown(
        """
        ## Итог

        Данные технически чистые, но задача неоднородна по категориям и содержит
        много сложных variant-level различий. Быстрые признаки уже дают осмысленный
        baseline, однако их разрыв по категориям и уверенные ошибки показывают, что
        основной прирост должен прийти от хорошо сериализованных атрибутов,
        cross-encoder и OOF-ансамбля. Следующий практический шаг — E1 и E2 на одном
        component-disjoint split с обязательным профилированием inference.
        """
    ),
]


notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.update(
    {
        "kernelspec": {
            "display_name": "Python 3 (product-matching)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
)

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, NOTEBOOK_PATH)
print(f"Wrote {NOTEBOOK_PATH.relative_to(ROOT)}")
