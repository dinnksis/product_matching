# MiniLM S2 hard audit

Полная интерпретация: `docs/minilm-s2-hard-audit.md`.

Кратко: hard validation специально отобрана по lexical surprises и ошибкам
предыдущих моделей. S2 даёт macro AP `0.307319`; CatBoost — `0.314517`, но
ухудшает IID и не улучшает numeric/code/SKU-срезы. Default остаётся S2 без
CatBoost. Перед следующим обучением требуется ручная проверка противоречивых
human labels и фиксация SKU-level против family-level matching policy.
