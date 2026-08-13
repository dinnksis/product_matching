# Names-only lexical CatBoost control submission

This submission reproduces experiment `01_names_lexical` (local component-
disjoint validation macro AP `0.501538`). It uses eight symmetric name features,
one-hot product category and the trained CatBoost model.

It deliberately contains no Qwen inference, embeddings, attributes, fallback
or deadline-dependent branch. It is a control submission for measuring how well
the local human-labelled validation transfers to the hidden competition test.
