# MiniLM 5ep: human train + graph closure

Controlled data ablation based on the locked
`minilm_5ep_team_ablation` template. The checkpoint, loss, optimizer,
one-epoch fine-tune recipe, serialization and IID/hard/OOD validation remain
unchanged. Only the editable data hook differs.

The hook computes positive components from human train and adds consequences of
the entity-equivalence relation:

- `A = B` and `B = C` imply `A = C`;
- `A = B` and `B != C` imply `A != C`.

For frozen `product-matching-validation-splits-v1` human train this produces:

| Source | Pairs |
| --- | ---: |
| Original human train | 306,669 |
| `graph_transitive_positive` | 1,739 |
| `graph_propagated_negative` | 4,024 |
| Total train | 312,432 |

The final positive rate is approximately `0.262057`. No frozen validation item
can enter train because all derivations use only `human_train_pairs`; the locked
template guard verifies this again before materialization.

This experiment increases optimizer steps by about `1.88%`, so it measures the
combined effect of the extra graph-derived data and that small compute increase.

Regenerate the notebook:

```bash
uv run python scripts/create_minilm_5ep_graph_closure_notebook.py
```

Dry-run:

```bash
uv run python scripts/run_kaggle_notebook.py \
  notebooks/minilm_5ep_graph_closure/minilm_5ep_graph_closure_2xt4.ipynb \
  --slug product-matching-minilm-5ep-graph-closure-v1 \
  --title "MiniLM 5ep: human plus graph closure v1" \
  --dataset alexproger23/product-matching-validation-splits-v1 \
  --dataset alexproger23/product-matching-minilm-llm-pretrain-5ep \
  --no-env-sources \
  --dry-run
```

Launch in the background by replacing `--dry-run` with `--no-wait`.
