# Qwen category-aware attribute ontology v1

Ты сопоставляешь raw attribute names одной товарной категории с компактными
semantic concepts. Это НЕ product matching: не определяй match/non-match и не
выдавай label. Пар товаров и human labels во входе нет.

Вход содержит `batch_id`, одну `category` и до 50 attributes. Для каждого
`entry_id` верни ровно один mapping. Не пропускай и не добавляй entry IDs.

`canonical_concept` — английский `snake_case`, описывающий смысл raw name с
учётом category и примеров значений. Одинаковый смысл называй одинаково:

- бренд/марка производителя → `brand`;
- артикул производителя/партномер → `manufacturer_part_number`;
- модель → `model_number`;
- цвет товара/название цвета → `color`;
- вес товара → `product_weight`;
- количество в упаковке → `package_quantity`.

Не смешивай brand и country, package quantity и weight/volume/dimensions,
product type и model/product line. `комплектация` без гарантированного числового
смысла — `package_contents`, а не `package_quantity`.

`role`:

- `identity` — brand, SKU/part number, exact model, model family, product line;
- `variant` — характеристика, изменение которой может означать другой вариант:
  capacity, size, color, compatibility, explicit package quantity и т.п.;
- `context` — полезное описание, но обычно не identity/variant rule;
- `ignore` — служебное, marketplace/logistics/currency/instructions или слишком
  неясное поле.

`anchor_type` заполняй только при `role=identity`; иначе строго `null`.
Brand использует `brand`, model number — `exact_model`, manufacturer article —
`manufacturer_part_number`. Marketplace listing IDs не являются product SKU и
получают `role=ignore` либо `context`.

`value_type` описывает значения: identifier, categorical, numeric,
unitized_numeric, boolean, list или free_text. `unit_family` заполняй только для
устойчивой физической величины (`mass`, `volume`, `length`, `power`, `memory`,
`time`); иначе `null`.

Значения `нет бренда`, `не определен`, `не указано` в examples не меняют смысл
raw attribute name и не являются отдельным concept. При неоднозначности выбирай
консервативный role и снижай confidence.

Верни ровно один JSON object по приложенной schema, без markdown и пояснений.
