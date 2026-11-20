# Third-party notices

The published package contains no third-party code and no icons. The model weights are trained on the data below. The word list in the weight file contains words that occur in icon names of Iconify icon sets.

## Training data

Training reads icon names from Iconify, and search keywords that icon set authors publish for their icons.

| Source | Data that training reads | Version | License |
| --- | --- | --- | --- |
| [Iconify icon sets](https://github.com/iconify/icon-sets) | Icon names, aliases | `@iconify/json` 2.2.559 | MIT for the package. Each icon set has its own license. |
| [Lucide](https://github.com/lucide-icons/lucide) | Tags | `lucide-static` 1.76.0 | ISC |
| [Tabler Icons](https://github.com/tabler/tabler-icons) | Tags | `@tabler/icons` 3.52.0 | MIT |
| [Material Design Icons](https://github.com/Templarian/MaterialDesign) | Aliases | `@mdi/svg` 7.4.47 | Apache-2.0 |
| [Font Awesome Free](https://github.com/FortAwesome/Font-Awesome) | Search terms | `@fortawesome/fontawesome-free` 7.3.1 | CC-BY-4.0, OFL-1.1 and MIT |
| [Octicons](https://github.com/primer/octicons) | Keywords | `@primer/octicons` 19.42.0 | MIT |
| [Ionicons](https://github.com/ionic-team/ionicons) | Tags | `ionicons` 8.1.0 | MIT |
| [Carbon](https://github.com/carbon-design-system/carbon) | Aliases | `@carbon/icons` 11.92.0 | Apache-2.0 |
| [Remix Icon](https://github.com/Remix-Design/RemixIcon) | Tags | commit `9fb7967` | Remix Icon License v1.0 |
| [Feather](https://github.com/feathericons/feather) | Tags | commit `3dc050d` | MIT |
| [Material Symbols](https://fonts.google.com/icons) | Tags | Google Fonts metadata, hash in `training/data/sources.json` | Apache-2.0 |
| [Fluent UI System Icons](https://github.com/microsoft/fluentui-system-icons) | Metaphors | commit `6c2c907` | MIT |

Font Awesome Free requires attribution: the search terms come from Font Awesome Free 7.3.1 by Fonticons, Inc. (https://fontawesome.com).

## Teacher model

Training copies word similarity from [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), in the ONNX conversion [Xenova/bge-small-en-v1.5](https://huggingface.co/Xenova/bge-small-en-v1.5). Its license is MIT.

## Test data only

Training never reads these sources. The benchmark uses them.

| Source | Use | License |
| --- | --- | --- |
| [Icones](https://github.com/antfu-collective/icones) | The manual alias list in `training/data/icones-alias.json`, and the search logic that `training/src/search.ts` replicates | MIT |
| [Phosphor Icons](https://github.com/phosphor-icons/core) | Tags, as queries of the `transfer` test set. `@phosphor-icons/core` 2.1.1 | MIT |
| [Bootstrap Icons](https://github.com/twbs/icons) | Tags, as queries of the `transfer` test set. Commit `829d2bc` | MIT |
