# Hydra Plugins

Hydra allows the use of external plugins. This directory contains useful plugings for projects using vital.

[Hydra plugin development documentation](https://hydra.cc/docs/advanced/plugins/develop)

# SearchPath

## Vital
This file allows other projects to access vital configs without specifying the search path in each primary config.

This replaces these lines in the primary configs:
 ```yaml
hydra:
  searchpath:
    - pkg://vital.config
```
[Hydra SearchPath documentation](https://hydra.cc/docs/advanced/search_path/)

[Hydra SearchPath example](https://github.com/facebookresearch/hydra/tree/main/examples/plugins/example_searchpath_plugin)
