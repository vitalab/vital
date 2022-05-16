# Hydra Plugins

Hydra allows the use of [external plugins](https://hydra.cc/docs/advanced/plugins/develop). This directory contains useful plugins for projects using `vital`.

## [SearchPath](https://hydra.cc/docs/advanced/search_path)

The [`vital` module](searchpath/vital.py) for `searchpath` allows other projects to access `vital` configs without specifying the search path in each primary config.

It replaces the following lines in primary configs:
 ```yaml
hydra:
  searchpath:
    - pkg://vital.config
```

### Additional resources
- [Hydra SearchPath example](https://github.com/facebookresearch/hydra/tree/main/examples/plugins/example_searchpath_plugin)
