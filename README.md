# VITAL

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Welcome to the git repo of the [Videos & Images Theory and Analytics Laboratory (VITAL)](http://vital.dinf.usherbrooke.ca/ "VITAL home page")
of Sherbrooke University, headed by Professor [Pierre-Marc Jodoin](http://info.usherbrooke.ca/pmjodoin/).

## How to use
This repository was not designed to be used as a standalone project, but was rather meant to be included inside other
repositories. The recommended way of including this repository is as a [git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
inside the project from which you want to use it.

## Description
To help you follow along with the organization of the repository, here is a summary of each major package's purpose:

- [data](data): utilities to process and interface with common medical image datasets, from processing raw image files
(e.g. `.mhd` or `nii.gz`) to implementations of torchvision's `VisionDataset`.

- [logs](logs): generic utilities for logging results during the evaluation phase.

- [metrics](metrics): common metrics that are not part of the traditional libraries, whether those metrics are losses for
training (see [train](metrics/train)) or scores to evaluate the systems' performance (see [evaluate](metrics/evaluate)).

- [modules](modules): generic models, organized by task (e.g. [classification](modules/segmentation),
[generative](modules/generative), etc.).

- [systems](systems): common boilerplate Lightning module code (split across mixins), from which concrete projects'
systems should inherit.

- [utils](utils): a wide range of common utilities that may be used in multiple other packages
(e.g. [image processing](utils/image), [parameter groups](utils/parameters.py), etc.).

- [VitalTrainer](vital_trainer.py): common boilerplate Lightning trainer code for handling generic trainer
configuration, as well as multiple systems and their configuration.

## Requirements
The [vital.yml](vital.yml) file lists the dependencies required by the whole repository. In case you include the
repository inside your own project, you may want to add project specific dependencies, or maybe even remove some
dependencies (if you only use some of the utilities provided by the repository).


## How to contribute

### Version Control Hooks
The [`.pre-commit-config.yaml`](.pre-commit-config.yaml) file defines the pre-commit hooks that should be installed in
any project contributing to the `vital` repository. For consistency's sake, it is recommended to use the same
configuration for the pre-commit hooks for both the project repository and the `vital` submodule. This can be achieved
by using symlinks to the `vital` configuration files from the project repository.

NOTE: `isort` must be configured slightly differently in the project than in the `vital` submodule, in order to list the
project's top-level packages as known first party.

Assuming you start at the root of your project directory, linking the configuration files in the project repository to
the `vital` configuration and installing hooks should look something like this:
```
# navigate to the vital submodule and
# install the pre-commit hooks for the vital submodule
cd vital
pre-commit install
cd ..

# symlink configuration files for the project repository to
# the configuration files in the vital submodule
ln -s ./vital/.pre-commit-config.yaml .pre-commit-config.yaml
ln -s ./vital/pyproject.toml pyproject.toml
ln -s ./vital/setup.cfg setup.cfg

# create an isort configuration for the project from the vital configuration
cp ./vital/.isort.cfg isort.cfg
# Afterwards, edit the copied configuration file manually to add all top-level
# project packages in the `known_first_party` tag

# install the pre-commit hooks for the project repository
pre-commit install
```
