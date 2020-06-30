# VITAL

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Welcome to the git repo of the
[Videos & Images Theory and Analytics Laboratory (VITAL)](http://vital.dinf.usherbrooke.ca/ "VITAL home page") of
Sherbrooke University, headed by Professor [Pierre-Marc Jodoin](http://info.usherbrooke.ca/pmjodoin/).

## How to use
This repository was not designed to be used as a standalone project, but was rather meant to be used as a third-party
library for more applied projects.

## Description
To help you follow along with the organization of the repository, here is a summary of each major package's purpose:

- [data](vital/data): utilities to process and interface with common medical image datasets, from processing raw image
files (e.g. `.mhd` or `nii.gz`) to implementations of torchvision's `VisionDataset`.

- [logs](vital/logs): generic utilities for logging results during the evaluation phase.

- [metrics](vital/metrics): common metrics that are not part of the traditional libraries, whether those metrics are
losses for training (see [train](vital/metrics/train)) or scores to evaluate the systems' performance (see
[evaluate](vital/metrics/evaluate)).

- [modules](vital/modules): generic models, organized by task (e.g. [classification](vital/modules/segmentation),
[generative](vital/modules/generative), etc.).

- [systems](vital/systems): common boilerplate Lightning module code (split across mixins), from which concrete
projects' systems should inherit.

- [utils](vital/utils): a wide range of common utilities that may be used in multiple other packages (e.g.
[image processing](vital/utils/image), [parameter groups](vital/utils/parameters.py), etc.).

- [VitalTrainer](vital/vital_trainer.py): common boilerplate Lightning trainer code for handling generic trainer
configuration, as well as multiple systems and their configuration.

## Requirements
The [vital.yml](requirements/vital.yml) file lists the dependencies required by the whole repository. In case you
include the repository inside your own project, you may want to add project specific dependencies, or maybe even remove
some dependencies (if you only use some of the utilities provided by the repository).


## How to contribute

### Version Control Hooks
The [`.pre-commit-config.yaml`](.pre-commit-config.yaml) file defines the pre-commit hooks that should be installed in
any project contributing to the `vital` repository. For consistency's sake, it is recommended to use the same
configuration for the pre-commit hooks for both the dependent project and the `vital` project. This

#### Notice
- `isort` must be configured slightly differently in the dependent project than in the `vital` repository, in order to
indicate the project as the known first party.

#### Version Control Hooks Setup: Git Submodule
If the `vital` repository is installed as a [git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules),
copying the pre-commit configurations can be achieved by using symlinks to the `vital` configuration files from the
dependent project's directory.

Linking the configuration files in the project repository to the `vital` configuration and installing hooks would then
look something like this:
```
# navigate to the vital repository location and install the pre-commit hooks
cd <vital_repository_dir>   # Likely ./vital
pre-commit install

# symlink configuration files for the project repository to
# the configuration files in the vital repository
cd <project_root_dir>   # Likely ..
ln -s <vital_repository_dir>/.pre-commit-config.yaml .pre-commit-config.yaml
ln -s <vital_repository_dir>/pyproject.toml pyproject.toml
ln -s <vital_repository_dir>/setup.cfg setup.cfg

# create an isort configuration for the project from the vital configuration
cp <vital_repository_dir>/.isort.cfg isort.cfg
# Afterwards, edit the copied configuration file manually to change the value
# of the `known_first_party` tag to the name of the project package
```
