<div align="center">

# VITAL

Welcome to the repo of the
[Videos & Images Theory and Analytics Laboratory (VITAL)](http://vital.dinf.usherbrooke.ca/ "VITAL home page") of
Sherbrooke University, headed by Professor [Pierre-Marc Jodoin](http://info.usherbrooke.ca/pmjodoin/)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Check Code Format](https://github.com/nathanpainchaud/vital/workflows/Check%20Code%20Format/badge.svg)](https://github.com/nathanpainchaud/vital/actions?query=workflow%3A%22Check+Code+Format%22)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/nathanpainchaud/vital/blob/dev/LICENSE)

</div>

## How to use
This repository was not designed to be used as a standalone project, but was rather meant to be used as a third-party
library for more applied projects.

## Description
To help you follow along with the organization of the repository, here is a summary of each major package's purpose:

- [data](vital/data): utilities to process and interface with common medical image datasets, from processing raw image
files (e.g. `.mhd` or `nii.gz`) to implementations of torchvision's `VisionDataset`.

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
The [`vital.yml`](requirements/vital.yml) file lists the dependencies required by the whole repository. In case you
include the repository inside your own project, you may want to add project specific dependencies, or maybe even remove
some dependencies (if you only use some of the utilities provided by the repository).


## How to contribute

### Version Control Hooks
Before first trying to commit to the project, it is important to setup the version control hooks, so that commits
respect the coding standards in place for the project. The [`.pre-commit-config.yaml`](.pre-commit-config.yaml) file
defines the pre-commit hooks that should be installed in any project contributing to the `vital` repository. To setup
the version control hooks, run the following command:
```
pre-commit install
```

> NOTE: In case you want to copy the pre-commit hooks configuration to your own project, you're welcome to :)
> The configuration file for each hook is located in the following files:
> - [isort](https://github.com/timothycrosley/isort): [`pyproject.toml`](./pyproject.toml), `[tool.isort]` section
> - [black](https://github.com/psf/black): [`pyproject.toml`](./pyproject.toml), `[tool.black]` section
> - [flake8](https://gitlab.com/pycqa/flake8): [`setup.cfg`](./setup.cfg), `[flake8]` section
>
> However, be advised that `isort` must be configured slightly differently in each project. The `src_paths` tag
> should thus reflect the package directory name of the current project, in place of `vital`.
