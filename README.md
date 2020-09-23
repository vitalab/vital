<div align="center">

# VITAL

Welcome to the repo of the
[Videos & Images Theory and Analytics Laboratory (VITAL)](http://vital.dinf.usherbrooke.ca/ "VITAL home page") of
Sherbrooke University, headed by Professor [Pierre-Marc Jodoin](http://info.usherbrooke.ca/pmjodoin/)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Check Code Format](https://github.com/nathanpainchaud/vital/workflows/Check%20Code%20Format/badge.svg)](https://github.com/nathanpainchaud/vital/actions?query=workflow%3A%22Check+Code+Format%22)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/nathanpainchaud/vital/blob/dev/LICENSE)

</div>

## Description
This repository was not designed to be used as a standalone project, but was rather meant to be used as a third-party
library for more applied projects.

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

- [VitalRunner](vital/vital_runner.py): common boilerplate code surrounding the use of Lightning's `Trainer` that
handles a generic train and eval run of a model.

## How to use

### Install
To install the project, run the following command from the project's root directory:
```shell script
pip install .
```
> NOTE: This instruction applies when you only want to use the project. If you want to play around with the code and
> contribute to the project, see [the section on how to contribute](#how-to-contribute).

## How to Contribute

### Environment Setup
If you want to contribute to the project, you must include it differently in your python environment. Once again, it is
recommended to use pip to install the project. However, this time the project should be installed in editable mode, with
the required additional development dependencies:
```shell script
pip install -e .[dev]
```

### Version Control Hooks
Before first trying to commit to the project, it is important to setup the version control hooks, so that commits
respect the coding standards in place for the project. The [`.pre-commit-config.yaml`](.pre-commit-config.yaml) file
defines the pre-commit hooks that should be installed in any project contributing to the `vital` repository. To setup
the version control hooks, run the following command:
```shell script
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
