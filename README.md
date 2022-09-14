<div align="center">

# VITAL

Welcome to the repo of the
[Videos & Images Theory and Analytics Laboratory (VITAL)](http://vital.dinf.usherbrooke.ca/ "VITAL home page") of
Sherbrooke University, headed by Professor [Pierre-Marc Jodoin](http://info.usherbrooke.ca/pmjodoin/)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI: Code Format](https://github.com/vitalab/vital/actions/workflows/code-format.yml/badge.svg?branch=dev)](https://github.com/vitalab/vital/actions/workflows/code-format.yml?query=branch%3Adev)

[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/vitalab/vital/blob/dev/LICENSE)

</div>

## Description
This repository was not designed to be used as a standalone project or a template for full-fledged projects, but is
rather meant to be used as a third-party library for more applied projects.

To help you follow along with the organization of the repository, here is a summary of each major package's purpose:

- [data](vital/data): utilities to process and interface with common medical image datasets, from processing raw image
files (e.g. `.mhd` or `nii.gz`) to implementations of torchvision's `VisionDataset`.

- [metrics](vital/metrics): common metrics that are not part of the traditional libraries, whether those metrics are
losses for training (see [train](vital/metrics/train)) or scores to evaluate the systems' performance (see
[evaluate](vital/metrics/evaluate)).

- [models](vital/models): generic models, organized by task (e.g. [classification](vital/models/segmentation),
[generative](vital/models/generative), etc.).

- [results](vital/results): generic utilities for processing results during the evaluation phase.

- [tasks](vital/tasks): common boilerplate Lightning module code to train architectures for specific tasks (e.g.
[classfication](vital/tasks/classification.py), [segmentation](vital/tasks/segmentation.py), etc.).

- [utils](vital/utils): a wide range of common utilities that may be used in multiple other packages (e.g.
[logging](vital/utils/logging.py), [image processing](vital/utils/image), etc.).

- [`VitalRunner`](vital/runner.py): common boilerplate code surrounding the use of Lightning's `Trainer` that
handles a generic train and eval run of a model.

## How to use

### Install
To install the project, run the following command from the project's root directory:
```shell script
pip install .
```
> **Note**
> These instructions apply when you only want to use the project. If you want to edit the code or contribute to the
> project, refer to [the section on how to contribute](#how-to-contribute).

### Configuring a run
This project uses Hydra to handle the configuration of the [`VitalRunner`](vital/runner.py) entry point. To understand
how to use Hydra's CLI, refer to its [documentation](https://hydra.cc/docs/intro/). For this particular project,
presets of configuration options for various parts of the `VitalRunner` pipeline are available in the
[config package](vital/config). These files are meant to be composed together by Hydra to produce a complete
configuration for a run.

For a concrete example of how to launch a run using the Hydra CLI, let's say we wanted to train an MLP for
classification on the MNIST dataset using the preset configuration [`mnist-mlp`](vital/config/experiment/mnist-mlp.yaml),
but with otherwise default options. Assuming we were working from the repo's root directory, then the command would
simply be:
```bash
# Run the training
python vital/runner.py +experiment=mnist-mlp

# Output the config that would have been used, without actually running the code (useful for debugging)
python vital/runner.py +experiment=mnist-mlp --cfg job
```

### Tracking experiments
By default, Lightning logs runs locally in a format interpretable by
[Tensorboard](https://www.tensorflow.org/tensorboard/).

Another option is to use [Comet](https://www.comet.ml/) to log experiments, either online or offline. To enable the
tracking of experiments using Comet, simply use one of the pre-built Hydra configuration for Comet. The default
configuration is for Comet in `online` mode, but you can use it in `offline` mode by selecting the corresponding config
file when launching the [VitalRunner](vital/runner.py):
```bash
python vital/runner.py logger=comet/offline ...
```
To configure the Comet API and experiment's metadata, Comet relies on either i) environment variables (which you can set
in a `.env` that will automatically be loaded using `python-dotenv`) or ii) a [`.comet.config`](.comet.config) file. For
more information on how to configure Comet using environment variables or the config file, refer to
[Comet's configuration variables documentation](https://www.comet.ml/docs/python-sdk/advanced/#comet-configuration-variables).

An example of a `.comet.config` file, with the appropriate fields to track experiments online, can be found
[here](.comet.config). You can simply copy the file to the directory of your choice within your project (be sure
not to commit your Comet API key!!!) and fill the values with your own Comet credentials and workspace setup.

> **Note**
> No change to the code is necessary to change how the `CometLogger` handles the configuration from the `.comet.config`
> file. The code simply reads the content of the `[comet]` section of the file and uses it to create a `CometLogger`
> instance. That way, you simply have to ensure that the fields present in your configuration match the behavior you
> want from the `CometLogger` integration in Lighting, and you're good to go!

## How to Contribute

### Environment Setup
If you want to contribute to the project, you must install it differently in your python environment. This time, it is
recommended to use an environment where [`poetry`](https://python-poetry.org/) is available, since it is easier to
install the project in development mode using `poetry`. Assuming you're working in a virtual environment where
[`poetry` is installed](https://python-poetry.org/docs/#installation), you can simply run the command:
```shell script
poetry install
```
from the project's root directory to install it in editable mode, along with its regular and development dependencies.

### Version Control Hooks
Before first trying to commit to the project, it is important to setup the version control hooks, so that commits
respect the coding standards in place for the project. The [`.pre-commit-config.yaml`](.pre-commit-config.yaml) file
defines the pre-commit hooks that should be installed in any project contributing to the `vital` repository. To setup
the version control hooks, run the following command:
```shell script
pre-commit install
```

> **Note**
> In case you want to copy the pre-commit hooks configuration to your own project, you're welcome to :)
> The configuration for each hook is located in the following files:
> - [isort](https://github.com/timothycrosley/isort): [`pyproject.toml`](./pyproject.toml), `[tool.isort]` section
> - [black](https://github.com/psf/black): [`pyproject.toml`](./pyproject.toml), `[tool.black]` section
> - [flake8](https://gitlab.com/pycqa/flake8): [`setup.cfg`](./setup.cfg), `[flake8]` section
>
> However, be advised that `isort` must be configured slightly differently in each project. The `src_paths` tag
> should thus reflect the package directory name of the current project, in place of `vital`.
