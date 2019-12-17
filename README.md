# VITAL

Welcome to the git repo of the [Videos & Images Theory and Analytics Laboratory (VITAL)](http://vital.dinf.usherbrooke.ca/ "VITAL home page")
of Sherbrooke University, headed by Professor [Pierre-Marc Jodoin](http://info.usherbrooke.ca/pmjodoin/).

## How to use

This repository was not designed to be used as a standalone project, but was rather meant to be included inside other
repositories. The recommended way for including this repository is as a [git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
inside the project from which you want to use it.

## Description

The `metrics` directory contains common metrics that are not part of the traditional libraries, whether those metrics
are losses for training (see `metrics/train`) or scores to evaluate the systems' performance (see `metrics/evaluate`).

The `modules` directory contains generic models, organized by task (e.g. `classification`, `generative`, etc.).

The `utils` directory contains a wide range of common utilities that may be used by several models
(e.g. image processing utilities, parameter groups definitions, etc.).

## Requirements

The `vital.yml` file lists the dependencies required by the whole repository. In case you include the repository inside
your own project, you may want to add project specific dependencies, or maybe even remove some dependencies (if you only
use some of the utilities provided by the repository).

